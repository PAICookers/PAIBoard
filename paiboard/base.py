import atexit
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, overload

import numpy as np
from paicorelib import ChipCoord, CoordLike, OfflineFrameGen, to_coord, to_coords
from paicorelib.framelib import PAYLOAD_DATA_DTYPE, FrameArrayType
from paicorelib.framelib.frame_defs import FF
from paicorelib.framelib.frame_defs import FrameHeader as FH

from .board_cfg import ChipSOMType
from .common import HostCtrlInterface, RegFile
from .exceptions import (
    PAIBoardDeviceError,
    PAIBoardFileNotFoundError,
    PAIBoardOptionalFileMissingWarning,
    PAIBoardRuntimeWarning,
)
from .global_cfg import (
    CFG_FILE_DTYPE,
    DEFAULT_FNAME_CFG_FILE_WO_SUFFIX,
    DEFAULT_FNAME_CORE_PARAMS_CONF,
    DEFAULT_FNAME_GRAPH_INFO,
    DEFAULT_FNAME_INPUT_NODE_INFO,
    DEFAULT_FNAME_LEARNING_MODE_DIS_CFG_FILE_WO_SUFFIX,
    DEFAULT_FNAME_LEARNING_MODE_EN_CFG_FILE_WO_SUFFIX,
    DEFAULT_FNAME_NEURON_PHY_LOC,
    DEFAULT_FNAME_OUTPUT_DEST_INFO,
)
from .runtime import PAIRuntime
from .runtime.types import (
    ChipCoordStr,
    CoordStr,
    InputNodeAttrsMap,
    NeuPhyLocMap,
    NodeName,
    OutputDestAttrsMap,
    coordstr2coord,
    get_n_timeslot_max,
    is_node_online,
)
from .types import (
    InferenceMode,
    InputMappingAnyType,
    InputMappingType,
    NeuVoltageMappingType,
    OutputMappingType,
    PayloadDataType,
    VoltageType,
)
from .utils import check_requirements, time_it
from .zynq import ZynqClient

__all__ = ["PAIBoard"]


class PAIBoard:
    """
    Base PAIBoard platform, handling configuration, inference, encoding, decoding &
    communication with the chip or chip array.
    """

    chip_som_type: ClassVar[ChipSOMType]
    working_dir: Path
    intf: HostCtrlInterface
    target_chip_list: list[ChipCoord]
    source_chip: ChipCoord

    # Variables in runtime, specific to each input node
    _infer_mode: InferenceMode
    _infer_encoding_ts: int
    _infer_decoding_ts: int
    _batch_size: int
    _chunk_ts: int
    _total_batch: int
    _infer_ts: int
    _infer_ts_forward: int
    """Current timestep after initialization"""

    _debug_mode: bool
    """Axuliary flag for debugging"""

    def __init__(
        self,
        toolchain_build_dir: Path | str,
        timestep: int,
        n_layer: int,
        *,
        batch_mode: bool = False,
        excluded_init_chips: list[CoordLike] | None = None,
        neu_vol_reading_mode: Literal["contiguous", "onebyone"] = "contiguous",
        debug_mode: bool = False,
    ) -> None:
        """Initialize PAIBoard with model parameters and configuration.

        Args:
            toolchain_build_dir (Path, str): directory containing toolchain output files.
            timestep (int): number of simulation timesteps.
            n_layer (int): number of layers in the neural network.
            batch_mode (bool): whether to use batch mode. If True, the shape of input data should be (batch_size, timestep, size).
            excluded_init_chips (list[CoordLike], optional): list of chip coordinates to exclude from initialization.
            neu_vol_reading_mode ("contiguous", "onebyone"): reading mode of the voltage data from the chip.
            debug_mode (bool): whether to enable debug mode for performance timing.
        """
        if not (p := Path(toolchain_build_dir)).is_dir():
            raise ValueError(f"{p} is not a directory")

        self.working_dir = p
        self._debug_mode = debug_mode

        assert timestep > 0
        assert n_layer >= 0
        self.timestep = timestep
        self.n_layer = n_layer

        # Load & parse graph info
        self._load_and_parse_graph_info()

        # Determine the chip of source signals
        self.source_chip = self._get_source_chip(self.target_chip_list)

        # Detect configuration file path, but load the frames later.
        self.cfg_fp = self._auto_detect_cfg_file()
        # Load & parse i/o info files
        self._load_and_parse_input_node_info()
        self._load_and_parse_output_dest_info()
        self._assert_single_io_node()

        # Load configuration files for switching the working mode of online cores
        self._load_learning_mode_switch_cfg_files()

        # Determine the inference mode
        self.batch_mode = batch_mode
        self._make_inference_mode(batch_mode)
        self._make_rtcfg_map()

        # Load & parse neuron physical location info file (optional)
        self._load_and_parse_neu_phy_loc_info(neu_vol_reading_mode)

        # Check the timestep <= the depth of the input buffer
        self._valid_timestep_limit()

        # Pre-generate frames before inference that can be reused
        if excluded_init_chips is None:
            self.excluded_init_chips = []
        else:
            self.excluded_init_chips = to_coords(excluded_init_chips)

        self.init_frames = self._pre_gen_init_frames(self.excluded_init_chips)
        # Cache the sync frames
        # Only need ts+layer-1 timesteps to retrieve ts valid outputs
        self.cache_n_sync = self.timestep + self.n_layer - 1
        self.cache_sync_frames = self._gen_sync(self.cache_n_sync)

        # Hardware-related operations. Call after the device is initialized & running.
        if not self.intf.running:
            raise PAIBoardDeviceError(
                f"the device of {self.__class__.__name__} is not running."
            )

        atexit.register(self.close)

    def reset(self) -> None:
        """Reset the chip(s) & the regfile."""
        self.intf.reset_chip()
        self.intf.reset_regfile()

    def close(self, quit_the_server: bool = False) -> None:
        if not self.intf.running:
            return

        if isinstance(self.intf, ZynqClient):
            self.intf.close(quit_the_server)  # Only zynq client has quit()
        else:
            self.intf.close()

    def _load_and_parse_graph_info(self) -> None:
        if not (p := self.working_dir / DEFAULT_FNAME_GRAPH_INFO).exists():
            warnings.warn(
                f"missing optional graph info file: {p}",
                PAIBoardOptionalFileMissingWarning,
            )

        with p.open("r") as f:
            self.graph_metadata: dict[str, Any] = json.load(f)

        self.clk_en_l2_dict: dict[ChipCoord, list[int]] = dict()
        self.target_chip_list = []

        if misc := self.graph_metadata.get("misc"):
            # If info exists, parse clk_en_L2 & target_chip_list
            if chip_lst := misc.get("target_chip_list"):
                self.target_chip_list: list[ChipCoord] = []
                for c in chip_lst:
                    if isinstance(c, int):
                        self.target_chip_list.append(ChipCoord.from_addr(c))
                    elif isinstance(c, dict):
                        self.target_chip_list.append(ChipCoord(c["x"], c["y"]))
                    else:
                        raise TypeError(f"invalid chip coordinate: {c}")

            if d := misc.get("clk_en_L2"):
                for k, v in d.items():
                    self.clk_en_l2_dict[coordstr2coord(k)] = v

    def _get_source_chip(self, target_chips: list[ChipCoord]) -> ChipCoord:
        """Get the signal source chip."""
        if len(target_chips) == 0:
            # Try to use core parameters file to get source chip
            return self._parse_source_chip_by_core_params()

        # XXX May be defined for other SOM types in the future
        return target_chips[0]
        # if self.chip_som_type == ChipSOMType.SINGLE_BONDING:
        #     return target_chips[0]
        # else:  # Array 2x2
        #     return target_chips[0]

    def _parse_source_chip_by_core_params(self) -> ChipCoord:
        """Parse source chip coordinates from core parameters file."""
        if not (p := self.working_dir / DEFAULT_FNAME_CORE_PARAMS_CONF).exists():
            raise PAIBoardFileNotFoundError(
                f"necessary core parameters file not found: {p}"
            )

        with p.open("r") as f:
            core_params = json.load(f)

        # The first chip coordinate in the 'core_params.json' is the source of global signals
        return coordstr2coord(next(iter(core_params)))

    def _auto_detect_cfg_file(self) -> Path:
        """Auto-detect the config file with suffix `.npy`, `.bin`, or `.txt`."""
        fp_wo_suffix = self.working_dir / DEFAULT_FNAME_CFG_FILE_WO_SUFFIX
        for ext in [".npy", ".bin", ".txt"]:
            if (p := fp_wo_suffix.with_suffix(ext)).exists():
                return p

        raise PAIBoardFileNotFoundError(
            f"necessary config file not found: {DEFAULT_FNAME_CFG_FILE_WO_SUFFIX}"
        )

    def _load_learning_mode_switch_cfg_files(self) -> None:
        """Auto-detect & load the config files with suffix `.npy`, `.bin`, or `.txt` for switching the working mode of online cores .

        NOTE: only used if the deployed network includes online cores for STDP learning.
        """
        en_learning_fp_wo_suffix = (
            self.working_dir / DEFAULT_FNAME_LEARNING_MODE_EN_CFG_FILE_WO_SUFFIX
        )
        dis_learning_fp_wo_suffix = (
            self.working_dir / DEFAULT_FNAME_LEARNING_MODE_DIS_CFG_FILE_WO_SUFFIX
        )
        self.en_learning_frames = None
        self.dis_learning_frames = None

        for ext in [".npy", ".bin", ".txt"]:
            if (p := en_learning_fp_wo_suffix.with_suffix(ext)).exists():
                self.en_learning_frames = np.fromfile(p, dtype=CFG_FILE_DTYPE)
                break

        for ext in [".npy", ".bin", ".txt"]:
            if (p := dis_learning_fp_wo_suffix.with_suffix(ext)).exists():
                self.dis_learning_frames = np.fromfile(p, dtype=CFG_FILE_DTYPE)
                break

    def _valid_timestep_limit(self) -> None:
        node_rtcfg = self.input_rtcfg_map[list(self.inode_attrs_map.keys())[0]]
        MAX_TIMESLOT = get_n_timeslot_max(node_rtcfg.is_online)

        # TODO Maybe for input nodes, there is no timestep limit. But for output nodes?
        if self.timestep > MAX_TIMESLOT:
            raise ValueError(
                f"the timestep out of range: {self.timestep} > {MAX_TIMESLOT}"
            )

    def _pre_gen_init_frames(self, excluded_chips: list[ChipCoord]) -> FrameArrayType:
        """Pre-generate frame bundles required for repeated runtime operations."""
        if not (p := self.working_dir / DEFAULT_FNAME_CORE_PARAMS_CONF).exists():
            raise PAIBoardFileNotFoundError(
                f"necessary core parameters file not found: {p}"
            )

        with p.open("r") as f:
            core_params: dict[ChipCoordStr, dict[CoordStr, Any]] = json.load(f)

        return PAIRuntime.gen_init_frame(core_params, exclude=excluded_chips)

    def _load_and_parse_input_node_info(self) -> None:
        """Parse input node information from JSON file & generate frames convenient for encoding."""
        if not (p := self.working_dir / DEFAULT_FNAME_INPUT_NODE_INFO).exists():
            raise PAIBoardFileNotFoundError(
                f"necessary input node info file not found: {p}"
            )

        with p.open("r") as f:
            self.inode_attrs_map: InputNodeAttrsMap = json.load(f)

    def _make_inference_mode(self, batch_mode: bool) -> None:
        """Determine the chunk timestep by given input nodes info."""
        # XXX Only one single input node is supported
        node_attr = self.inode_attrs_map[list(self.inode_attrs_map.keys())[0]]
        MAX_TIMESLOT = get_n_timeslot_max(is_node_online(node_attr))

        # Get the maximum LCN required for all input nodes
        lcn_max = node_attr.get("lcn", 0)
        if lcn_max > MAX_TIMESLOT:
            raise ValueError(
                f"max LCN ({lcn_max}) exceeds maximum allowed timeslots({MAX_TIMESLOT})."
            )

        chunk_ts = self.timestep  # chunk_ts <= timestep
        while lcn_max * chunk_ts > MAX_TIMESLOT:
            chunk_ts >>= 1
            if chunk_ts < 1:
                chunk_ts = 1
                break

        self._chunk_ts = chunk_ts

        no_chunk = chunk_ts == self.timestep
        if not batch_mode:
            self._batch_size = 1
            if no_chunk:
                self._infer_mode = InferenceMode.NO_BATCH_NO_CHUNK
            else:
                self._infer_mode = InferenceMode.NO_BATCH_WITH_CHUNK
        else:
            if no_chunk:
                # can process multiple batches
                self._batch_size = MAX_TIMESLOT // (lcn_max * chunk_ts)
                self._infer_mode = InferenceMode.WITH_BATCH_NO_CHUNK
            else:
                self._batch_size = 1
                self._infer_mode = InferenceMode.WITH_BATCH_WITH_CHUNK

    def _load_and_parse_output_dest_info(self) -> None:
        """Parse output destination information from JSON file & generate output frames convenient for decoding."""
        if not (p := self.working_dir / DEFAULT_FNAME_OUTPUT_DEST_INFO).exists():
            raise PAIBoardFileNotFoundError(
                f"necessary output destination info file not found: {p}"
            )

        with p.open("r") as f:
            self.onode_attrs_map: OutputDestAttrsMap = json.load(f)

    def _assert_single_io_node(self) -> None:
        # XXX Only one single input & output node is supported for now
        if len(self.inode_attrs_map) > 1:
            raise NotImplementedError(
                f"only support single input node, but got {len(self.inode_attrs_map)}"
            )
        if len(self.onode_attrs_map) > 1:
            raise NotImplementedError(
                f"only support single output node, but got {len(self.onode_attrs_map)}"
            )

    def _load_and_parse_neu_phy_loc_info(
        self, reading_mode: Literal["contiguous", "onebyone"] = "contiguous"
    ) -> None:
        if not (p := self.working_dir / DEFAULT_FNAME_NEURON_PHY_LOC).exists():
            warnings.warn(
                f"missing optional neuron physical location info file: {p}",
                PAIBoardOptionalFileMissingWarning,
            )
            return

        with p.open("r") as f:
            self.neu_phy_locs_map: dict[NodeName, NeuPhyLocMap] = json.load(f)

        # Prepare frames for neuron voltage reading
        self.neu_vol_reading_mode: Literal["contiguous", "onebyone"] = reading_mode

        info: dict[str, list[tuple[FrameArrayType, int]]] = dict()
        for neu, phy_loc in self.neu_phy_locs_map.items():
            info[neu] = PAIRuntime.gen_read_neuron_attrs_frames(
                phy_loc, self.neu_vol_reading_mode
            )

        self.neu_vol_reading_frames = info

    def _make_rtcfg_map(self) -> None:
        # NO_BATCH_NO_CHUNK: ts(cts)
        # NO_BATCH_WITH_CHUNK: cts
        # WITH_BATCH_NO_CHUNK: bs*ts(cts)
        # WITH_BATCH_WITH_CHUNK: bs(1)*cts
        # The timestep of the input encoding <= MAX. the chunk timestep ensures the input data no more than MAX.
        self._infer_encoding_ts = self._batch_size * self._chunk_ts
        self.input_rtcfg_map = PAIRuntime.gen_input_rtcfg_map(
            self._infer_encoding_ts, self.inode_attrs_map
        )

        # NO_BATCH_NO_CHUNK: ts(cts)
        # NO_BATCH_WITH_CHUNK: ts
        # WITH_BATCH_NO_CHUNK: bs*ts(cts)
        # WITH_BATCH_WITH_CHUNK: bs(1)*ts
        # The decoded output must in shape (batch_size, timestep, size)(if batch mode is enabled)
        self._infer_decoding_ts = self._batch_size * self.timestep
        self.output_rtcfg_map = PAIRuntime.gen_output_rtcfg_map(
            self._infer_decoding_ts, self.onode_attrs_map
        )

    def _check_prerequisites(
        self, check_before_inference: bool = False, check_before_reading_v: bool = False
    ) -> None:
        if check_before_inference:
            assert bool(self.inode_attrs_map)
            assert bool(self.input_rtcfg_map)
            assert bool(self.onode_attrs_map)
            assert bool(self.output_rtcfg_map)

        if check_before_reading_v:
            assert bool(self.neu_phy_locs_map)
            assert bool(self.neu_vol_reading_frames)

    def _gen_sync(self, n_sync: int) -> FrameArrayType:
        """Regenerate sync frames based on runtime requirements."""
        return PAIRuntime.gen_sync_frame(n_sync, self.source_chip)

    def set_n_max_oframe(self, value: int | None = None) -> None:
        """Set the maximum number of output frames to be received from the chip.

        Args:
            n_max_oframe: maximum number of output frames, uint32.
        """
        self.intf.set_n_max_oframe(value)

    def chip_hw_init(self) -> None:
        """Initialize the chip with init signal & init frames.

        NOTE: due to the defect of the chip, must set GPIO INIT & then send init frames.
        """
        v = 1 << RegFile.CHIP_TOP_INIT_BIT
        self.intf.write_reg(RegFile.CHIP_TOP_CTRL, v)
        self.intf.send_frames(self.init_frames)
        self.intf.write_reg(RegFile.CHIP_TOP_CTRL, 0)

    def chip_hw_model_download(self, **kwargs) -> None:
        """Download configuration frames to the chip."""
        cfg_frames = np.fromfile(self.cfg_fp, dtype=CFG_FILE_DTYPE)

        print("----------------------------------")
        print("----------PAICORE CONFIG----------")
        size = self.intf.send_frames(cfg_frames, **kwargs)
        if size != cfg_frames.nbytes:
            raise RuntimeError(
                f"expected send: {cfg_frames.nbytes} bytes, but send: {size}"
            )

        print("----------------------------------")

    def chip_hw_config(self, **kwargs) -> None:
        """Configure the chip(s), downloading config frames to the chip(s)."""
        return self.chip_hw_model_download(**kwargs)

    def chip_hw_send(self, payload: FrameArrayType, **kwargs) -> int:
        return self.intf.send_frames(payload, **kwargs)

    def chip_hw_inference(
        self,
        init: bool,
        input_frames: FrameArrayType,
        sync_frames: FrameArrayType,
        infer_step_by_step: bool = False,
        *,
        recv_max_size: int | None = None,
        **kwargs,
    ) -> FrameArrayType:
        """Perform inference operation on the chip.

        Args:
            work_and_sync_payload: Input working frames & sync frames for inference
            recv_size: Size of the output frames to receive
            init: Whether to initialize the chip before inference

        Returns:
            Received output frames from the chip
        """
        if init:
            self.chip_hw_init()

        if infer_step_by_step:
            self.chip_hw_send(input_frames, **kwargs)
            return self.intf.send_and_recv_frames(sync_frames, recv_max_size, **kwargs)
        else:
            payload = np.concatenate([input_frames, sync_frames])
            return self.intf.send_and_recv_frames(payload, recv_max_size, **kwargs)

    def chip_hw_sync(self, n: int = 1) -> None:
        """Synchronize the chip operations."""
        sync_frames = self._gen_sync(n)
        self.intf.send_frames(sync_frames)

    def chip_hw_clear(self, *chip_coords: CoordLike) -> None:
        """Clear specified chips."""
        wf3 = []
        for coord in chip_coords:
            wf3.append(OfflineFrameGen.gen_work_frame3(to_coord(coord)))

        clear_frames = np.concatenate([f.value for f in wf3])
        self.intf.send_frames(clear_frames)

    def chip_hw_status(self) -> None:
        """Display current chip status. The chip status is described in the registers of the DMA."""
        self.intf.get_regfile_status()

    def _multi_inputs_mapping(self, inputs: InputMappingAnyType) -> InputMappingType:
        inode_names = list(self.inode_attrs_map.keys())
        n_inode = len(inode_names)
        imap = InputMappingType()  # Only record valid inputs

        if isinstance(inputs, np.ndarray):
            if n_inode > 1:
                warnings.warn(
                    "not all inputs are mapped to input nodes. Other inputs will be set to 0"
                )

            imap[inode_names[0]] = inputs
        elif isinstance(inputs, Sequence):
            if len(inputs) == 0:
                raise ValueError("no inputs")
            if len(inputs) > n_inode:
                raise ValueError("too many inputs")
            if len(inputs) < n_inode:
                warnings.warn(
                    "not all inputs are mapped to input nodes, other inputs will be set to 0"
                )

            for inp, inode in zip(inputs, inode_names, strict=False):
                imap[inode] = inp
        else:
            for inp in inputs:
                if inp not in self.inode_attrs_map:
                    raise ValueError(f"input node '{inp}' not found")

                imap[inp] = inputs[inp]

        return imap

    def _reshape_inputs_and_get_batch(
        self, imap: InputMappingType
    ) -> tuple[InputMappingType, int]:
        """Always reshape inputs to (actual batch_size, timestep, size)."""
        if not self.batch_mode:
            n_batch = 1
        else:
            # check the batches of each input are equal
            _batches = [inp.shape[0] for inp in imap.values()]
            if len(set(_batches)) > 1:
                raise ValueError(
                    f"inputs have different batch sizes: {_batches}, but expected equal batch size"
                )

            n_batch = _batches[0]

        for name, inp in imap.items():
            if not self.batch_mode:
                if inp.ndim < 2:
                    raise ValueError(
                        f"input '{name}' has shape {inp.shape}, but expected at least 2d array when 'batch_mode' is disabled"
                    )
                elif inp.shape[0] != self.timestep:
                    raise ValueError(
                        f"input '{name}' has shape {inp.shape}, but expected first dimension to be timestep={self.timestep} when 'batch_mode' is disabled. If you want to use batch mode, please set 'batch_mode'"
                    )

                reshaped = inp.reshape(n_batch, self.timestep, -1)
                if inp.ndim > 2:
                    warnings.warn(
                        f"the data of input '{name}' has shape {inp.shape}, will be reshaped to {reshaped.shape}. If you want to use batch mode, please set 'batch_mode'",
                        PAIBoardRuntimeWarning,
                    )
            else:
                if inp.ndim < 3:
                    raise ValueError(
                        f"input '{name}' has shape {inp.shape}, but expected at least 3d array when 'batch_mode' is enabled"
                    )
                elif inp.shape[1] != self.timestep:
                    raise ValueError(
                        f"input '{name}' has shape {inp.shape}, but expected shape (batch_size, timestep, ...) - timestep dimension mismatch"
                    )

                reshaped = inp.reshape(n_batch, self.timestep, -1)
                if inp.ndim > 3:
                    warnings.warn(
                        f"the data of input '{name}' has shape {inp.shape}, will be reshaped to {reshaped.shape}",
                        PAIBoardRuntimeWarning,
                    )

            imap[name] = reshaped

        return imap, n_batch

    def _update_sync_cache(self, init: bool, n_batch: int) -> int:
        """Update the sync cache & return the number of timesteps to forward of this inference."""
        ts_forward = n_batch * self._chunk_ts
        if self._infer_mode == InferenceMode.WITH_BATCH_NO_CHUNK or init:
            # In the case, chunk_ts = timestep
            n_sync = ts_forward + self.n_layer - 1
        else:  # not init or in other modes
            n_sync = ts_forward

        if self.cache_n_sync != n_sync:
            self.cache_n_sync = n_sync
            self.cache_sync_frames = self._gen_sync(n_sync)

        return ts_forward

    @time_it
    def _encode_inputs(
        self,
        imap: InputMappingType,
        batch_idx: int,
        n_batch: int,
        ts_idx: int,
        n_ts: int,
    ) -> FrameArrayType:
        """Encode input data into frame templates."""
        encoded_frames = []

        for name, data in imap.items():
            # A batch may not be full, e.g. batch size is 64 but total batch in inference is 16.
            data = data[batch_idx : batch_idx + n_batch, ts_idx : ts_idx + n_ts]
            encoded = PAIRuntime.encode(
                data,
                self.input_rtcfg_map[name].template[: data.size],
                is_dest_online=self.input_rtcfg_map[name].is_online,
            )
            encoded_frames.append(encoded)

        return np.concatenate(encoded_frames)

    def _outputs_chunk_ts_concat(
        self, omap: list[OutputMappingType]
    ) -> OutputMappingType:
        """Return the complete output in the timestep dimension."""
        accumulated = OutputMappingType()
        for name in self.onode_attrs_map:
            if len(omap) == 1:
                output = omap[0][name]
            else:
                arr = [o[name] for o in omap]
                output = np.sum(arr, axis=0)

            accumulated[name] = np.asarray(output, dtype=PAYLOAD_DATA_DTYPE)

        return accumulated

    def _outputs_batch_concat(self, omap: list[OutputMappingType]) -> OutputMappingType:
        """Stack the output in the batch dimension."""
        concat = OutputMappingType()
        for name in self.onode_attrs_map:
            # Reshape & concatenate the output in the batch dimension.
            # ------- bs[0] ------- | ------- bs[1] ------- | ------- bs[2] ------- |
            # ts[0][1][2][3] -- [7] | ts[0][1][2][3] -- [7] | ts[0][1][2][3] -- [7] |
            if len(omap) == 1:
                output = omap[0][name].reshape(self._batch_size, self.timestep, -1)
            else:
                arr_to_concat = []
                for o in omap:
                    arr_to_concat.append(
                        o[name].reshape(self._batch_size, self.timestep, -1)
                    )
                output = np.concatenate(arr_to_concat, dtype=PAYLOAD_DATA_DTYPE)

            # Only get the actual total batch as the final output
            concat[name] = output[: self._total_batch]

        return concat

    def _unwrap_onode(
        self, omap: OutputMappingType
    ) -> PayloadDataType | OutputMappingType:
        """Return the output payload data if there is only one output node, otherwise return the complete output mapping."""
        if len(self.onode_attrs_map) == 1:
            return omap[list(self.onode_attrs_map.keys())[0]]

        return omap

    @time_it
    def _inference(
        self,
        imap: InputMappingType,
        *,
        init: bool = True,
        batch_idx: int = 0,
        n_batch: int | None = None,
        ts_idx: int = 0,
        n_ts: int | None = None,
        recv_max_size: int | None = None,
        filter_output_strict: bool = True,
        decoding_raise_if_not_matched: bool = True,
        decoding_raise_if_duplicated: bool = True,
        infer_step_by_step: bool = False,
        **kwargs,
    ) -> OutputMappingType:
        """Run a single inference.

        Args:
            imap (InputMappingType): the input mapping.
            init (bool, optional): whether to initialize the chip. Default to True.
            batch_idx (int, optional): the batch index. Default to 0.
            n_batch (int, optional): the number of batches. Default to None.
            ts_idx (int, optional): the timestep index. Default to 0.
            n_ts (int, optional): the number of timesteps. Default to None.
        """
        if n_batch is None:
            n_batch = self._batch_size
        if n_ts is None:
            n_ts = self._chunk_ts

        # Update sync frames
        self._infer_ts_forward = self._update_sync_cache(init, n_batch)

        if init:
            self._infer_ts = 0

        input_frames = self._encode_inputs(imap, batch_idx, n_batch, ts_idx, n_ts)

        recv_frames_raw = self.chip_hw_inference(
            init,
            input_frames,
            self.cache_sync_frames,
            infer_step_by_step,
            recv_max_size=recv_max_size,
            **kwargs,
        )

        # Update the current inference timestep
        self._infer_ts += self._infer_ts_forward

        output = PAIRuntime.decode(
            recv_frames_raw,
            self.output_rtcfg_map,
            raise_if_has_other_type=filter_output_strict,
            raise_if_not_matched=decoding_raise_if_not_matched,
            raise_if_duplicated=decoding_raise_if_duplicated,
        )
        return output

    @time_it
    def _inference_no_chunk(
        self, imap: InputMappingType, recv_max_size: int | None, **kwargs
    ) -> PayloadDataType | OutputMappingType:
        """Inference with batch, without chunk."""
        o_batches: list[OutputMappingType] = []
        for b_idx in range(0, self._total_batch, self._batch_size):
            cur_n_batch = min(self._batch_size, self._total_batch - b_idx)

            o_batch = self._inference(
                imap,
                init=True,
                batch_idx=b_idx,
                n_batch=cur_n_batch,
                recv_max_size=recv_max_size,
                **kwargs,
            )

            o_batches.append(o_batch)

        if self._infer_mode == InferenceMode.NO_BATCH_NO_CHUNK:
            result = o_batches[0]
        else:
            result = self._outputs_batch_concat(o_batches)

        return self._unwrap_onode(result)

    @time_it
    def _inference_with_chunk(
        self, imap: InputMappingType, recv_max_size: int | None, **kwargs
    ) -> PayloadDataType | OutputMappingType:
        """Inference with batch."""
        o_batches: list[OutputMappingType] = []

        for b_idx in range(0, self._total_batch, self._batch_size):
            cur_n_batch = min(self._batch_size, self._total_batch - b_idx)
            o_chunks = []

            for t_idx in range(0, self.timestep, self._chunk_ts):
                cur_n_ts = min(self._chunk_ts, self.timestep - t_idx)

                o_chunk = self._inference(
                    imap,
                    init=(t_idx == 0),
                    batch_idx=b_idx,
                    n_batch=cur_n_batch,
                    ts_idx=t_idx,
                    n_ts=cur_n_ts,
                    recv_max_size=recv_max_size,
                    **kwargs,
                )
                o_chunks.append(o_chunk)

            o_batch = self._outputs_chunk_ts_concat(o_chunks)
            o_batches.append(o_batch)

        if self._infer_mode == InferenceMode.NO_BATCH_WITH_CHUNK:
            result = o_batches[0]
        else:
            result = self._outputs_batch_concat(o_batches)

        return self._unwrap_onode(result)

    @overload
    def inference(
        self,
        inputs: np.ndarray,
        recv_max_size: int | None = None,
        filter_output_strict: bool = True,
        decoding_check_matched: bool = True,
        decoding_check_duplicated: bool = True,
        infer_step_by_step: bool = False,
        **kwargs,
    ) -> PayloadDataType: ...

    @overload
    def inference(
        self,
        inputs: Sequence[np.ndarray] | InputMappingType,
        recv_max_size: int | None = None,
        filter_output_strict: bool = True,
        decoding_check_matched: bool = True,
        decoding_check_duplicated: bool = True,
        infer_step_by_step: bool = False,
        **kwargs,
    ) -> OutputMappingType: ...

    @time_it
    def inference(
        self,
        inputs: InputMappingAnyType,
        recv_max_size: int | None = None,
        filter_output_strict: bool = True,
        decoding_check_matched: bool = True,
        decoding_check_duplicated: bool = True,
        infer_step_by_step: bool = False,
        **kwargs,
    ) -> PayloadDataType | OutputMappingType:
        """Main entry point for running inference by given a single input, a sequnce of inputs or a dictionary of inputs.

        Args:
            inputs: input data for inference. It can be a single input, a sequence of inputs, or a dictionary of inputs.
            recv_max_size (int, optional): maximum size of received frames. Default to None, which will use the default     \
                value defined in `global_cfg.py`.
            filter_output_strict (bool): whether to strictly filter output frames. Default to true which will raise an      \
                exception if there are frames of other types.
            decoding_check_matched (bool): whether to check if the output frames are matched with the runtime configuration.\
                Default to true which will raise an exception if there are frames that are not matched. Disabling this check\
                will speed up the decoding.
            decoding_check_duplicated (bool): whether to check if the output frames are duplicated. Default to true which   \
                will raise an exception if there are frames that are duplicated. Disabling this check will speed up the     \
                decoding.
            infer_step_by_step (bool): whether to run chip inference step by step which is used to debug.

        Returns:
            One single output if there is only one output, or a dictionary of outputs if there are multiple outputs.

        NOTE: the signature of this function can be modified by its subclasses, passing more specified arguments by `kwargs`.
        """
        # Check prerequisites
        self._check_prerequisites(check_before_inference=True)

        # Parameters about checking
        kwargs.setdefault("filter_output_strict", filter_output_strict)
        kwargs.setdefault("infer_step_by_step", infer_step_by_step)
        kwargs.setdefault("decoding_raise_if_not_matched", decoding_check_matched)
        kwargs.setdefault("decoding_raise_if_duplicated", decoding_check_duplicated)

        imap = self._multi_inputs_mapping(inputs)
        imap, self._total_batch = self._reshape_inputs_and_get_batch(imap)

        if self._infer_mode.is_chunk_mode():
            return self._inference_with_chunk(imap, recv_max_size, **kwargs)
        else:
            return self._inference_no_chunk(imap, recv_max_size, **kwargs)

    def __call__(
        self, input: InputMappingAnyType, *args, **kwargs
    ) -> PayloadDataType | OutputMappingType:
        """Alias of `inference`."""
        return self.inference(input, *args, **kwargs)

    @overload
    def read_voltage_of_neuron(
        self,
        neuron: str,
        *,
        is_online: bool = False,
        weight_width: Literal[1, 2, 4, 8] = 8,
    ) -> VoltageType: ...

    @overload
    def read_voltage_of_neuron(
        self,
        neuron=None,
        *,
        is_online: bool = False,
        weight_width: Literal[1, 2, 4, 8] = 8,
    ) -> dict[NodeName, VoltageType]: ...

    @check_requirements("neu_phy_locs_map", DEFAULT_FNAME_NEURON_PHY_LOC)
    def read_voltage_of_neuron(
        self,
        neuron: str | None = None,
        *,
        is_online: bool = False,
        weight_width: Literal[1, 2, 4, 8] = 8,
    ) -> NeuVoltageMappingType:
        """Read the voltage of specified neurons. If no neuron is specified, read all defined in the neuron physical locations file.

        Args:
            neuron (str, None): name of the neuron to read. If None, read all neurons defined in the neuron physical locations file.
            is_online (bool): whether to read the voltage of online cores. Defaults to False.
            weight_width (Literal[1, 2, 4, 8]): width of the weight. Defaults to 8. Only valid when `is_online` is true.

        Returns:
            out (dict[str, list[tuple[FrameArrayType, int]]]): voltage of specified neurons.
        """
        self._check_prerequisites(check_before_reading_v=True)

        if isinstance(neuron, str):
            if neuron not in self.neu_vol_reading_frames:
                raise ValueError(
                    f"neuron '{neuron}' not found in neuron physical locations file."
                )

            otframe3 = []
            for tframes, n_package in self.neu_vol_reading_frames[neuron]:
                # Recv: start frame + #n of packages
                recv_frames = self.intf.send_and_recv_frames(tframes, 1 + n_package)
                otframe3.append(recv_frames)

            return PAIRuntime.decode_voltage(
                self.neu_phy_locs_map[neuron],
                otframe3,
                self.neu_vol_reading_mode,
                is_online=is_online,
                weight_width=weight_width,
            )

        # Read all neurons
        return {
            neu: self.read_voltage_of_neuron(
                neu, is_online=is_online, weight_width=weight_width
            )
            for neu in self.neu_vol_reading_frames
        }

    def learning_mode(self, enable: bool = True) -> None:
        """Switch the working mode for online cores. If `enable` is true, enable the learning mode.  \
            Otherwise, enbale the inference mode.
        """
        if enable:
            if self.en_learning_frames is None:
                raise RuntimeError(
                    "configuration file for enabling learning mode is not provided."
                )

            self.intf.send_frames(self.en_learning_frames)
        else:
            if self.dis_learning_frames is None:
                raise RuntimeError(
                    "configuration file for disabling learning mode is not provided."
                )

            self.intf.send_frames(self.dis_learning_frames)

    def inference_mode(self) -> None:
        """Enable the inference mode (or disabling the learning mode) for online cores."""
        return self.learning_mode(False)

    # Auxiliary functions.

    # def record_time(self, full_time) -> None:
    #     """
    #     Record timing information for performance measurement.

    #     Args:
    #         full_time: Full execution time
    #     """
    #     core_time = self.intf.read_reg(RegFile.US_TIME_TICK)
    #     record_time(core_time, full_time)

    # def perf(self, img_num) -> None:
    #     """
    #     Print performance statistics.

    #     Args:
    #         img_num: Number of images processed
    #     """
    #     print_time(img_num)


def filter_frame_type(
    frames: FrameArrayType, expect: FH = FH.WORK_TYPE1, strict: bool = False
) -> FrameArrayType:
    """filter the frames that are not the expected type. If strict is true, raise an error if there are any \
        frames that are not the expected type.
    """
    headers = (frames >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK

    if strict and np.any(headers != expect):
        raise ValueError(
            f"there are frames that are not the expected type. Expected type: {expect.name}({expect.value}), "
            f"but got: {headers[np.where(headers != expect)][0]}"
        )

    return frames[np.where(np.isin(headers, expect))]
