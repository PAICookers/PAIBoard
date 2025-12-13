from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast, overload, TYPE_CHECKING
import warnings
import numpy as np
import sys
from numpy.typing import ArrayLike, NDArray
from paicorelib import OffCoreCfg, OfflineFrameGen, OnlineFrameGen, ChipFrameGen
from paicorelib.coordinate import (
    ChipCoord,
    Coord,
    CoordLike,
    CoreType,
    to_coords,
    ReplicationId as RId,
)
from paicorelib.framelib import (
    OfflineWorkFrame1,
    OfflineTestInFrame3,
    OnlineWorkFrame1_1,
)
from paicorelib.framelib import OfflineTestOutFrame3 as Off_ToF3
from paicorelib.framelib.frame_defs import FrameFormat as FF
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import FramePackageType as FPType
from paicorelib.framelib.frame_defs import OfflineConfigFrame3Format as Off_NRAMF
from paicorelib.framelib.frame_defs import (
    OfflineWorkFrame1Format as Off_WF1F,
    OnlineWorkFrame1Format_1 as On_WF1F_1,
)
from paicorelib.framelib.frame_defs import OnlineConfigFrame3Format_WW1 as ON_NRAMF_WW1
from paicorelib.framelib.frame_defs import OnlineConfigFrame3Format_WWn as ON_NRAMF_WWn
from paicorelib.framelib.types import (
    PayloadDataType,
    PAYLOAD_DATA_DTYPE,
    FrameArrayType,
    FRAME_DTYPE,
)
from paicorelib.framelib.utils import framearray_header_check

from ..exceptions import PAIBoardWarning
from ..utils import time_it
from .types import (
    ChipCoordStr,
    CoordStr,
    CoreNeuSegLocMap,
    InputNodeAttrsMap,
    InputNodeRTCfg,
    InputNodeRTCfgMap,
    NeuPhyLocMap,
    OutputDestAttrs,
    OutputDestAttrsMap,
    NeuSegAddrAttrs,
    OutputDestRTCfg,
    OutputDestRTCfgMap,
    coordstr2coord,
    attrs2coord,
    get_n_timeslot_max,
)

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

if TYPE_CHECKING:
    from paiboard.types import OutputMappingType

__all__ = ["PAIRuntime"]

VOLTAGE_DTYPE = np.int32
VoltageType = NDArray[VOLTAGE_DTYPE]


# Use the key to represent the length extension factor of the output node.
LEN_EX_FACTOR_KEY = "tick_relative"
_RID_UNSET = RId(0, 0)


def max_timeslot_check(
    timestep: int, raw_ts: ArrayLike, is_online: bool = False
) -> int:
    max_raw_ts = np.max(raw_ts) + 1  # start from 0
    MAX_TIMESLOT = get_n_timeslot_max(is_online)

    if (max_ts := timestep * max_raw_ts) > MAX_TIMESLOT:
        raise ValueError(
            f"the maximum timeslot is out of range: {max_ts}({timestep}*{max_raw_ts}) > {MAX_TIMESLOT}"
        )

    return max_ts


def valid_indices_len_check(idx: list[int], recv_arr: np.ndarray) -> None:
    if len(idx) != len(recv_arr):
        raise ValueError(
            "length of valid indices & received array do not match: "
            f"{len(idx)} != {len(recv_arr)}"
        )


def get_len_ex_factor(
    onode_attrs: dict[CoordStr, OutputDestAttrs],
    key_name: str = LEN_EX_FACTOR_KEY,
) -> int:
    """Retrieve the length extension factor of the output node by the given attributes."""
    if not all(key_name in dest for dest in onode_attrs.values()):
        raise KeyError(f"key '{key_name}' not found in output destination attributes")

    return max(max(dest[key_name]) for dest in onode_attrs.values()) + 1


class PAIRuntime:
    work_fgen_handler = {
        CoreType.OFFLINE: OfflineWorkFrame1,
        CoreType.ONLINE: OnlineWorkFrame1_1,
    }

    @classmethod
    def parse_chip_coords(
        cls,
        core_params: dict[ChipCoordStr, dict[CoordStr, Any]],
    ) -> list[ChipCoord]:
        """Parse chip coordinates from core parameters dictionary."""
        chip_coords = []
        for chip_addr in core_params:
            chip_coord = coordstr2coord(chip_addr)
            chip_coords.append(chip_coord)

        return list(set(chip_coords))

    @classmethod
    def gen_init_frame(
        cls,
        core_params: dict[ChipCoordStr, dict[CoordStr, Any]],
        exclude: Sequence[CoordLike] | None = None,
    ) -> FrameArrayType:
        if exclude is not None:
            excluded_chips = to_coords(exclude)
        else:
            excluded_chips = []

        init_frames_list = []
        for chip_addr in core_params:
            chip_coord = coordstr2coord(chip_addr)
            if chip_coord in excluded_chips:
                continue

            for core_addr in core_params[chip_addr]:
                core_coord = coordstr2coord(core_addr)
                core_init_frame1, core_init_frame2 = ChipFrameGen.gen_magic_init_frame(
                    chip_coord, core_coord, redundant_init=False
                )
                init_frames_list.append(core_init_frame1)
                init_frames_list.append(core_init_frame2)

        return np.concatenate(init_frames_list, dtype=FRAME_DTYPE)

    @classmethod
    def gen_sync_frame(
        cls, n_sync: int, chip_coords: ChipCoord | Sequence[ChipCoord]
    ) -> FrameArrayType:
        # Both for online & offline cores
        if isinstance(chip_coords, ChipCoord):
            coords = (chip_coords,)
        else:
            coords = chip_coords

        return np.concatenate(
            [OfflineFrameGen.gen_work_frame2(coord, n_sync).value for coord in coords]
        )

    @classmethod
    @time_it
    def encode(
        cls,
        data: ArrayLike,
        icfg_template: FrameArrayType,
        repeat: int = 1,
        *,
        is_dest_online: bool = False,
    ) -> FrameArrayType:
        """Encode input data with common information of input frames.

        Args:
            data: the raw data for one input node. It will be flatten after encoding.
            icfg_template: the template of the input nodes, included in the runtime configurations.
            repeat: used to tile the data. For example, if timestep = 3, the original
                data [0, 1, 2] will be tiled as [0, 1, 2, 0, 1, 2, 0, 1, 2].
            is_dest_online: whether the input frames are to the online cores. Since the `icfg_template` can only  \
                be one input node, so either it is online input node or offline input node.

        Returns:
            Return the encoded arrays in working frame type I format.
        """
        _data = np.tile(np.asarray(data, dtype=PAYLOAD_DATA_DTYPE).ravel(), repeat)
        if is_dest_online:
            return OnlineFrameGen.gen_work_frame1_1_fast(icfg_template, _data)
        else:
            return OfflineFrameGen.gen_work_frame1_fast(icfg_template, _data)

    @classmethod
    @time_it
    def _decode(
        cls,
        oframes: FrameArrayType,
        rtcfg_map: OutputDestRTCfg,
        raise_if_not_matched: bool = True,
        raise_if_duplicated: bool = True,
    ) -> PayloadDataType:
        """Decode output frames from the chips. This method has real-time requirement.
        
        Args:
            oframes (FrameArrayType): Output frames from the chips.
            rtcfg_map (OutputDestRTCfg): The runtime configuration of output destinations.
            raise_if_not_matched (bool): Whether to raise an error if the output frames are not matched with the \
                runtime configuration.
            raise_if_duplicated (bool): Whether to raise an error if the output frames are duplicated.

        Returns:
            Decoded data.
        """
        result = np.zeros(rtcfg_map.output_shape, dtype=PAYLOAD_DATA_DTYPE)
        if oframes.size == 0:
            return result  # no data of this output node

        if rtcfg_map.is_online:
            o_ts = (oframes >> On_WF1F_1.TIMESLOT_OFFSET) & On_WF1F_1.TIMESLOT_MASK
            o_ax = (oframes >> On_WF1F_1.AXON_OFFSET) & On_WF1F_1.AXON_MASK
        else:
            o_ts = (oframes >> Off_WF1F.TIMESLOT_OFFSET) & Off_WF1F.TIMESLOT_MASK
            o_ax = (oframes >> Off_WF1F.AXON_OFFSET) & Off_WF1F.AXON_MASK

        if raise_if_duplicated:
            _, counts = np.unique(
                np.stack([o_ts, o_ax], axis=1), axis=0, return_counts=True
            )
            if np.any(counts > 1):
                raise ValueError("duplicated output frames are detected.")

        if raise_if_not_matched:
            for _ts, _ax in zip(o_ts, o_ax):
                if (_ts, _ax) not in rtcfg_map.tpl_pairs:
                    raise ValueError(
                        f"the output frame at (ts, ax)=({_ts},{_ax}) is not expected."
                    )

        loc_ts, ax_len_ext = np.divmod(o_ts, rtcfg_map.len_ex_factor)
        loc_ax = o_ax + (OffCoreCfg.ADDR_AXON_MAX + 1) * ax_len_ext

        if rtcfg_map.is_online:
            result[loc_ts, loc_ax] = 1
        else:
            o_v = (oframes >> Off_WF1F.DATA_OFFSET) & Off_WF1F.DATA_MASK
            result[loc_ts, loc_ax] = o_v

        return result

    @classmethod
    @time_it
    def decode(
        cls,
        recv_frames: FrameArrayType,
        rtcfg_map: OutputDestRTCfgMap,
        raise_if_has_other_type: bool = True,
        raise_if_not_matched: bool = True,
        raise_if_duplicated: bool = True,
    ) -> "OutputMappingType":
        """Decode output data from received frames using runtime configurations.
        Args:
            recv_frames (FrameArrayType): Received frames from the chips.
            rtcfg_map (OutputDestRTCfgMap): The runtime configurations of output nodes.
            raise_if_has_other_type (bool): Whether to raise an exception if there are frames with other types.
            raise_if_not_matched (bool): Whether to raise an exception if there are axons that are not matched.
            raise_if_duplicated (bool): Whether to raise an exception if there are axons that are duplicated.

        Returns:
            A mapping from output node names to decoded data.
        
        NOTE: This method has real-time requirement. To speed up decoding, disable `raise_if_has_other_type`,   \
            `raise_if_not_matched` & `raise_if_duplicated`.
        """
        if recv_frames.size == 0:
            decoded_omap = dict()
            for name, rtcfg in rtcfg_map.items():
                decoded_omap[name] = np.zeros(
                    rtcfg.output_shape, dtype=PAYLOAD_DATA_DTYPE
                )
            return decoded_omap

        framearray_header_check(
            recv_frames, FH.WORK_TYPE1, strict=raise_if_has_other_type
        )

        decoded_omap = dict()
        # Record whether the frame is seen
        seen_indices = np.zeros(len(recv_frames), dtype=bool)

        recv_frames.sort()
        recv_core_coords = (
            recv_frames >> FF.GENERAL_CORE_ADDR_OFFSET
        ) & FF.GENERAL_CORE_ADDR_MASK

        for name, rtcfg in rtcfg_map.items():
            mask = recv_core_coords == rtcfg.core_coord.address
            seen_indices[mask] = True
            decoded_omap[name] = cls._decode(
                recv_frames[mask], rtcfg, raise_if_not_matched, raise_if_duplicated
            )

        if np.any(~seen_indices):
            msg = "some of received frames not matched with the templates"
            if raise_if_not_matched:
                raise ValueError(msg)
            else:
                warnings.warn(msg, PAIBoardWarning)

        return decoded_omap

    @classmethod
    @deprecated("Use `gen_input_rtcfg_map` instead.")
    def gen_input_frames_template(
        cls, timestep: int, _attrs_map: dict[str, Any]
    ) -> list[FrameArrayType]:
        """Generate the common information of input frames by given the dictionary of input projections.

        Args:
            input_nodes_info (dict): the dictionary of input projections exported from `paibox.Mapper`, or  \
                you can specify the following parameters.
            chip_coord: the destination chip coordinate of the output node.
            core_coord: the destination coord coordinate of the output node.
            rid: the replication ID.
            timeslots: the range of timeslots from 0 to T.
            axons: the range of destination address of axons, from 0 to N.

        NOTE: If there are #C input nodes, the total shape of inputs will be: C*T*N.
        """
        frames = []
        ts = []

        # Traverse the input nodes
        attrs_map = InputNodeAttrsMap(**_attrs_map)

        for inode in attrs_map.values():
            target_coord = Coord(inode["addr_core_x"], inode["addr_core_y"])
            raw_ts = inode["tick_relative"]
            MAX_TIMESLOT = get_n_timeslot_max(target_coord.core_type == CoreType.ONLINE)

            if (lcn := inode.get("lcn", None)) is not None:
                if lcn * timestep > MAX_TIMESLOT:
                    raise ValueError(
                        f"with lcn={lcn}, the required timeslot ({lcn}*{timestep}) of input nodes "
                        f"is out of range {MAX_TIMESLOT}"
                    )

            is_online = target_coord.core_type == CoreType.ONLINE
            max_timeslot_check(timestep, raw_ts, is_online)

            interval = max(raw_ts) - min(raw_ts) + 1

            ts.clear()
            for i in range(timestep):
                ts.extend([addr + (i * interval) for addr in raw_ts])

            inode["tick_relative"] = ts
            # addr_axon: [0-X] -> [0-X]*timestep
            inode["addr_axon"] *= timestep

            frames_of_inp = cls.work_fgen_handler[
                target_coord.core_type
            ]._frame_dest_reorganized(cast(dict[str, Any], inode))

            frames.append(frames_of_inp)

        return frames

    @classmethod
    def gen_input_rtcfg_map(
        cls, timestep: int, _attrs_map: dict[str, Any]
    ) -> InputNodeRTCfgMap:
        """Generate the input node runtime configurations map.

        Args:
            timestep (int): the number of timeslots.
            _attrs_map (dict): the dictionary of input projections exported from `paibox.Mapper`.

        Returns:
            out (InputNodeRTCfgMap): the input node runtime configurations map.
        """
        if timestep > OffCoreCfg.N_TIMESLOT_MAX:
            raise ValueError(
                f"timestep ({timestep}) exceeds maximum allowed timeslots({OffCoreCfg.N_TIMESLOT_MAX})."
            )

        attrs_map = InputNodeAttrsMap(**_attrs_map)
        rtcfg_map = InputNodeRTCfgMap()

        for name, attrs in attrs_map.items():
            target_chip, target_coord, target_rid = attrs2coord(attrs)
            raw_ts = attrs["tick_relative"]
            is_online = target_coord.core_type == CoreType.ONLINE

            if (lcn := attrs.get("lcn", None)) is not None:
                MAX_TIMESLOT = get_n_timeslot_max(is_online)
                if lcn > MAX_TIMESLOT:
                    raise ValueError(
                        f"required lcn ({lcn}) exceeds maximum allowed timeslots({MAX_TIMESLOT})."
                    )

            interval = max(raw_ts) - min(raw_ts) + 1
            ts = []
            for i in range(timestep):
                ts.extend([addr + i * interval for addr in raw_ts])

            # addr_axon: [0-X] -> [0-X]*timestep
            axons = attrs["addr_axon"] * timestep

            # Use processed axons & timeslots to generate frame template
            frames_of_inp = cls.work_fgen_handler[
                target_coord.core_type
            ].concat_frame_dest(target_chip, target_coord, target_rid, axons, ts)

            rtcfg_map[name] = InputNodeRTCfg(attrs, frames_of_inp)

        return rtcfg_map

    @classmethod
    @deprecated("Use `gen_output_rtcfg_map` instead.")
    def gen_output_frames_template(
        cls,
        timestep: int,
        output_dest_info: OutputDestAttrsMap,
    ) -> FrameArrayType | list[FrameArrayType]:
        """Generate the common information of output frames by given the dictionary of output destinations.

        Args:
            timestep (int): used to tile the 'tick_relative' info of output destinations.
            output_dest_info (dict) : the dictionary of output destinations exported from `paibox.Mapper`,  \
                or you can specify the following parameters.
        """
        frames = []
        for onode in output_dest_info.values():
            # One output node must be on cores of one type.
            target_coord = coordstr2coord(next(iter(onode)))

            # Get the length expansion multiple of the output node
            len_ex_factor = get_len_ex_factor(onode)
            MAX_TIMESLOT = get_n_timeslot_max(target_coord.core_type == CoreType.ONLINE)

            if len_ex_factor * timestep > MAX_TIMESLOT:
                raise ValueError(
                    "required timeslot of output nodes is out of maximum timeslot: "
                    f"{len_ex_factor}*{timestep} > {MAX_TIMESLOT}"
                )

            # Traverse output destinations of a node
            frames_of_dest = []
            for t in range(timestep):
                for dest_on_coord in onode.values():
                    # For example:
                    # TR: [0,0,0,0,1,1] with T=3 -> [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,5]
                    if t > 0:
                        dest_on_coord["tick_relative"] = [
                            x + len_ex_factor for x in dest_on_coord["tick_relative"]
                        ]

                    frames_of_dest.append(
                        cls.work_fgen_handler[
                            target_coord.core_type
                        ]._frame_dest_reorganized(cast(dict[str, Any], dest_on_coord))
                    )

            frames.append(np.concatenate(frames_of_dest))

        return frames

    @classmethod
    def gen_output_rtcfg_map(
        cls, timestep: int, _attrs_map: dict[str, Any]
    ) -> OutputDestRTCfgMap:
        if timestep > OffCoreCfg.N_TIMESLOT_MAX:
            raise ValueError(
                f"timestep ({timestep}) exceeds maximum allowed timeslots({OffCoreCfg.N_TIMESLOT_MAX})."
            )

        attrs_map = OutputDestAttrsMap(**_attrs_map)
        rtcfg_map = OutputDestRTCfgMap()

        for name, onode in attrs_map.items():
            src_coords = [coordstr2coord(c) for c in onode]
            # All source cores of one output node must be of one type.
            if not all(c.core_type == src_coords[0].core_type for c in src_coords):
                raise ValueError(
                    "all source cores of one output node must be of one type"
                )

            # NOTE: Currently, all dest attributes of one output node must be the same.
            dest_coord_tuple = [attrs2coord(attrs) for attrs in onode.values()]
            if not all(t == dest_coord_tuple[0] for t in dest_coord_tuple):
                raise ValueError(
                    "all dest attributes of one output node must be the same"
                )

            # Get the length expansion multiple of the output node
            len_ex_factor = get_len_ex_factor(onode)

            MAX_TIMESLOT = get_n_timeslot_max(
                src_coords[0].core_type == CoreType.ONLINE
            )
            if len_ex_factor * timestep > MAX_TIMESLOT:
                raise ValueError(
                    "required total length of the output node is out of maximum timeslot: "
                    f"{len_ex_factor}*{timestep} > {MAX_TIMESLOT}"
                )

            # Traverse all destinations of an output node
            frames_of_dest = []
            for t in range(timestep):  # The order is important
                # Generate the template of all output nodes at timestep=t
                for attrs in onode.values():
                    target_chip, target_coord, target_rid = attrs2coord(attrs)
                    raw_ax = attrs["addr_axon"]
                    raw_ts = attrs["tick_relative"]

                    # For example: TR[0,0,0,0,1,1] with T=3 -> [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,5]
                    ts = [i + t * len_ex_factor for i in raw_ts]

                    frames_of_dest.append(
                        cls.work_fgen_handler[
                            src_coords[0].core_type
                        ].concat_frame_dest(
                            target_chip, target_coord, target_rid, raw_ax, ts
                        )
                    )

            rtcfg_map[name] = OutputDestRTCfg(
                src_coords,
                attrs,
                timestep,
                len_ex_factor,
                template=np.concatenate(frames_of_dest),
            )

        return rtcfg_map

    @classmethod
    def gen_read_neuron_attrs_frames(
        cls,
        neu_phy_loc: dict[str, Any],
        reading_mode: Literal["onebyone", "contiguous"] = "contiguous",
    ) -> list[tuple[FrameArrayType, int]]:
        """Generate test input frame type III for single neuron node to read their attributes.

        Args:
            neu_phy_loc (dict[str, Any]): the physical locations of a single neuron node. For example:
            reading_mode ("onebyone", "contiguous"):
                - "onebyone": read the addresses of neurons that contain the correct voltage at intervals.
                - "contiguous" (default): read the addresses of neurons contiguously. Necessary to retrieve \
                    the addresses that store the correct voltage based on the interval.

        Returns:
            A list of tuples of test input frames & the number of packages to read.

        Usage:

            >>> d = {
                    "IF_0": {
                        "(0,0)": {
                            "(0,0)": {
                                "n_neuron": 50,
                                "addr_offset": 0,
                                "interval": 8,
                                "idx_offset": 0
                            },
                            "(0,1)": {
                                "n_neuron": 50,
                                "addr_offset": 0,
                                "interval": 8,
                                "idx_offset": 50
                            }
                        }
                    }
                }
            >>> tframes3_if0 = PAIBoxRuntime.gen_read_neuron_attrs_frames(d["IF_0"])
            >>> tframes3_if0[0]
            >>> (input test frame, #N of packages to read)

        NOTE: Test output frames will be output to `test_chip_addr` which is configured before. The output  \
            frames will be out of order if using the replication id to test multiple cores at the same time.

        NOTE: Since the chip has a hardware flaw that once read the neuron addresses contiguously, the 2nd  \
            address will be missed maybe. According to out experiments, for example, reading the neuron     \
            addresses [0]~[99], the output frame will be of the neurons addresses [0] & [2]-[100].

            This behavior is not officially documented in any chip manuals.
        """
        # [(frame package, n_package)]
        tframe3: list[tuple[OfflineTestInFrame3, int]] = []
        if reading_mode not in ("onebyone", "contiguous"):
            raise ValueError(f"unknown reading mode '{reading_mode}'")

        N_FRAME_PAYLOAD = Off_ToF3.N_FRAME_PAYLOAD

        for _chip_coord, core_locs in neu_phy_loc.items():
            chip_coord = coordstr2coord(_chip_coord)
            for _core_coord, _seg_addr in core_locs.items():
                core_coord = coordstr2coord(_core_coord)
                nseg_addr = NeuSegAddrAttrs(_seg_addr)  # cast to typed dict
                n_neuron = nseg_addr["n_neuron"]
                addr_offset = nseg_addr["addr_offset"]
                interval = nseg_addr["interval"]

                if reading_mode == "onebyone":
                    for i in range(n_neuron):
                        # NOTE: Mapping between logical neuron indexes, neuron addresses & SRAM addresses:
                        # Logical idx:                    [0]                           [1]
                        #                   |<------- interval=8 -------->|                             |
                        # Neuron address:      [0]      [1]   ...   [7]      [8]        ...       [15]
                        # SRAM address:     [0*4+:4] [1*4+:4] ... [7*4+:4] [8*4+:4]     ...     [15*4+:4]
                        # NOTE: According to our experiments, the attributes of each logical neuron at index `i` is stored
                        # many times repeatedly in the **neuron address** addr_offset+[i],[i+1],...,[i+interval-1].
                        # However, the correct voltage is stored in the **neuron address** addr_offset+[i], or
                        # [(addr_offset+i)*4+:4] in SRAM. Reading [i] is the most efficient way to get all attributes.
                        # This behavior above is not officially documented in any chip manuals.
                        tframe3.append(
                            (
                                OfflineFrameGen.gen_testin_frame3(
                                    chip_coord,
                                    core_coord,
                                    _RID_UNSET,
                                    # NOTE: Attention! The argument `sram_base_addr` of config/test frame 3 & 4 is incorrectly
                                    # named. In fact, it is the **neuron start address** as described above.
                                    addr_offset + i * interval,
                                    N_FRAME_PAYLOAD,
                                ),
                                N_FRAME_PAYLOAD,
                            )
                        )
                else:
                    if interval == 1:
                        # Read two times if #N of neurons > 1, otherwise read once.
                        # 1. Set neuron start addr=addr_offset,   n_package=4*1*(N-1), to read neuron addr_offset+[0] & [2]~[N-1].
                        # 2. Set neuron start addr=1+addr_offset, n_package=4*1*1,     to read neuron addr_offset+[1](if N > 1).
                        n_package = (
                            N_FRAME_PAYLOAD * (n_neuron - 1)
                            if n_neuron > 1
                            else N_FRAME_PAYLOAD
                        )
                        tframe3.append(
                            (
                                OfflineFrameGen.gen_testin_frame3(
                                    chip_coord,
                                    core_coord,
                                    _RID_UNSET,
                                    addr_offset,
                                    n_package,
                                ),
                                n_package,
                            )
                        )
                        if n_neuron > 1:
                            tframe3.append(
                                (
                                    OfflineFrameGen.gen_testin_frame3(
                                        chip_coord,
                                        core_coord,
                                        _RID_UNSET,
                                        1 + addr_offset,
                                        N_FRAME_PAYLOAD,
                                    ),
                                    N_FRAME_PAYLOAD,
                                )
                            )
                    else:
                        # Although the addresses of neurons read contiguously have deviations, the address
                        # containing the correct voltage have still been read accurately.
                        # When interval > 1, for example 4:
                        # In order to read neuron [0], [4], [4*2], ..., [4*N], set test input frame with:
                        #   start addr = addr_offset
                        #   n_package = 4*interval(4)*N
                        # Return addresses: [0], [2], [3], [4], ..., [4*N+1]
                        n_package = 4 * interval * n_neuron
                        tframe3.append(
                            (
                                OfflineFrameGen.gen_testin_frame3(
                                    chip_coord,
                                    core_coord,
                                    _RID_UNSET,
                                    addr_offset,
                                    n_package,
                                ),
                                n_package,
                            )
                        )

        return [(f.value, n_package) for (f, n_package) in tframe3]

    @classmethod
    def decode_voltage(
        cls,
        phy_loc_of_neu: dict[str, Any],
        otf3_package: Iterable[FrameArrayType],
        reading_mode: Literal["onebyone", "contiguous"] = "contiguous",
        *,
        is_complete: bool = True,
        is_online: bool = False,
        weight_width: Literal[1, 2, 4, 8] = 8,
    ) -> VoltageType:
        """Decode type III test output frames of a single neuron node for reading the voltage. The physical \
            locations of the neurons will be aligned with their logical positions.

        Args:
            neu_phy_loc (dict[str, Any]): the physical locations of all neurons.
            otf3_package (Iterable[FrameArrayType]): the test output frames of type III.
            reading_mode ("onebyone", "contiguous"): the reading mode of neuron addresses.
            is_complete (bool): whether the current decoding is for all neurons declared in `neu_phy_loc`.  \
                If not, it's necessary to distinguish the decoded voltage returned by multiple calls to this\
                function.
            is_online (bool): whether the decoding is for online cores.
            weight_width (1, 2, 4, 8): the weight width of online cores. Only valid when `is_online` is true.

        Usage:

            >>> ...
            >>> reading_mode = "contiguous"
            >>> tframes3_if0 = PAIRuntime.gen_read_neuron_attrs_frames(d["IF_0"], reading_mode)

            >>> # At hardware platform:
            >>> otframes = []
            >>> for item, n_package in tframes3_if0:
            >>>     itf = item[0]
            >>>     # Send to the chip
            >>>     send_to_chip(itf)
            >>>     # Receive & retrieve: start frame + #N packages
            >>>     r = recv_from_chip(n_package+1)[:1+n_package]
            >>>     otframes.append(r)
            >>> decoded_v = PAIRuntime.decode_voltage(d["IF_0"], otframes, reading_mode="contiguous)
        """
        n_neu_total, core_nseg_locs = parse_ney_phy_loc(NeuPhyLocMap(phy_loc_of_neu))
        decoded_v = np.zeros((n_neu_total,), dtype=VOLTAGE_DTYPE)

        n_neu_proc = 0
        for otf3 in otf3_package:
            assert otf3.ndim == 1
            n_neu_proc += decode_partial_voltage(
                otf3, core_nseg_locs, reading_mode, decoded_v, is_online, weight_width
            )

        if is_complete and (n_neu_proc != n_neu_total):
            raise ValueError(
                f"the number of total neurons decoded is {n_neu_proc}, but expected {n_neu_total}"
            )

        return decoded_v


def parse_ney_phy_loc(phy_loc_of_neu: NeuPhyLocMap) -> tuple[int, CoreNeuSegLocMap]:
    """Parse the physical locations of neurons.

    Returns:
        A tuple of the total number of neurons & the dictionary of core-neuron segment locations.
    """
    if (n_chip := len(phy_loc_of_neu)) > 1:
        raise ValueError(f"the neuron is on {n_chip} chips")

    n_total = 0
    locs = CoreNeuSegLocMap()
    for phy_loc_on_coord in phy_loc_of_neu.values():
        for coord_str, seg_addr in phy_loc_on_coord.items():
            cur_coord = coordstr2coord(coord_str)
            locs[cur_coord] = seg_addr
            n_total += seg_addr["n_neuron"]

    return n_total, locs


@overload
def decode_partial_voltage(
    otframe3: FrameArrayType,
    core_locs: CoreNeuSegLocMap,
    reading_mode: Literal["onebyone", "contiguous"],
    out: VoltageType,
    is_online: Literal[False] = False,
    weight_width: Literal[1, 2, 4, 8] = 8,
) -> int: ...


@overload
def decode_partial_voltage(
    otframe3: FrameArrayType,
    core_locs: CoreNeuSegLocMap,
    reading_mode: Literal["onebyone", "contiguous"],
    out: VoltageType,
    is_online: Literal[True] = True,
    weight_width: Literal[1, 2, 4, 8] = 8,
) -> int: ...


def decode_partial_voltage(
    otframe3: FrameArrayType,
    core_locs: CoreNeuSegLocMap,
    reading_mode: Literal["onebyone", "contiguous"],
    out: VoltageType,
    is_online: bool = False,
    weight_width: Literal[1, 2, 4, 8] = 8,
) -> int:
    """According to the test output frames of each core `otframe3` for a single neuron node, decode the \
        corresponding voltage for this part and store it in the array `out`.

    Args:
        otframe3 (FrameArrayType): the test output frame of type III.
        core_locs (dict[Coord, NeuSegAddrAttrs]): the dictionary of core-neuron segment locations.
        reading_mode ("onebyone", "contiguous"): the reading mode of neuron addresses.
        out (VoltageType): the array to store the complete decoded voltages.
        is_online (bool): whether the decoding is for online cores.
        weight_width (1, 2, 4, 8): the weight width of online cores. Only valid when `is_online` is true.

    Returns:
        n_neuron (int): The number of processed neurons in the current package.

    NOTE: If reading mode is "onebyone", `otframe3` is a frame package of length 4*1*1. Decode 1 neuron \
        for each package.
        If reading mode is "contiguous", `otframe3` is a frame package of length 4*interval*N. Decode N \
        neurons for each package.

        Since the chip has a hardware flaw that once read the neuron addresses contiguously, the 2nd    \
        address will be missed maybe. The method for retrieving the addresses that contain the correct  \
        voltage from the discontiguous neuron addresses, and corresponding them with logical positions, \
        is derived from our experiments.

        This behavior is not officially documented in any chip manuals.

    NOTE: Not validated on online cores yet.
    """
    start_frame = int(otframe3[0])
    if not (
        FH.TEST_TYPE3
        == FH((start_frame >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK)
        and (start_frame >> FF.GENERAL_PACKAGE_TYPE_OFFSET)
        & FF.GENERAL_PACKAGE_TYPE_MASK
        == FPType.CONF_TESTOUT
    ):
        raise ValueError("invalid test output frame type III")

    core_coord = (
        start_frame >> FF.GENERAL_CORE_ADDR_OFFSET
    ) & FF.GENERAL_CORE_ADDR_MASK
    neu_addr = (
        start_frame >> FF.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET
    ) & FF.GENERAL_PACKAGE_NEU_START_ADDR_MASK
    n_package = (
        start_frame >> FF.GENERAL_PACKAGE_NUM_OFFSET
    ) & FF.GENERAL_PACKAGE_NUM_MASK

    if n_package != otframe3.size - 1:
        raise ValueError(
            f"the number of packages is expected to be {n_package}, but got {otframe3.size-1}"
        )

    if (coord := Coord.from_addr(core_coord)) not in core_locs:
        expected = ", ".join(str(c) for c in core_locs)
        raise ValueError(f"{coord} is not in expected locations: {expected}.")

    nseg_addr = core_locs[coord]
    interval = nseg_addr["interval"]

    if is_online:
        if weight_width == 1:
            N_FRAME_PAYLOAD = 2
        else:
            N_FRAME_PAYLOAD = 4
    else:
        N_FRAME_PAYLOAD = Off_ToF3.N_FRAME_PAYLOAD

    if reading_mode not in ("onebyone", "contiguous"):
        raise ValueError(f"wrong reading mode '{reading_mode}'")

    if reading_mode == "onebyone":
        if (n_neu_proc := n_package // N_FRAME_PAYLOAD) != 1:
            raise ValueError(
                f"when reading neuron addresses one by one, the number of packages is expected to be {N_FRAME_PAYLOAD}, "
                f"but got {n_package}"
            )
    else:
        if (n_neu_proc := n_package // (N_FRAME_PAYLOAD * interval)) > out.size:
            raise ValueError(
                f"the number of packages exceeds the max size: {n_package} > {N_FRAME_PAYLOAD}*{interval}*{out.size}"
            )

    # Get the voltage of neuron[0]. Slice starting from 1 to skip the start frame.
    if is_online:
        if weight_width == 1:
            nramf_hdlr, idx_v_at_arr = ON_NRAMF_WW1, 2
        else:
            nramf_hdlr, idx_v_at_arr = ON_NRAMF_WWn, 2
    else:
        nramf_hdlr, idx_v_at_arr = Off_NRAMF, 1

    v_array_idx0 = (
        int(otframe3[idx_v_at_arr]) >> nramf_hdlr.VOLTAGE_OFFSET
    ) & nramf_hdlr.VOLTAGE_MASK

    # See comments in `gen_read_neuron_attrs_frames()` above.
    logic_idx = nseg_addr["idx_offset"] + (
        (neu_addr - nseg_addr["addr_offset"]) // interval
    )
    out[logic_idx] = convert_30bit_to_signed(v_array_idx0)

    if reading_mode == "onebyone" or n_neu_proc == 1:
        return 1

    # In contiguous mode, read the rest frames containing the voltage of neurons[2:N-1]
    start_idx_2nd = (
        1 + N_FRAME_PAYLOAD * (interval - 1)
        if interval > 1
        else 1 + N_FRAME_PAYLOAD * 1
    )
    end_idx_2nd = (-1 * N_FRAME_PAYLOAD) if interval > 1 else None

    # When interval>1, the 2nd~#N neuron address is in order in `otframe3`, while interval=1,
    # the 2nd neuron address is missed, so start from `logic_idx+2`.
    start_logic_idx_2nd = logic_idx + 1 if interval > 1 else logic_idx + 2

    v_array = (
        otframe3[start_idx_2nd : end_idx_2nd : N_FRAME_PAYLOAD * interval]
        >> Off_NRAMF.VOLTAGE_OFFSET
    ) & Off_NRAMF.VOLTAGE_MASK

    assert v_array.size == n_neu_proc - 1

    vf_convert = np.vectorize(convert_30bit_to_signed, otypes=[VOLTAGE_DTYPE])
    out[start_logic_idx_2nd : start_logic_idx_2nd + (n_neu_proc - 1)] = vf_convert(
        v_array
    )

    return n_neu_proc


def convert_30bit_to_signed(x: int) -> VOLTAGE_DTYPE:
    """Convert an integer to a 32-bit signed number."""
    x_30b = x & ((1 << 30) - 1)

    if (x_30b >> 29) & 1:
        x_30b -= 0x4000_0000  # Negative integer

    return VOLTAGE_DTYPE(x_30b)
