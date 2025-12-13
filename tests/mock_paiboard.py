from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from paicorelib.coordinate import CoordLike
from paiboard.base import PAIBoard
from paiboard.common import HostCtrlInterface
from paiboard.runtime import PAIRuntime
from paiboard.types import FrameArrayType
from tests.utils import gen_random_array


class MockCtrlIntf(HostCtrlInterface):
    def __init__(self) -> None:
        self._regs: dict[int, int] = {}
        self.writes: list[tuple[int, int]] = []
        self.reads: list[int] = []
        self.sent_frames: list[tuple[FrameArrayType, dict[str, Any]]] = []
        self.send_and_recv_calls: list[
            tuple[FrameArrayType, int | None, dict[str, Any]]
        ] = []
        self.reset_chips: list[tuple[int, ...]] = []
        self.running = True

    def open(self, *args, **kwargs) -> None:
        self.running = True

    def close(self) -> None:
        self.running = False

    def reset_regfile(self) -> None:
        for r in self._regs:
            self._regs[r] = 0

    def reset_chip(self, *chip_idx: int) -> None:
        self.reset_chips.append(chip_idx)

    def write_reg(self, addr: int, value: int) -> None:
        self._regs[addr] = value
        self.writes.append((addr, value))

    def read_reg(self, addr: int) -> int:
        self.reads.append(addr)
        return self._regs.get(addr, 0)

    def send_frames(self, frames: FrameArrayType, **kwargs) -> int:
        self.sent_frames.append((frames, kwargs))
        # Mimic hardware: return number of bytes sent.
        if hasattr(frames, "nbytes"):
            return int(frames.nbytes)
        return int(len(frames))

    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType:
        self.send_and_recv_calls.append((frames, recv_size, kwargs))
        size = recv_size if recv_size is not None else frames.size
        return np.zeros(
            int(size),
            dtype=frames.dtype if isinstance(frames, np.ndarray) else np.uint64,
        )


class MockPAIBoard(PAIBoard):
    _fake_return_data: np.ndarray

    def __init__(
        self,
        toolchain_build_dir: Path | str,
        timestep: int,
        n_layer: int,
        batch_mode: bool = False,
        *,
        excluded_init_chips: Sequence[CoordLike] | None = None,
        neu_vol_reading_mode: Literal["contiguous", "onebyone"] = "contiguous",
        debug_mode: bool = True,
    ) -> None:
        self.intf = MockCtrlIntf()
        super().__init__(
            toolchain_build_dir,
            timestep,
            n_layer,
            batch_mode=batch_mode,
            excluded_init_chips=(
                list(excluded_init_chips) if excluded_init_chips else None
            ),
            neu_vol_reading_mode=neu_vol_reading_mode,
            debug_mode=debug_mode,
        )

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
        """Take over the `PAIBoard.chip_hw_inference()`, return the fake result."""
        onode = list(self.output_rtcfg_map.keys())[0]
        oshape = self.output_rtcfg_map[onode].output_shape
        tpl = self.output_rtcfg_map[onode].template
        tpl_in_ts_range = tpl.reshape(oshape)[
            self._infer_ts : self._infer_ts + self._infer_ts_forward
        ]

        _shape = tpl_in_ts_range.shape
        n, m = _shape
        repeated_indices = np.repeat(np.arange(1, 1 + n), m)

        data = repeated_indices.reshape(n, m)
        # data = gen_random_array(tpl_in_ts_range.shape, dtype=np.uint8)
        self._fake_return_data = data

        # Encode the fake frames as the chip output.
        return PAIRuntime.encode(
            data,
            tpl_in_ts_range.ravel(),
            is_dest_online=self.output_rtcfg_map[onode].is_online,
        )
