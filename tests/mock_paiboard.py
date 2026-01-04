import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from paicorelib.coordinate import CoordLike
from paicorelib.framelib import FrameArrayType

from paiboard.base import PAIBoard
from paiboard.common import HostCtrlInterface
from paiboard.runtime import PAIRuntime

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class MockCtrlIntf(HostCtrlInterface):
    def __init__(self) -> None:
        self.running = False
        self.open()

    def open(self, *args, **kwargs) -> None:
        self.running = True

    def close(self) -> None:
        self.running = False

    def reset_regfile(self) -> None:
        pass

    def reset_chip(self, *chip_idx: int) -> None:
        pass

    def write_reg(self, addr: int, value: int) -> None:
        pass

    def read_reg(self, addr: int) -> int:
        return 0

    def send_frames(self, frames: FrameArrayType, **kwargs) -> int:
        return 0

    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType:
        return np.zeros_like(frames)


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

    @override
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
        """Override the `PAIBoard.chip_hw_inference()`, return the fake result."""
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
