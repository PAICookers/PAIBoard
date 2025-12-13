from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from paicorelib import CoordLike

from ..base import PAIBoard
from ..board_cfg import ChipSOMType
from ..common.uart_cfg import ChipUartCfg
from .client import ZynqClient
from .overlay import ON_ZYNQ
from .zynq_abc import ZynqCtrlInterface


class PAIBoardZynqBase(PAIBoard):
    chip_som_type = ChipSOMType.ARRAY_2X2
    intf: ZynqCtrlInterface

    def chip_array_uart_config(
        self,
        chip_coords: Sequence[CoordLike],
        clk_freq: int = 240,
        debug_en: bool = False,
    ) -> None:
        """Configure all chips in the array via UART."""
        self.intf.reset_chip()
        for i, chip_coord in enumerate(chip_coords):
            self.chip_uart_config(i, chip_coord, clk_freq, debug_en=debug_en)

    def chip_uart_config(
        self,
        idx: int,
        chip_coord: CoordLike,
        clk_freq: int = 240,
        debug_en: bool = False,
        clk_en_L2: list[int] | None = None,
    ) -> None:
        """Configure a single chip via UART."""
        self.intf.reset_chip(idx)
        uart_cmd = ChipUartCfg.gen_uart_cmd(
            chip_coord, clk_freq, clk_en_L2=clk_en_L2, debug=debug_en
        )
        echo = self.intf.config_uart_chip(idx, uart_cmd)
        print(f"UART {idx} echo: {echo}")

    def chip_uart_debug_enable(
        self,
        idx: int,
        chip_coord: CoordLike,
        clk_freq: int = 240,
        clk_en_L2: list[int] | None = None,
    ) -> None:
        uart_cmd = ChipUartCfg.gen_uart_cmd(
            chip_coord, clk_freq, clk_en_L2=clk_en_L2, debug=True
        )
        echo = self.intf.config_uart_chip(idx, uart_cmd)
        print(f"UART {idx} echo: {echo}")


if ON_ZYNQ:
    from .local import ZynqLocal

    class PAIBoardZynq(PAIBoardZynqBase):

        def __init__(
            self,
            toolchain_build_dir: Path | str,
            timestep: int,
            n_layer: int,
            batch_mode: bool = False,
            excluded_init_chips: Sequence[CoordLike] | None = None,
            *,
            bitfile: Path | str,
            n_max_oframe: int | None = None,
            neu_vol_reading_mode: Literal["contiguous", "onebyone"] = "contiguous",
            debug_mode: bool = False,
        ) -> None:
            self.intf = ZynqLocal(bitfile, n_max_oframe)
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


class PAIBoardZynqClient(PAIBoardZynqBase):

    def __init__(
        self,
        toolchain_build_dir: Path | str,
        timestep: int,
        n_layer: int,
        server_ip: str,
        server_port: int,
        batch_mode: bool = False,
        excluded_init_chips: Sequence[CoordLike] | None = None,
        *,
        n_max_oframe: int | None = None,
        neu_vol_reading_mode: Literal["contiguous", "onebyone"] = "contiguous",
        timeout: float | None = None,
        debug_mode: bool = False,
    ) -> None:
        self.intf = ZynqClient(server_ip, server_port, n_max_oframe, timeout)
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
