from pathlib import Path
from typing import Literal

from ..base import PAIBoard
from ..board_cfg import ChipSOMType, get_board_cfg
from ..common import ChipUartCfg
from ..types import InputMappingAnyType, OutputMappingType, PayloadDataType
from .xdma_ctrl import XDMACtrl


class PAIBoardPCIe(PAIBoard):
    chip_som_type = ChipSOMType.SINGLE_BONDING
    intf: XDMACtrl

    def __init__(
        self,
        toolchain_build_dir: Path | str,
        timestep: int,
        n_layer: int,
        batch_mode: bool = False,
        *,
        board_name: Literal[
            "FLIP8", "BONDING003", "BONDING004", "BONDING008", "BONDING8"
        ] = "FLIP8",
        xdma_dev_idx: int = 0,
        xdma_channel: int = 0,
        n_max_channel: int = 4,
        neu_vol_reading_mode: Literal["contiguous", "onebyone"] = "contiguous",
        timeout: int = 2,
        debug_mode: bool = False,
    ) -> None:
        self.intf = XDMACtrl(
            get_board_cfg(board_name, n_max_channel),
            xdma_dev_idx,
            xdma_channel,
            timeout=timeout,
        )
        super().__init__(
            toolchain_build_dir,
            timestep,
            n_layer,
            batch_mode=batch_mode,
            neu_vol_reading_mode=neu_vol_reading_mode,
            debug_mode=debug_mode,
        )

    def chip_uart_config(
        self,
        port: str | None = None,
        baudrate: int = 9600,
        clk_freq: int = 312,
        *,
        debug: bool = False,
        clk_en_L2: list[int] | None = None,
    ) -> bytes:
        """Configure the chip via UART.

        Args:
            port: The serial port of UART. If None, auto search the port.
            baudrate: The baudrate of UART.
            clk_freq: The clock frequency of the chip in MHz.
            debug: Whether to enable the debug mode of UART.
            clk_en_L2: The clock enable list for L2 layers. If None, all offline cores are enabled &    \
                all online cores are disabled.
        """
        return ChipUartCfg.serial_config(
            port,
            baudrate,
            self.source_chip,
            clk_freq,
            clk_en_L2=clk_en_L2,
            debug=debug,
            global_signal_delay=self.intf.board_cfg.global_signal_delay,
        )

    def inference(
        self,
        inputs: InputMappingAnyType,
        recv_max_size: int | None = None,
        filter_output_strict: bool = True,
        decoding_output_strict: bool = True,
        *,
        multi_channel_enable: bool = False,
    ) -> PayloadDataType | OutputMappingType:
        return super().inference(
            inputs,
            recv_max_size,
            filter_output_strict,
            decoding_output_strict,
            multi_channel_enable=multi_channel_enable,
        )

    def prepare(
        self,
        clk_freq: int = 312,
        n_max_oframe: int | None = None,
        uart_debug_en: bool = False,
    ) -> None:
        self.chip_uart_config(clk_freq=clk_freq, debug=uart_debug_en)
        self.set_n_max_oframe(n_max_oframe)
        self.chip_hw_model_download()
