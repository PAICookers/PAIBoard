from pathlib import Path

import numpy as np
from paicorelib.framelib import FrameArrayType
from pynq import Overlay  # pyright: ignore

from paiboard.common import RegFile
from paiboard.exceptions import PAIBoardDeviceError
from paiboard.global_cfg import FRAME_VALUE_FILTER_LIST

from ..utils import auto_search_bitfile
from .hw_hierarchy import HwDatapathHier


class ZynqPlatformOverlay(Overlay):
    # TODO Add zynq clock frequency interface
    # TODO Debugbridge https://pynq-testing.readthedocs.io/en/latest/pynq_libraries/debugbridge.html
    # Overlay: https://github.com/Xilinx/PYNQ/blob/e5d0c033/docs/source/overlay_design_methodology/overlay_tutorial.ipynb
    pl_datapath: HwDatapathHier

    def __init__(
        self, bitfile: Path | str, start_debug_bridge: bool = False, **kwargs
    ) -> None:
        bitfile_fp = auto_search_bitfile(bitfile)
        super().__init__(str(bitfile_fp), **kwargs)
        if not self.is_loaded():
            raise PAIBoardDeviceError(f"overlay {bitfile_fp} not loaded")

        # For debug
        for k in self.ip_dict:
            print(f"ip dict: {k}")

        for h in self.hierarchy_dict:
            print(f"hierarchy dict: {h}")

        if start_debug_bridge:
            self.pl_datapath.start_debug_bridge()

    def reset_regfile(self) -> None:
        self.write_reg(RegFile.CPU2FIFO_CNT, 0)
        self.write_reg(RegFile.FIFO2SNN_CNT, 0)
        self.write_reg(RegFile.SNN2FIFO_CNT, 0)
        self.write_reg(RegFile.FIFO2CPU_CNT, 0)
        self.write_reg(RegFile.DATAPATH_RSTN, 0x01)
        self.write_reg(RegFile.OEN, 0x01)
        self.write_reg(RegFile.CHANNEL_MASK, 0x01)

    def write_reg(self, addr: int, value: int) -> None:
        self.pl_datapath.write_regfile(addr, value)

    def read_reg(self, addr: int, length: int = 4) -> int:
        return self.pl_datapath.read_regfile(addr, length)

    def send_frame(self, frame: FrameArrayType) -> int:
        return self.pl_datapath.write_dma(frame)

    def recv_frame(self, size: int) -> FrameArrayType:
        recv_arr = self.pl_datapath.read_dma(size)
        # Remove frames with zero or dummy value.
        return recv_arr[~np.isin(recv_arr, FRAME_VALUE_FILTER_LIST)]

    def reset_chip(self, *chip_idx: int) -> None:
        """Reset chips. If `chip_idx` is empty, reset all chips."""
        if chip_idx:
            v = sum(1 << i for i in chip_idx)
        else:
            v = 0xF

        self.pl_datapath.write_axi_gpio_chip_reset(v)

    def send_uart_cmd(self, uart_idx: int, uart_cmd_bytes: bytes) -> bytes:
        return self.pl_datapath.send_uart_cmd(uart_idx, uart_cmd_bytes)
