import atexit
from pathlib import Path

from paicorelib.framelib import FrameArrayType

from ..overlay import ZynqPlatformOverlay
from ..zynq_abc import ZynqCtrlInterface


class ZynqLocal(ZynqCtrlInterface):
    ol: ZynqPlatformOverlay

    def __init__(
        self,
        bitfile: str | Path,
        n_max_oframe: int | None = None,
        *,
        start_debug_bridge: bool = False,
    ) -> None:
        self.running = False
        atexit.register(self.close)
        self.ol = ZynqPlatformOverlay(bitfile, start_debug_bridge=start_debug_bridge)

        self.open()
        self.set_n_max_oframe(n_max_oframe)

    def open(self) -> None:
        self.reset_regfile()
        self.running = True

    def close(self, reset_chip: bool = True) -> None:
        if self.ol.is_loaded():
            if reset_chip:
                self.reset_chip()

            self.reset_regfile()

        self.running = False

    def reset_regfile(self) -> None:
        """Reset the regfile."""
        self.ol.reset_regfile()

    def write_reg(self, addr: int, value: int) -> None:
        """Write a register."""
        self.ol.write_reg(addr, value)

    def read_reg(self, addr: int) -> int:
        """Read a register."""
        return self.ol.read_reg(addr, length=4)

    def reset_chip(self, *chip_idx: int) -> None:
        """Reset the chip or chip array."""
        self.ol.reset_chip(*chip_idx)

    def config_uart_chip(self, uart_idx: int, cmd: bytes) -> bytes:
        """Configure the chips via UART."""
        return self.ol.send_uart_cmd(uart_idx, cmd)

    def chip_uart_debug_en(self, uart_idx: int, enable: bool = True) -> None:
        """Enable or disable debug mode of the UARTs."""
        self.ol.set_uart_debug(uart_idx, enable)

    def send_frames(self, frames: FrameArrayType, **kwargs) -> int:
        """Send frames to the server."""
        size = self.ol.send_frame(frames)
        if size != frames.nbytes:
            raise ValueError(f"send bytes {size} != frames bytes {frames.nbytes}")

        return size

    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType:
        """Send frames to the server & receive. Format: SEND_AND_RECV + frames.

        The server returns: SEND_AND_RECV + received frames.
        """
        self.send_frames(frames, **kwargs)
        if recv_size is None:
            recv_size = self.n_max_oframe

        return self.ol.recv_frame(recv_size)
