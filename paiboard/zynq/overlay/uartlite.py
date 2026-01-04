from time import sleep, time
from typing import Any

from pynq import MMIO  # pyright: ignore

__all__ = ["UartAXI"]


class UartAXI:
    """AXI UART Lite controller for FPGA communication. See pg142 for details."""

    RX_FIFO = 0x00
    TX_FIFO = 0x04
    STATUS_REG = 0x08
    CTRL_REG = 0x0C

    # Status register bits
    STATUS_RX_VALID = 0
    STATUS_RX_FULL = 1
    STATUS_TX_EMPTY = 2
    STATUS_TX_FULL = 3
    STATUS_IS_INT_EN = 4
    STATUS_OVERRUN_ERR = 5
    STATUS_FRAME_ERR = 6
    STATUS_PARITY_ERR = 7

    # Control register bits
    CTRL_RST_TX_FIFO = 0
    CTRL_RST_RX_FIFO = 1
    CTRL_INT_EN = 4

    def __init__(self, address: int, length: int = 0x1000) -> None:
        self.address = address
        self.uart = MMIO(address, length)  # length 4K, defined in the hardware design

    @staticmethod
    def get_bit(num: int, pos: int) -> int:
        return (num & (1 << pos)) >> pos

    def setup_ctrl_reg(self) -> None:
        # Reset FIFOs, disable interrupts
        self.uart.write(
            self.CTRL_REG, (1 << self.CTRL_RST_TX_FIFO) | (1 << self.CTRL_RST_RX_FIFO)
        )
        sleep(0.5)
        self.uart.write(self.CTRL_REG, 0)

    def get_status(self) -> dict[str, Any]:
        """Returns object that specifies current status of axi core"""
        status = self.uart.read(self.STATUS_REG)
        return {
            "RX_VALID": self.get_bit(status, self.STATUS_RX_VALID),
            "RX_FULL": self.get_bit(status, self.STATUS_RX_FULL),
            "TX_EMPTY": self.get_bit(status, self.STATUS_TX_EMPTY),
            "TX_FULL": self.get_bit(status, self.STATUS_TX_FULL),
            "IS_INTR": self.get_bit(status, self.STATUS_IS_INT_EN),
            "OVERRUN_ERR": self.get_bit(status, self.STATUS_OVERRUN_ERR),
            "FRAME_ERR": self.get_bit(status, self.STATUS_FRAME_ERR),
            "PARITY_ERR": self.get_bit(status, self.STATUS_PARITY_ERR),
        }

    def is_rx_valid(self) -> bool:
        return bool(self.uart.read(self.STATUS_REG) & (1 << self.STATUS_RX_VALID))

    def is_tx_full(self) -> bool:
        return bool(self.uart.read(self.STATUS_REG) & (1 << self.STATUS_TX_FULL))

    def read(self, count: int, timeout: float = 3) -> str:
        buf = []
        stop_time = time() + timeout

        for _ in range(count):
            # Wait till RX fifo has valid data, stop waiting if timeout passes
            while (not self.is_rx_valid()) and (time() < stop_time):
                sleep(0.001)

            if time() >= stop_time:
                break

            d = self.uart.read(self.RX_FIFO) & 0xFF
            buf.append(chr(d))

        return "".join(buf)

    def read_bytes(self, count: int, timeout: float = 3) -> bytes:
        buf = []
        stop_time = time() + timeout

        for _ in range(count):
            # Wait till RX fifo has valid data, stop waiting if timeout passes
            while (not self.is_rx_valid()) and (time() < stop_time):
                sleep(0.001)

            if time() >= stop_time:
                break

            d = self.uart.read(self.RX_FIFO) & 0xFF
            buf.append(d)

        return bytes(buf)

    def write(self, buf: str | bytes | bytearray, timeout: float = 3) -> int:
        wr_count = 0
        stop_time = time() + timeout

        for char in buf:
            # Wait while TX FIFO is full, stop waiting if timeout passes
            while self.is_tx_full() and (time() < stop_time):
                sleep(0.001)

            if time() > stop_time:
                break

            self.uart.write(
                self.TX_FIFO, char & 0xFF if isinstance(char, int) else ord(char)
            )
            wr_count += 1

        return wr_count

    def write_hex(self, buf: str, timeout: float = 3) -> int:
        return self.write(buf, timeout)

    def write_bytes(self, buf: bytes, timeout: float = 3) -> int:
        return self.write(buf, timeout)

    def readline(self) -> str:
        buf = self.read(1)
        if len(buf) == 0:
            return ""
        while "\n" not in buf:
            buf += self.read(1)
        return buf
