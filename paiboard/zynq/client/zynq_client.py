from typing import Literal
import atexit
import numpy as np
import socket

from paiboard.types import FRAME_DTYPE, FrameArrayType
from ..zynq_abc import ZynqCtrlInterface
from ..zynq_packet import (
    EnetCommand as CMD,
    PayloadType,
    ZynqPacket,
    ZYNQ_PACKET_BYTEORDER,
)


class ZynqClient(ZynqCtrlInterface):
    sock: socket.socket

    def __init__(
        self,
        ip: str,
        port: int,
        n_max_oframe: int | None = None,
        timeout: float | None = None,
        quit_the_server_when_exit: bool = False,
    ) -> None:
        self.running = False
        atexit.register(self.close, quit_the_server_when_exit)
        self.socket_addr = (ip, port)
        self.open(timeout)

        # Do after init
        self.set_n_max_oframe(n_max_oframe)

    def open(self, timeout: float | None = None) -> None:
        self.sock = socket.create_connection(self.socket_addr, timeout)
        self.running = True
        # NOTE: No need to reset the regfile, the server will reset it.

        print(f"connected to {self.socket_addr}")

    def close(self, quit_the_server: bool = False) -> None:
        if not self.running:
            return

        if quit_the_server:
            self.quit()

        self.sock.close()
        self.running = False

    def quit(self) -> None:
        """Quit the server. Format: QUIT.

        The server returns: QUIT.
        """
        recv_bytes = self._send_cmd(CMD.QUIT)
        assert recv_bytes == b""

    def reset_regfile(self) -> None:
        """Reset the chip & the regfile. Format: REGFILE_RESET.

        The server returns: REGFILE_RESET.
        """
        recv_bytes = self._send_cmd(CMD.REGFILE_RESET)
        assert recv_bytes == b""

    def write_reg(self, addr: int, value: int) -> None:
        """Write a register. Format: WRITE_REG + addr + value.

        The server returns: WRITE_REG.
        """
        payload = np.array([addr, value], dtype=FRAME_DTYPE)
        recv_bytes = self._send_payload(CMD.WRITE_REG, payload)
        assert recv_bytes == b""

    def read_reg(self, addr: int) -> int:
        """Read a register. Format: READ_REG + addr.

        The server returns: READ_REG + value.
        """
        payload = np.array([addr], dtype=FRAME_DTYPE)
        recv_bytes = self._send_payload(CMD.READ_REG, payload)
        return int.from_bytes(recv_bytes, byteorder=ZYNQ_PACKET_BYTEORDER)

    def reset_chip(self) -> None:
        """Reset the chip or chip array. Format: CHIP_RESET.

        The server returns: CHIP_RESET.
        """
        recv_bytes = self._send_cmd(CMD.CHIP_RESET)
        assert recv_bytes == b""

    def config_uart_chip(self, uart_idx: int, cmd: bytes) -> bytes:
        """Configure the chips via UART. Format: CHIP_UART + cmd.

        The server returns: CHIP_UART + cmd + debug info(if enabled).
        """
        return self._send_payload(
            CMD.CHIP_UART, uart_idx.to_bytes(1, byteorder=ZYNQ_PACKET_BYTEORDER) + cmd
        )

    def send_frames(self, frames: FrameArrayType, **kwargs) -> int:
        """Send frames to the server. Format: SEND + frames.

        The server returns: SEND.
        """
        recv_bytes = self._send_payload(CMD.SEND, frames)
        assert recv_bytes == b""
        return frames.nbytes

    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType:
        """Send frames to the server & receive. Format: SEND_AND_RECV + frames.

        The server returns: SEND_AND_RECV + received frames.
        """
        recv_bytes = self._send_payload(CMD.SEND_AND_RECV, frames)
        frames = np.frombuffer(recv_bytes, dtype=FRAME_DTYPE)
        return frames

    def _send_cmd(
        self, cmd: Literal[CMD.CHIP_RESET, CMD.QUIT, CMD.REGFILE_RESET]
    ) -> bytes:
        return self._send_payload(cmd)

    def _send_payload(self, cmd: CMD, payload: PayloadType = b"") -> bytes:
        packet = ZynqPacket.pack(cmd, payload)
        return self._send_and_wait_ack(cmd, packet)

    def _send_and_wait_ack(self, cmd: CMD, packet: bytes) -> bytes:
        """Server will echo the command + return bytes if needed."""
        self.sock.sendall(packet)
        ret_cmd, recv_bytes = ZynqPacket.recv_from_server(self.sock)

        if ret_cmd in ZynqPacket.ERR_CODE:
            raise ValueError(
                f"return error code: {ret_cmd.name}, but expected: {cmd.name}"
            )
        elif ret_cmd != cmd:
            raise ValueError(
                f"return unexpected cmd: {ret_cmd.name}, but expected: {cmd.name}"
            )

        return recv_bytes
