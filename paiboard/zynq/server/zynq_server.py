import atexit
import signal
import socket
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from paicorelib.framelib import FRAME_DTYPE

from paiboard.common import RegFile
from paiboard.global_cfg import DEFAULT_N_OUTPUT_FRAMES

from ..overlay import ZynqPlatformOverlay
from ..zynq_packet import ZYNQ_PACKET_BYTEORDER, PayloadType, ZynqPacket
from ..zynq_packet import EnetCommand as CMD


class ZynqServer:
    sock: socket.socket
    client: socket.socket
    ol: ZynqPlatformOverlay
    n_max_oframe: int  # uint32, controlled by the client
    running: bool

    def __init__(
        self,
        ip: str,
        port: int,
        bitfile: Path | str,
        *,
        start_debug_bridge: bool = False,
    ) -> None:
        self.running = False
        self.ol = ZynqPlatformOverlay(bitfile, start_debug_bridge=start_debug_bridge)
        self.socket_addr = (ip, port)
        self.n_max_oframe = DEFAULT_N_OUTPUT_FRAMES
        self.open_server()
        self.reset_regfile()

        atexit.register(self._clean_up)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object) -> NoReturn:
        print(
            f"received signal {signal.Signals(signum).name}, shutting down gracefully..."
        )
        sys.exit(0)

    def _clean_up(self, reset_chip: bool = True) -> None:
        if not self.running:
            return

        self.running = False

        try:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
        except Exception:
            pass

        if reset_chip:
            self.ol.reset_chip()

        self.reset_regfile()

    def open_server(self) -> None:
        self.sock = socket.create_server(self.socket_addr)
        self.sock.settimeout(None)  # blocking mode
        self.running = True

    def close(self, reset_chip: bool = True) -> None:
        atexit.unregister(self._clean_up)
        self._clean_up(reset_chip)

    def reset_regfile(self) -> None:
        """Reset the registers in the DMA regfile."""
        return self.ol.reset_regfile()

    def run(self, reset_chip_after_close: bool = True) -> None:
        while self.running:
            client, addr = self.wait_for_connection()
            with client:
                while self.running:
                    try:
                        recv_cmd, recv_payload = ZynqPacket.recv_from_client(client)
                        self.process_cmd(recv_cmd, recv_payload, client)

                        if recv_cmd == CMD.QUIT:
                            print("Quit because of receiving QUIT command")
                            break
                    except Exception:
                        # print(e)
                        pass
                        # break

        print(f"connection with {addr} closed")
        self.close(reset_chip_after_close)

    def wait_for_connection(self) -> tuple[socket.socket, Any]:
        print(f"server listening on {self.socket_addr}")
        client, addr = self.sock.accept()
        print(f"connected by {addr}")
        return client, addr

    def _ack(self, client: socket.socket, cmd: CMD, payload: PayloadType = b"") -> None:
        packet = ZynqPacket.pack(cmd, payload)
        client.sendall(packet)

    def _handle_cmd_send(self, payload: bytes, client: socket.socket) -> None:
        """Send frames to the PL. Return SEND as an acknowledge."""
        arr = np.frombuffer(payload, dtype=FRAME_DTYPE)
        size = self.ol.send_frame(arr)
        if size != arr.nbytes:
            raise ValueError(f"send bytes {size} != frames bytes {arr.nbytes}")

        self._ack(client, CMD.SEND)

    def _handle_cmd_send_and_recv(self, payload: bytes, client: socket.socket) -> None:
        """The client wants to send a frame to the chips & expects to receive a frame from the chips.
        Return SEND_AND_RECV + received frames payload.
        """
        arr = np.frombuffer(payload, dtype=FRAME_DTYPE)
        size = self.ol.send_frame(arr)
        if size != arr.nbytes:
            raise ValueError(f"send bytes {size} != frames bytes {arr.nbytes}")

        recv_frames = self.ol.recv_frame(self.n_max_oframe)
        self._ack(client, CMD.SEND_AND_RECV, recv_frames)

    def _handle_cmd_write_reg(self, payload: bytes, client: socket.socket) -> None:
        """Write a register in the PL. Return WRITE_REG as an acknowledge."""
        arr = np.frombuffer(payload, dtype=FRAME_DTYPE)
        addr, value = int(arr[0]), int(arr[1])
        if addr == RegFile.OFAME_NUM_REG:
            self.n_max_oframe = value

        self.ol.write_reg(addr, value)
        self._ack(client, CMD.WRITE_REG)

    def _handle_cmd_read_reg(self, payload: bytes, client: socket.socket) -> None:
        """Read a register in the PL. Return READ_REG + value."""
        arr = np.frombuffer(payload, dtype=FRAME_DTYPE)
        addr = int(arr[0])
        value: int = self.ol.read_reg(addr, length=4)
        self._ack(
            client, CMD.READ_REG, value.to_bytes(4, byteorder=ZYNQ_PACKET_BYTEORDER)
        )

    def _handle_cmd_chip_reset(self, payload: bytes, client: socket.socket) -> None:
        """Reset the chip. Return CHIP_RESET as an acknowledge."""
        self.ol.reset_chip()
        self._ack(client, CMD.CHIP_RESET)

    def _handle_cmd_regfile_reset(self, payload: bytes, client: socket.socket) -> None:
        """Reset the chip & the regfile. Return REGFILE_RESET as an acknowledge."""
        self.reset_regfile()
        self._ack(client, CMD.REGFILE_RESET)

    def _handle_cmd_chip_uart(self, payload: bytes, client: socket.socket) -> None:
        """Send command to the UART. Return CHIP_UART + return value.

        Payload: uart_idx(1) + command(15).
        """
        assert len(payload) == 16
        uart_idx, cmd = payload[0], payload[1:]
        recv_str = self.ol.send_uart_cmd(uart_idx, cmd)

        print(f"UART {uart_idx} return: {recv_str}")
        self._ack(client, CMD.CHIP_UART, recv_str)

    def _handle_cmd_quit(self, payload: bytes, client: socket.socket) -> None:
        """The server quits."""
        self._ack(client, CMD.QUIT)
        self.running = False

    CMD_HANDLERS = {
        CMD.SEND: _handle_cmd_send,
        CMD.SEND_AND_RECV: _handle_cmd_send_and_recv,
        CMD.WRITE_REG: _handle_cmd_write_reg,
        CMD.READ_REG: _handle_cmd_read_reg,
        CMD.CHIP_RESET: _handle_cmd_chip_reset,
        CMD.CHIP_UART: _handle_cmd_chip_uart,
        CMD.REGFILE_RESET: _handle_cmd_regfile_reset,
        CMD.QUIT: _handle_cmd_quit,
    }

    def process_cmd(self, cmd: CMD, payload: bytes, client: socket.socket) -> None:
        return self.CMD_HANDLERS[cmd](self, payload, client)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(reset_chip=True)
