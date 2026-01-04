import binascii
import socket
from enum import IntEnum, unique

import numpy as np

PayloadType = bytes | np.ndarray
ZYNQ_PACKET_BYTEORDER = "big"  # x86_64 default


def recv_exact_bytes(client: socket.socket, n_bytes: int) -> bytes:
    assert n_bytes > 0

    data = b""
    while len(data) < n_bytes:
        chunk = client.recv(n_bytes - len(data))
        if not chunk:
            raise ConnectionError("socket connection closed during receiving")
        data += chunk

    return data


@unique
class EnetCommand(IntEnum):
    """Command codes, unsigned int8."""

    SEND = 0  # Send to the server
    SEND_AND_RECV = 1  # Send to the server & ask for a response
    WRITE_REG = 3
    READ_REG = 4
    CHIP_RESET = 5
    CHIP_UART = 6
    REGFILE_RESET = 7
    QUIT = 99

    HW_ERROR = 255
    CRC_ERROR = 254
    INVALID_CMD = 253


class ZynqPacket:
    """Application layer protocol on Zynq platform"""

    N_BYTE_CMD = 1
    N_BYTE_SIZE = 4
    N_BYTE_CRC = 4
    ERR_CODE = (EnetCommand.HW_ERROR, EnetCommand.CRC_ERROR, EnetCommand.INVALID_CMD)

    @classmethod
    def pack(cls, cmd: EnetCommand, payload: PayloadType = b"") -> bytes:
        """Packs the packet. Format: cmd(1) + size in bytes(4) + payload(N) + checksum(4)."""
        if isinstance(payload, np.ndarray):
            payload = payload.tobytes()

        # NOTE: in py3.10, argument 'byteorder' doesn't has a default value
        cmd_bytes = cmd.value.to_bytes(cls.N_BYTE_CMD, byteorder=ZYNQ_PACKET_BYTEORDER)
        size_bytes = len(payload).to_bytes(
            cls.N_BYTE_SIZE, byteorder=ZYNQ_PACKET_BYTEORDER
        )

        main_data = cmd_bytes + size_bytes + payload
        crc32_bytes = binascii.crc32(main_data).to_bytes(
            cls.N_BYTE_CRC, byteorder=ZYNQ_PACKET_BYTEORDER
        )

        return main_data + crc32_bytes

    @classmethod
    def recv_from_server(cls, server: socket.socket) -> tuple[EnetCommand, bytes]:
        """Receive cmd + value from the server. Only called by the client."""
        cmd_byte = recv_exact_bytes(server, cls.N_BYTE_CMD)

        try:
            cmd = EnetCommand(cmd_byte[0])
        except ValueError:
            return EnetCommand.INVALID_CMD, b""

        payload, recv_crc32, exp_crc32 = cls._recv_and_parse_payload(cmd_byte, server)
        if cmd in cls.ERR_CODE:
            return cmd, b""

        if recv_crc32 != exp_crc32:
            raise ValueError(f"CRC mismatch: {recv_crc32} != {exp_crc32}")

        return cmd, payload

    @classmethod
    def recv_from_client(cls, client: socket.socket) -> tuple[EnetCommand, bytes]:
        """Receive command from the client. Only called by the server."""
        cmd_byte = recv_exact_bytes(client, cls.N_BYTE_CMD)

        try:
            cmd = EnetCommand(cmd_byte[0])
        except ValueError:
            return EnetCommand.INVALID_CMD, b""

        payload, recv_crc32, exp_crc32 = cls._recv_and_parse_payload(cmd_byte, client)
        if cmd in cls.ERR_CODE:
            return EnetCommand.INVALID_CMD, b""

        if recv_crc32 != exp_crc32:
            return EnetCommand.CRC_ERROR, b""

        return cmd, payload

    @classmethod
    def _recv_and_parse_payload(
        cls, cmd_byte: bytes, client: socket.socket
    ) -> tuple[bytes, int, int]:
        size_bytes = recv_exact_bytes(client, cls.N_BYTE_SIZE)
        payload_size = int.from_bytes(size_bytes, byteorder=ZYNQ_PACKET_BYTEORDER)

        payload = recv_exact_bytes(client, payload_size) if payload_size > 0 else b""

        recv_crc_bytes = recv_exact_bytes(client, cls.N_BYTE_CRC)
        recv_crc32 = int.from_bytes(recv_crc_bytes, byteorder=ZYNQ_PACKET_BYTEORDER)

        main_data = cmd_byte + size_bytes + payload
        exp_crc32 = binascii.crc32(main_data)

        return payload, recv_crc32, exp_crc32
