import numpy as np
from paicorelib.framelib import FRAME_DTYPE

from paiboard.common.uart_cfg import ChipUartCfg
from paiboard.zynq import ZynqClient
from paiboard.zynq.zynq_packet import EnetCommand as CMD


class TestZynqClientMock:
    def test_read_reg(self, zynq_client: ZynqClient, mock_server):
        def read_reg_response(payload):
            data = np.frombuffer(payload, dtype=FRAME_DTYPE)
            assert len(data) == 1
            assert data[0] == 0x100
            return (0xABCD).to_bytes(4)

        mock_server.set_handler(CMD.READ_REG, read_reg_response)

        value = zynq_client.read_reg(0x100)
        assert value == 0xABCD

    def test_config_uart_chip(self, zynq_client, mock_server):
        received_data = {}

        def uart_handler(payload):
            received_data["uart_idx"] = payload[0]
            received_data["cmd"] = payload[1:]
            if received_data["cmd"][-1] & 0x1 == 1:
                return received_data["cmd"] + bytes([0xFF] * 19)
            else:
                return received_data["cmd"]

        mock_server.set_handler(CMD.CHIP_UART, uart_handler)

        uart_idx = 0
        uart_cmd = ChipUartCfg.gen_uart_cmd((0, 0), debug=False)
        response = zynq_client.config_uart_chip(uart_idx, uart_cmd)

        assert received_data["uart_idx"] == uart_idx
        assert len(received_data["cmd"]) == 15
        assert response == uart_cmd

        uart_id2 = 2
        uart_cmd2 = ChipUartCfg.gen_uart_cmd((0, 0), debug=True)
        response2 = zynq_client.config_uart_chip(uart_id2, uart_cmd2)

        assert received_data["uart_idx"] == uart_id2
        assert len(received_data["cmd"]) == 15
        assert len(response2) == 15 + 19

    def test_send_frames(self, zynq_client, mock_server):
        test_frames = np.array([1, 2, 3, 4, 5], dtype=FRAME_DTYPE)
        received_frames = {}

        def send_handler(payload):
            received_frames["data"] = np.frombuffer(payload, dtype=FRAME_DTYPE)
            return b""

        mock_server.set_handler(CMD.SEND, send_handler)

        nbytes = zynq_client.send_frames(test_frames)

        assert np.array_equal(received_frames["data"], test_frames)
        assert nbytes == test_frames.nbytes

    def test_send_and_recv_frames(self, zynq_client, mock_server):
        test_frames = np.array([1, 2, 3, 4, 5], dtype=FRAME_DTYPE)
        expected_response = np.array([6, 7, 8, 9, 10], dtype=FRAME_DTYPE)
        received_frames = {}

        def send_and_recv_handler(payload):
            received_frames["sent"] = np.frombuffer(payload, dtype=FRAME_DTYPE)
            return expected_response.tobytes()

        mock_server.set_handler(CMD.SEND_AND_RECV, send_and_recv_handler)

        response_frames = zynq_client.send_and_recv_frames(test_frames)

        assert np.array_equal(received_frames["sent"], test_frames)
        assert np.array_equal(response_frames, expected_response)


class TestZynqClient:
    pass
