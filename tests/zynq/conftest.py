import socket
import threading
from pathlib import Path

import pytest

from paiboard.zynq import ON_ZYNQ, ZynqClient
from paiboard.zynq.utils import auto_search_bitfile
from paiboard.zynq.zynq_packet import EnetCommand as CMD
from paiboard.zynq.zynq_packet import ZynqPacket

# Default path of FPGA bitstream & ovarlay hwh file for testing
FPGA_BITFILE_PATH = Path(__file__).parent / "fpga"
DEFAULT_ZYNQ_SERVER_IP = "192.168.31.99"
DEFAULT_ZYNQ_SERVER_PORT = 8080


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--bitfile",
        action="store",
        default=FPGA_BITFILE_PATH,
        help="Specify the bitfile path for zynq tests",
    )
    parser.addoption(
        "--port",
        action="store",
        default=DEFAULT_ZYNQ_SERVER_PORT,
        type=int,
        help=f"Specify the port for zynq server connection(default: {DEFAULT_ZYNQ_SERVER_PORT})",
    )
    parser.addoption(
        "--ip",
        action="store",
        default=DEFAULT_ZYNQ_SERVER_IP,
        type=str,
        help=f"Specify the IP address for zynq server connection(default: {DEFAULT_ZYNQ_SERVER_IP})",
    )


class MockZynqServer:
    def __init__(self, host: str = "localhost", port: int = 0):
        self.host = host
        self.sock = socket.create_server((host, port))
        self.port = self.sock.getsockname()[1]
        self.running = False
        self.server_thread = None
        self.client = None

        self.response_hdlrs = dict()

    def set_handler(self, command, handler) -> None:
        self.response_hdlrs[command] = handler

    def default_response(self, command, payload):
        if command == CMD.WRITE_REG:
            return b""
        elif command == CMD.READ_REG:
            return (0).to_bytes(4)
        elif command == CMD.CHIP_RESET:
            return b""
        elif command == CMD.CHIP_UART:
            return b"OK"
        elif command == CMD.SEND:
            return b""
        elif command == CMD.SEND_AND_RECV:
            return payload
        elif command == CMD.QUIT:
            return b""
        else:
            return b""

    def handle_client(self):
        self.client, _ = self.sock.accept()

        with self.client:
            while self.running:
                try:
                    recv_cmd, recv_payload = ZynqPacket.recv_from_client(self.client)

                    if recv_cmd in self.response_hdlrs:
                        response_data = self.response_hdlrs[recv_cmd](recv_payload)
                    else:
                        response_data = self.default_response(recv_cmd, recv_payload)

                    packet = ZynqPacket.pack(recv_cmd, response_data)
                    self.client.sendall(packet)

                    if recv_cmd == CMD.QUIT:
                        break

                except Exception:
                    break

    def start(self) -> None:
        self.running = True
        self.server_thread = threading.Thread(target=self.handle_client, daemon=True)
        self.server_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=1)

        self.sock.close()


@pytest.fixture
def mock_server():
    server = MockZynqServer()
    server.start()
    yield server

    server.stop()


@pytest.fixture
def zynq_client(mock_server):
    client = ZynqClient(mock_server.host, mock_server.port)
    assert client.running
    yield client

    client.close()


@pytest.fixture(scope="class")
def zynq_client_instance(request):
    ip = request.config.getoption("--ip")
    port = request.config.getoption("--port")
    try:
        client = ZynqClient(ip, port)
        assert client.running
    except Exception:
        pytest.skip(f"Failed to connect to zynq server at {ip}:{port}")

    yield client
    client.close(quit_the_server=False)


@pytest.fixture(autouse=True)
def zynq_client_reset_every_test(zynq_client_instance):
    # Reset chip & regfile after each test
    zynq = zynq_client_instance
    yield zynq

    zynq.reset_chip()
    zynq.reset_regfile()
    print("zynq reset chip & regfile")


if ON_ZYNQ:
    from paiboard.zynq import ZynqLocal

    @pytest.fixture(scope="class")
    def zynq_local_instance(request):
        bitfile_fp = Path(request.config.getoption("--bitfile"))
        bitfile = auto_search_bitfile(bitfile_fp)
        print("Using bitfile:", bitfile)

        zynq = ZynqLocal(bitfile, start_debug_bridge=False)
        assert zynq.running
        yield zynq

        zynq.close()
        print("zynq closed")

    @pytest.fixture(autouse=True)
    def zynq_local_reset_every_test(zynq_local_instance):
        # Reset chip & regfile after each test
        zynq = zynq_local_instance
        yield zynq

        zynq.reset_chip()
        zynq.reset_regfile()
        print("zynq reset chip & regfile")
