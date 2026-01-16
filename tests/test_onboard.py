import time

from paicorelib import Coord

from paiboard import PAIBoardZynqClient
from tests.utils import file_not_exist_fail
from tests.zynq.conftest import DEFAULT_ZYNQ_SERVER_IP, DEFAULT_ZYNQ_SERVER_PORT


class TestOnBoard:
    def test_online_cores(self, toolchain_build_dir):
        # Target is (0,0), output to (2,0)
        model_dir = toolchain_build_dir / "test_001_STDPLinear"
        config_fp = model_dir / "config_all.bin"
        file_not_exist_fail(config_fp)

        paiboard = PAIBoardZynqClient(
            model_dir,
            1,
            2,
            DEFAULT_ZYNQ_SERVER_IP,
            DEFAULT_ZYNQ_SERVER_PORT,
            batch_mode=False,
        )
        chip_list = [Coord(0, 0), Coord(0, 1), Coord(1, 1), Coord(1, 0)]

        paiboard.chip_uart_config(0, chip_list[0])
        paiboard.set_n_max_oframe(100)
        paiboard.chip_hw_config()

        print("Enable learning mode")
        paiboard.learning_mode(enable=True)

        time.sleep(1)
        print("Disable learning mode")
        paiboard.learning_mode(enable=False)

        # FIXME If not quiting the server, the next connection is OK but sending will fail
        paiboard.close(quit_the_server=True)
