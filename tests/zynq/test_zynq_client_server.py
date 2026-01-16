import numpy as np
import pytest
from paicorelib import Coord

from paiboard.common import ChipUartCfg, RegFile
from paiboard.global_cfg import CFG_FILE_DTYPE
from paiboard.zynq import ZynqClient
from tests.utils import file_not_exist_fail


class TestZynqClientServer:
    def test_read_write_regfile(self, zynq_client_instance: ZynqClient):
        zynq = zynq_client_instance

        v = 1234
        zynq.write_reg(RegFile.OFAME_NUM_REG, v)
        recv = zynq.read_reg(RegFile.OFAME_NUM_REG)
        assert recv == v

        v2 = 1 << RegFile.CHIP_TOP_INIT_BIT
        zynq.write_reg(RegFile.CHIP_TOP_CTRL, v2)
        recv = zynq.read_reg(RegFile.CHIP_TOP_CTRL)
        assert recv == v2

    @pytest.mark.parametrize("clk_freq", [192, 240, 288, 312, 360])
    def test_config_uart_chip(self, zynq_client_instance: ZynqClient, clk_freq):
        zynq = zynq_client_instance
        chip_list = [Coord(0, 0), Coord(0, 1), Coord(1, 1), Coord(1, 0)]

        for i, chip in enumerate(chip_list):
            # uart debug on
            uart_cmd = ChipUartCfg.gen_uart_cmd(chip, clk_freq, debug=True)
            resp = zynq.config_uart_chip(i, uart_cmd)
            assert len(resp) == len(uart_cmd) + 19
            assert resp[:-19] == uart_cmd

            # uart debug off
            uart_cmd2 = ChipUartCfg.gen_uart_cmd(chip, clk_freq, debug=False)
            resp2 = zynq.config_uart_chip(i, uart_cmd2)
            assert resp2 == uart_cmd2

    def test_send_frames_cfg(
        self, zynq_client_instance: ZynqClient, toolchain_build_dir
    ):
        zynq = zynq_client_instance

        # Target is (0,0), output to (2,0)
        model_dir = toolchain_build_dir / "test_001_Conv1d"
        config_fp = model_dir / "config_all.bin"
        file_not_exist_fail(config_fp)

        cfg_frames = np.fromfile(config_fp, dtype=CFG_FILE_DTYPE)

        # UART
        chip_list = [Coord(0, 0), Coord(0, 1), Coord(1, 1), Coord(1, 0)]
        for i, chip in enumerate(chip_list):
            # uart debug off
            uart_cmd = ChipUartCfg.gen_uart_cmd(chip, 240)
            resp = zynq.config_uart_chip(i, uart_cmd)
            assert resp == uart_cmd

        zynq.send_frames(cfg_frames)
