import numpy as np
import pytest

from paiboard.common import RegFile
from paiboard.global_cfg import CONFIG_FILE_DTYPE
from paiboard.pcie.xdma_ctrl import XDMACtrl
from tests.utils import file_not_exist_fail, skip_if_in_ci_env

from .conftest import has_xdma_device


def skip_if_no_device_or_in_ci_env() -> pytest.MarkDecorator:
    if not has_xdma_device():
        return pytest.mark.skip("XDMA device not found, skipping test")

    return skip_if_in_ci_env()


@skip_if_no_device_or_in_ci_env()
class TestXDMACtrl:
    def test_read_write_regfile(self, xdma_dev_init: XDMACtrl):
        xdma = xdma_dev_init

        v = 1234
        xdma.write_reg(RegFile.OFAME_NUM_REG, v)
        recv = xdma.read_reg(RegFile.OFAME_NUM_REG)
        assert recv == v

        v2 = 1 << RegFile.CHIP_TOP_INIT_BIT
        xdma.write_reg(RegFile.CHIP_TOP_CTRL, v2)
        recv = xdma.read_reg(RegFile.CHIP_TOP_CTRL)
        assert recv == v2

    def test_send_frames_cfg(self, xdma_dev_init: XDMACtrl, toolchain_build_dir):
        xdma = xdma_dev_init

        model_dir = toolchain_build_dir / "test_001_Conv1d"
        config_fp = model_dir / "config_all.bin"
        file_not_exist_fail(config_fp)

        cfg_frames = np.fromfile(config_fp, dtype=CONFIG_FILE_DTYPE)
        xdma.send_frames(cfg_frames)
