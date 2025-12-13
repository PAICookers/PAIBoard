import pytest
from pathlib import Path
from paiboard.board_cfg import get_board_cfg
from paiboard.exceptions import PAIBoardDMADeviceError
from paiboard.pcie.xdma_ctrl import XDMACtrl


def has_xdma_device() -> bool:
    """Check if any XDMA device is available."""
    return bool(list(Path("/dev").glob("xdma*")))


@pytest.fixture(scope="class")
def xdma_dev_init():
    board_name = "FLIP8"
    n_max_channel = 4
    board_cfg = get_board_cfg(board_name, n_max_channel)

    try:
        with XDMACtrl(board_cfg, 0, 0) as xdma:
            yield xdma
    except PAIBoardDMADeviceError:
        pytest.fail("failed to open device")
