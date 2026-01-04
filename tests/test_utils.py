import time

import pytest

from paiboard.exceptions import PAIBoardTimeoutError
from paiboard.utils import wait_with_timeout


def test_wait_with_timeout():
    with pytest.raises(PAIBoardTimeoutError):
        with wait_with_timeout(0.5, "timeout"):
            time.sleep(1)
