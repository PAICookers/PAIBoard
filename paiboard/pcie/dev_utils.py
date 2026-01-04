import mmap
import os
import sys
from contextlib import contextmanager
from enum import Enum, unique

import numpy as np


@unique
class XDMADevice(Enum):
    CTRL = "control"
    BYPASS = "bypass"
    H2C = "h2c"
    C2H = "c2h"


@contextmanager
def open_dev(device: str, flags: int):
    fd = os.open(device, flags)
    if fd < 0:
        raise IOError(f"device {device} open error")

    yield fd
    os.close(fd)


def send_dev(fd: int, buffer: np.ndarray) -> int:
    size = os.write(fd, buffer.tobytes())
    if size != buffer.nbytes:
        raise ValueError(f"expected send {buffer.nbytes} bytes, but sent {size} bytes")

    return size


def read_dev(fd: int, size: int) -> bytes:
    recv = os.read(fd, size)
    if (recv_len := len(recv)) != size:
        raise ValueError(f"expected read {size} bytes, but read {recv_len} bytes")

    return recv


def open_mmap(fd: int, length: int) -> mmap.mmap:
    if sys.platform == "win32":
        mm = mmap.mmap(fd, length)
    else:
        mm = mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    return mm
