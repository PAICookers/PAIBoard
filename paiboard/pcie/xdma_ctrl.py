import mmap
import os
import sys
import time
import warnings
from typing import ClassVar

import numpy as np
from paicorelib.framelib import FRAME_DTYPE, FrameArrayType

from ..board_cfg import BoardCfg
from ..common import HostCtrlInterface, RegFile
from ..exceptions import PAIBoardDMADeviceError, PAIBoardPlatformNotSupportedWarning
from ..global_cfg import FRAME_VALUE_FILTER_LIST
from ..utils import time_calc_addText, wait_with_timeout
from .dev_utils import XDMADevice, open_dev, open_mmap, read_dev, send_dev

__all__ = ["XDMACtrl"]


class XDMACtrl(HostCtrlInterface):
    xdma_dev_name: ClassVar[dict[XDMADevice, str]] = {
        XDMADevice.CTRL: "/dev/xdma{0}_user",
        XDMADevice.BYPASS: "/dev/xdma{0}_bypass",
        XDMADevice.H2C: "/dev/xdma{0}_h2c_{1}",
        XDMADevice.C2H: "/dev/xdma{0}_c2h_{1}",
    }
    BYPASS_MAP_SIZE = 256 * 1024
    h2c_fd: int
    c2h_fd: int
    ctrl_fd: int
    bypass_fd: int
    bypass_mm: mmap.mmap

    def __init__(
        self,
        board_cfg: BoardCfg,
        xdma_dev_idx: int = 0,
        xdma_channel: int = 0,
        *,
        n_max_oframe: int | None = None,
        timeout: int = 2,
    ) -> None:
        """
        Args:
            board_cfg: The board configuration.
            regfile_base_addr: The base address of register file in bypass space. Defaults to 0.
            xdma_dev_idx: The device index of XDMA, such as 0 for '/dev/xdma0_*'. Defaults to 0.
            xdma_channel: The channel index of XDMA, such as 0 for '/dev/xdma*_h2c_0'. Defaults to 0.
            timeout: Timeout in seconds for send/recv operations. Defaults to 2.
        """
        self.running = False
        self.board_cfg = board_cfg

        self.dev_idx = xdma_dev_idx
        self.channel = xdma_channel
        self.h2c_fd = -1
        self.c2h_fd = -1
        self.ctrl_fd = -1
        self.bypass_fd = -1

        self.timeout = timeout

        self.open()
        self.set_n_max_oframe(n_max_oframe)

    def _get_dev_name(self, dev: XDMADevice) -> str:
        if dev in (XDMADevice.CTRL, XDMADevice.BYPASS):
            return self.xdma_dev_name[dev].format(self.dev_idx)
        else:
            return self.xdma_dev_name[dev].format(self.dev_idx, self.channel)

    def open(self) -> None:
        if self.running:
            return

        name_user = self._get_dev_name(XDMADevice.CTRL)
        if (fd := os.open(name_user, os.O_RDWR | os.O_SYNC)) < 0:
            raise PAIBoardDMADeviceError("Control file descriptor not open")
        else:
            self.ctrl_fd = fd

        name_bypass = self._get_dev_name(XDMADevice.BYPASS)
        if (fd := os.open(name_bypass, os.O_RDWR | os.O_SYNC)) < 0:
            raise PAIBoardDMADeviceError("Bypass file descriptor not open")
        else:
            self.bypass_fd = fd
            self.bypass_mm = open_mmap(self.bypass_fd, self.BYPASS_MAP_SIZE)

        name_h2c = self._get_dev_name(XDMADevice.H2C)
        if (fd := os.open(name_h2c, os.O_WRONLY)) < 0:
            raise PAIBoardDMADeviceError("H2C file descriptor not open")
        else:
            self.h2c_fd = fd

        name_c2h = self._get_dev_name(XDMADevice.C2H)
        if (fd := os.open(name_c2h, os.O_RDONLY)) < 0:
            raise PAIBoardDMADeviceError("C2H file descriptor not open")
        else:
            self.c2h_fd = fd

        self.reset_regfile()
        self.running = True

    def reset_regfile(self) -> None:
        self.write_reg(RegFile.CPU2FIFO_CNT, 0)
        self.write_reg(RegFile.FIFO2SNN_CNT, 0)
        self.write_reg(RegFile.SNN2FIFO_CNT, 0)
        self.write_reg(RegFile.FIFO2CPU_CNT, 0)
        self.write_reg(RegFile.DATAPATH_RSTN, 0x1)
        self.write_reg(RegFile.SINGLE_CHANNEL, 0x1)

        self.write_reg(RegFile.OEN, self.board_cfg.oen)
        self.write_reg(RegFile.CHANNEL_MASK, self.board_cfg.channel_mask)

    @time_calc_addText("SendFrame     ")
    def send_frames(self, frames: FrameArrayType, **kwargs) -> int:
        kwargs.setdefault("multi_channel_enable", False)
        self.write_reg(RegFile.SINGLE_CHANNEL, int(~kwargs["multi_channel_enable"]))

        self.write_reg(RegFile.SEND_LEN, frames.size)
        size = self._send_dma(frames)
        if size != frames.nbytes:
            raise ValueError(f"send bytes {size} != frames bytes {frames.nbytes}")

        with wait_with_timeout(
            self.timeout, "Timeout waiting for TX_STATE to become 0"
        ) as timed_out:
            while self.read_reg(RegFile.TX_STATE) == 0:
                if timed_out():
                    break
                time.sleep(0.0001)  # 100us

        self.write_reg(RegFile.TX_STATE, 0)
        return size

    @time_calc_addText("RecvFrame     ")
    def recv_frames(self, size: int) -> FrameArrayType:
        self.write_reg(RegFile.RX_STATE, 1)

        recv_arr = self._read_dma(size << 3)
        frames = recv_arr[~np.isin(recv_arr, FRAME_VALUE_FILTER_LIST)]

        with wait_with_timeout(
            self.timeout, "Timeout waiting for RX_STATE to become 0"
        ) as timed_out:
            while self.read_reg(RegFile.RX_STATE) == 1:
                if timed_out():
                    break
                time.sleep(0.0001)  # 100us

        self.write_reg(RegFile.RX_STATE, 0)
        return frames

    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType:
        kwargs.setdefault("multi_channel_enable", False)
        sent_size = self.send_frames(frames, **kwargs)
        if sent_size != frames.size:
            raise ValueError(
                f"send size mismatch: expected {frames.size}, got {sent_size}"
            )

        if recv_size is None:
            recv_size = self.n_max_oframe

        return self.recv_frames(recv_size)

    def write_reg(self, addr: int, value: int) -> int:
        if self.bypass_mm is None:
            with open_mmap(self.bypass_fd, self.BYPASS_MAP_SIZE) as mm:
                mm.seek(addr)
                size = mm.write(value.to_bytes(4, byteorder=sys.byteorder))
        else:
            self.bypass_mm.seek(addr)
            size = self.bypass_mm.write(value.to_bytes(4, byteorder=sys.byteorder))

        if size != 4:
            raise ValueError(f"write size mismatch: expected 4, got {size}")

        return size

    def read_reg(self, addr: int) -> int:
        assert self.bypass_mm is not None

        self.bypass_mm.seek(addr)
        # NOTE: in py3.10, argument 'byteorder' doesn't has a default value
        return int.from_bytes(self.bypass_mm.read(4), byteorder=sys.byteorder)

    def reset_chip(self, *chip_idx: int) -> None:
        warnings.warn(
            "chip reset is not supported in this platform.",
            PAIBoardPlatformNotSupportedWarning,
        )

    def _send_dma(self, buffer: np.ndarray) -> int:
        if self.h2c_fd > 0:
            return send_dev(self.h2c_fd, buffer)
        else:
            devname = self._get_dev_name(XDMADevice.H2C)
            with open_dev(devname, os.O_WRONLY) as fd:
                size = send_dev(fd, buffer)

            return size

    def _read_dma(self, size: int) -> FrameArrayType:
        if self.c2h_fd > 0:
            recv = read_dev(self.c2h_fd, size)
        else:
            devname = self._get_dev_name(XDMADevice.C2H)
            with open_dev(devname, os.O_RDONLY) as fd:
                recv = read_dev(fd, size)

        return np.frombuffer(recv, dtype=FRAME_DTYPE)

    def close(self) -> None:
        if not self.running:
            return

        if self.bypass_fd > 0:
            os.close(self.bypass_fd)

        if self.bypass_mm:
            self.bypass_mm.close()

        if self.ctrl_fd > 0:
            os.close(self.ctrl_fd)

        if self.h2c_fd > 0:
            os.close(self.h2c_fd)

        if self.c2h_fd > 0:
            os.close(self.c2h_fd)

        self.running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
