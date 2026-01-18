"""
XDMA Controller Interface

Provides a high-level interface to Xilinx XDMA PCIe device using C library bindings.
Uses the C extensions (example.so) instead of pure Python device I/O for better reliability
and debugging of error conditions like "Unknown Error 512".
"""

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
from .dev_utils import XDMADevice, init_xdma, read_dev, send_dev

# Import C library functions with error handling
try:
    from .example import read_bypass, write_bypass
except ImportError as e:
    # Store error for later - will be raised when actually used
    _xdma_import_error = str(e)
    read_bypass = None
    write_bypass = None

__all__ = ["XDMACtrl"]


class XDMACtrl(HostCtrlInterface):
    """
    XDMA Controller using C library bindings.
    
    Replaces direct os.open/os.write/os.read with Xilinx XDMA C library functions
    for improved reliability in detecting and debugging communication errors.
    
    Error codes:
        - 512 (negative): Indicates potential buffer size mismatch, alignment issues,
          or device not ready. Common causes:
          - Buffer size not aligned to device requirements
          - DMA engine internal error
          - Device timeout or unresponsive state
          - Buffer allocation failure
    """

    xdma_dev_name: ClassVar[dict[XDMADevice, str]] = {
        XDMADevice.CTRL: "/dev/xdma{0}_user",
        XDMADevice.BYPASS: "/dev/xdma{0}_bypass",
        XDMADevice.H2C: "/dev/xdma{0}_h2c_{1}",
        XDMADevice.C2H: "/dev/xdma{0}_c2h_{1}",
    }
    
    REGFILE_BASE = 0x00000

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
        Initialize XDMA controller using C library.
        
        Args:
            board_cfg: The board configuration.
            xdma_dev_idx: The device index of XDMA (for reference). Defaults to 0.
            xdma_channel: The channel index of XDMA (for reference). Defaults to 0.
            n_max_oframe: Maximum output frames.
            timeout: Timeout in seconds for send/recv operations. Defaults to 2.
            
        Raises:
            PAIBoardDMADeviceError: If PCIe initialization fails.
        """
        self.running = False
        self.board_cfg = board_cfg
        self.dev_idx = xdma_dev_idx
        self.channel = xdma_channel
        self.timeout = timeout
        
        self.open()
        self.set_n_max_oframe(n_max_oframe)

    def open(self) -> None:
        """
        Initialize XDMA PCIe device via C library.
        
        Raises:
            PAIBoardDMADeviceError: If initialization fails.
        """
        if self.running:
            return
        
        # Check if C library import was successful
        if read_bypass is None or write_bypass is None:
            raise PAIBoardDMADeviceError(
                f"XDMA C library not available: {_xdma_import_error}\n"
                f"Please ensure:\n"
                f"  1. Xilinx XDMA driver is installed\n"
                f"  2. Python version matches .so file (currently using Python {__import__('sys').version_info.major}.{__import__('sys').version_info.minor})\n"
                f"  3. The .so file exists in paiboard/pcie/"
            )
        
        try:
            init_xdma()
            self.reset_regfile()
            self.running = True
        except Exception as e:
            raise PAIBoardDMADeviceError(
                f"Failed to initialize XDMA device: {e}"
            ) from e

    def reset_regfile(self) -> None:
        """Reset register file to default state."""
        write_bypass(self.REGFILE_BASE + RegFile.CPU2FIFO_CNT, 0)
        write_bypass(self.REGFILE_BASE + RegFile.FIFO2SNN_CNT, 0)
        write_bypass(self.REGFILE_BASE + RegFile.SNN2FIFO_CNT, 0)
        write_bypass(self.REGFILE_BASE + RegFile.FIFO2CPU_CNT, 0)

        write_bypass(self.REGFILE_BASE + RegFile.DATAPATH_RSTN, 0)
        write_bypass(self.REGFILE_BASE + RegFile.DATAPATH_RSTN, 1)

        write_bypass(self.REGFILE_BASE + RegFile.OEN, self.board_cfg.oen)
        write_bypass(self.REGFILE_BASE + RegFile.CHANNEL_MASK, self.board_cfg.channel_mask)

    @time_calc_addText("SendFrame     ")
    def send_frames(self, frames: FrameArrayType, **kwargs) -> int:
        """
        Send frames via DMA.
        
        Args:
            frames: Frame array to send (uint64, C-contiguous).
            **kwargs: Additional options (multi_channel_enable).
            
        Returns:
            int: Number of bytes sent.
            
        Raises:
            PAIBoardDMADeviceError: If send fails (error 512 indicates buffer/device issue).
            ValueError: If sent bytes don't match expected.
        """
        multi_channel_enable = bool(kwargs.setdefault("multi_channel_enable", False))
        write_bypass(
            self.REGFILE_BASE + RegFile.SINGLE_CHANNEL,
            0 if multi_channel_enable else 1
        )

        write_bypass(self.REGFILE_BASE + RegFile.SEND_LEN, frames.size)
        
        try:
            size = send_dev(frames, frames.nbytes)
        except PAIBoardDMADeviceError as e:
            raise PAIBoardDMADeviceError(
                f"DMA send failed (error 512 may indicate buffer alignment or device issue): {e}"
            ) from e

        if size != frames.nbytes:
            raise ValueError(
                f"send bytes mismatch: sent {size} != expected {frames.nbytes}"
            )

        # Wait for TX_STATE to become non-zero
        with wait_with_timeout(
            self.timeout, "Timeout waiting for TX_STATE to become 1"
        ) as timed_out:
            while read_bypass(self.REGFILE_BASE + RegFile.TX_STATE) == 0:
                if timed_out():
                    break
                time.sleep(0.0001)  # 100us

        write_bypass(self.REGFILE_BASE + RegFile.TX_STATE, 0)
        return size

    @time_calc_addText("RecvFrame     ")
    def recv_frames(self, size: int) -> FrameArrayType:
        """
        Receive frames via DMA.
        
        Args:
            size: Number of frames to receive.
            
        Returns:
            FrameArrayType: Received frames with filter list values removed.
            
        Raises:
            PAIBoardDMADeviceError: If recv fails (error 512 indicates buffer/device issue).
        """
        write_bypass(self.REGFILE_BASE + RegFile.RX_STATE, 1)

        try:
            recv_bytes = read_dev(size << 3)
        except PAIBoardDMADeviceError as e:
            raise PAIBoardDMADeviceError(
                f"DMA recv failed (error 512 may indicate buffer allocation or device issue): {e}"
            ) from e

        # Convert bytes to uint64 array
        recv_arr = np.frombuffer(recv_bytes, dtype=FRAME_DTYPE)
        frames = recv_arr[~np.isin(recv_arr, FRAME_VALUE_FILTER_LIST)]

        # Wait for RX_STATE to become zero
        with wait_with_timeout(
            self.timeout, "Timeout waiting for RX_STATE to become 0"
        ) as timed_out:
            while read_bypass(self.REGFILE_BASE + RegFile.RX_STATE) == 1:
                if timed_out():
                    break
                time.sleep(0.0001)  # 100us

        write_bypass(self.REGFILE_BASE + RegFile.RX_STATE, 0)
        return frames

    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType:
        """
        Send and receive frames in sequence.
        
        Args:
            frames: Frames to send.
            recv_size: Number of frames to receive. Defaults to n_max_oframe.
            **kwargs: Additional options for send_frames.
            
        Returns:
            FrameArrayType: Received frames.
        """
        kwargs.setdefault("multi_channel_enable", False)
        sent_size = self.send_frames(frames, **kwargs)
        if sent_size != frames.nbytes:
            raise ValueError(
                f"send size mismatch: expected {frames.nbytes}, got {sent_size}"
            )

        if recv_size is None:
            recv_size = self.n_max_oframe

        return self.recv_frames(recv_size)

    def write_reg(self, addr: int, value: int) -> int:
        """
        Write a 32-bit register value.
        
        Args:
            addr: Register address (absolute, including REGFILE_BASE offset).
            value: Value to write (32-bit).
            
        Returns:
            int: 0 on success.
        """
        write_bypass(addr, value)
        return 4

    def read_reg(self, addr: int) -> int:
        """
        Read a 32-bit register value.
        
        Args:
            addr: Register address (absolute, including REGFILE_BASE offset).
            
        Returns:
            int: Register value (32-bit).
        """
        return read_bypass(addr)

    def reset_chip(self, *chip_idx: int) -> None:
        """
        Chip reset is not supported in XDMA platform.
        
        Raises:
            PAIBoardPlatformNotSupportedWarning: Always warns since reset is not supported.
        """
        warnings.warn(
            "chip reset is not supported in XDMA PCIe platform.",
            PAIBoardPlatformNotSupportedWarning,
        )

    def close(self) -> None:
        """Close XDMA device (C library handles cleanup)."""
        self.running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
