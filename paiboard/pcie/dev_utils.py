"""
PCIe XDMA Device Utilities

Provides low-level wrappers for XDMA C library functions to replace pure Python
os-based device I/O operations. Uses Xilinx XDMA driver C extensions.
"""

import sys
import warnings
from enum import Enum, unique
from pathlib import Path

import numpy as np

from ..exceptions import PAIBoardDMADeviceError

# Try to import XDMA C library with detailed error diagnostics
_XDMA_AVAILABLE = False
_XDMA_ERROR = None

try:
    from .example import pcie_init, read_bypass, write_bypass, send_dma_np, read_dma_np
    _XDMA_AVAILABLE = True
except ImportError as e:
    _XDMA_ERROR = e
    # Check for common issues
    so_file = Path(__file__).parent / "example.cpython-311-x86_64-linux-gnu.so"
    so_exists = so_file.exists()
    current_python = f"Python {sys.version_info.major}.{sys.version_info.minor}"
    
    error_msg = (
        f"Failed to import XDMA C library bindings:\n"
        f"  Error: {e}\n"
        f"  Python version: {current_python}\n"
        f"  .so file exists: {so_exists}\n"
        f"  System: {sys.platform}\n\n"
        f"Common causes:\n"
        f"  1. Python version mismatch:\n"
        f"     - .so file compiled for Python 3.11\n"
        f"     - Current: {current_python}\n"
        f"     - Solution: Recompile .so for Python {sys.version_info.major}.{sys.version_info.minor}\n"
        f"  2. Architecture mismatch (32-bit vs 64-bit)\n"
        f"  3. Missing Xilinx XDMA driver\n"
        f"  4. Incorrect file location\n"
    )
    _XDMA_ERROR = error_msg


@unique
class XDMADevice(Enum):
    """XDMA device types (for reference only, actual control is via C library)."""
    CTRL = "control"
    BYPASS = "bypass"
    H2C = "h2c"
    C2H = "c2h"


def _check_xdma_available() -> None:
    """Check if XDMA C library is available. Raise detailed error if not."""
    if not _XDMA_AVAILABLE:
        raise PAIBoardDMADeviceError(_XDMA_ERROR)


def init_xdma() -> None:
    """
    Initialize XDMA PCIe device via C library.
    
    Raises:
        PAIBoardDMADeviceError: If PCIe initialization fails or C library not available.
    """
    _check_xdma_available()
    rc = pcie_init()
    if rc < 0:
        raise PAIBoardDMADeviceError(
            f"PCIe XDMA initialization failed with error code {rc}"
        )


def send_dev(buffer: np.ndarray, num_bytes: int | None = None) -> int:
    """
    Send data via DMA (Host to Card).
    
    Args:
        buffer: NumPy array containing data to send (must be uint64, C-contiguous).
        num_bytes: Number of bytes to send. If None, use buffer.nbytes.
        
    Returns:
        int: Number of bytes sent.
        
    Raises:
        PAIBoardDMADeviceError: If DMA send fails (error code 512 may indicate
            buffer size mismatch or alignment issues).
    """
    if num_bytes is None:
        num_bytes = buffer.nbytes
    
    try:
        # Ensure buffer is the right type for C library
        if buffer.dtype != np.uint64:
            buffer = buffer.astype(np.uint64)
        if not buffer.flags['C_CONTIGUOUS']:
            buffer = np.ascontiguousarray(buffer, dtype=np.uint64)
        
        rc = send_dma_np(buffer, num_bytes)
        if rc < 0:
            # Error code 512 typically indicates device communication or buffer issue
            error_code = -rc
            raise PAIBoardDMADeviceError(
                f"XDMA send_dma_np failed: Unknown Error {error_code} (rc={rc}). "
                f"This may indicate: buffer alignment issues, size mismatch, or device not ready. "
                f"Attempted to send {num_bytes} bytes."
            )
        return rc
    except Exception as e:
        if isinstance(e, PAIBoardDMADeviceError):
            raise
        raise PAIBoardDMADeviceError(
            f"Failed to send DMA data: {e}"
        ) from e


def read_dev(num_bytes: int) -> bytes:
    """
    Read data via DMA (Card to Host).
    
    Args:
        num_bytes: Number of bytes to read.
        
    Returns:
        bytes: Data read from device.
        
    Raises:
        PAIBoardDMADeviceError: If DMA read fails (error code 512 may indicate
            buffer allocation or device issues).
    """
    try:
        rc, buffer = read_dma_np(num_bytes)
        if rc < 0:
            error_code = -rc
            raise PAIBoardDMADeviceError(
                f"XDMA read_dma_np failed: Unknown Error {error_code} (rc={rc}). "
                f"This may indicate: buffer allocation failure or device not responding. "
                f"Attempted to read {num_bytes} bytes."
            )
        return buffer.tobytes()
    except Exception as e:
        if isinstance(e, PAIBoardDMADeviceError):
            raise
        raise PAIBoardDMADeviceError(
            f"Failed to read DMA data: {e}"
        ) from e
