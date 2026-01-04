try:
    import pynq  # pyright: ignore

    ON_ZYNQ = True
except ImportError:
    ON_ZYNQ = False

if ON_ZYNQ:
    from .base import ZynqPlatformOverlay

__all__ = ["ON_ZYNQ", "ZynqPlatformOverlay"]
