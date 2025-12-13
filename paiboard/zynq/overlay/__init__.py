try:
    import pynq

    ON_ZYNQ = True
except ImportError:
    ON_ZYNQ = False

if ON_ZYNQ:
    from .base import ZynqPlatformOverlay
