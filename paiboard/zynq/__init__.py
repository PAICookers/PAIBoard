from .overlay import ON_ZYNQ
from .client import ZynqClient
from .paiboard_zynq import PAIBoardZynqClient

if ON_ZYNQ:
    from .paiboard_zynq import PAIBoardZynq
    from .local import ZynqLocal
    from .server import ZynqServer

__all__ = [
    "ZynqClient",
    "ZynqLocal",
    "ZynqServer",
    "PAIBoardZynq",
    "PAIBoardZynqClient",
]
