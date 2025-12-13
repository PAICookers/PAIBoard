# from .simulator.paiboard_sim import PAIBoard_SIM as PAIBoard_SIM
# from .pcie.paiboard_pcie import PAIBoard_PCIe as PAIBoard_PCIe
# from .ethernet.paiboard_ethernet import PAIBoard_Ethernet as PAIBoard_Ethernet
# from .zynq import PAIBoard_Arm as PAIBoard_Arm
from .zynq import ON_ZYNQ

if ON_ZYNQ:
    from .zynq import PAIBoardZynq
else:
    from .zynq import PAIBoardZynqClient

from .pcie import PAIBoardPCIe
