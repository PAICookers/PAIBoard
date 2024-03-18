import __init__
import numpy as np

from paiboard.pcie.paiboard_pcie import PAIBoard_PCIe
from paiboard.simulator.paiboard_sim import PAIBoard_SIM
from paiboard.ethernet.paiboard_ethernet import PAIBoard_Ethernet

if __name__ == "__main__":
    timestep = 3
    layer_num = 0
    baseDir = "./result/01_bypass_net"
    snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.config(oFrmNum=20)

    test_num = 1
    for i in range(test_num):
        input_spike = np.eye(timestep, dtype=np.int8)
        output_spike = snn(input_spike, TimeMeasure=False)
        assert np.equal(input_spike, output_spike).all()
    print("Test passed!")

    # snn.dma_inst.show_reg_status()