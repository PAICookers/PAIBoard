import __init__
import numpy as np

from paiboard import PAIBoard_SIM
# from paiboard import PAIBoard_PCIe
# from paiboard import PAIBoard_Ethernet


if __name__ == "__main__":
    timestep = 3
    layer_num = 1
    baseDir = "./result/02_bypass_net_2layer"
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.config(oFrmNum=20)

    test_num = 1
    for i in range(test_num):
        input_spike = np.eye(timestep, dtype=np.int8)
        output_spike = snn(input_spike, TimeMeasure=False)
        assert np.equal(input_spike, output_spike).all()
    print("Test passed!")

    snn.perf(test_num)