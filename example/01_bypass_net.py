import __init__
import numpy as np
import time
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

if __name__ == "__main__":
    timestep = 3
    layer_num = 0
    baseDir = "./result/01_bypass_net"
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.config(oFrmNum=20)

    test_num = 100
    for i in range(test_num):
        input_spike = np.eye(timestep, dtype=np.int8)

        t1 = time.time()
        output_spike = snn(input_spike, TimeMeasure=False)
        t2 = time.time()
    
        snn.record_time(t2 - t1)
        assert np.equal(input_spike, output_spike).all()
    print("Test passed!")
    snn.paicore_status()
    snn.perf(test_num)