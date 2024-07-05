import __init__
import os
import json
import numpy as np
import time
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

from paiboard.utils.utils_for_frame import frame_np2txt

if __name__ == "__main__":

    layer_list = [1, 10, 20, 30, 40, 50, 63, 64, 63 * 2, 63 * 3, 63 * 4]
    layer_list = [1]

    for layer in layer_list:

        baseDir = "./result/bypass_linear/bypass_linear_C" + str(layer*16)

        netInfoPath = os.path.join(baseDir, "net_info.json")
        with open(netInfoPath, "r", encoding="utf8") as fp:
            netInfo = json.load(fp)

        timestep = netInfo["timesteps"]
        layer_num = netInfo["layer_num"]
        oFrmNum = netInfo["max_spike_num"] * 2

        snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
        # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
        # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)

        snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
        snn.config(oFrmNum=oFrmNum)
        # snn.config(oFrmNum=oFrmNum, send=False)

        input_spike = np.ones((timestep, 1000), dtype=np.int8)
        output_spike = snn(input_spike, TimeMeasure=False)
        print(output_spike.sum())

        snn.paicore_status()