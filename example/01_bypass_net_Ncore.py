import __init__
import os
import json
import numpy as np
import time
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

if __name__ == "__main__":
    net_num_list = [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1008,
    ]
    net_num_list = [100]
    
    for core_num in net_num_list:
        baseDir = "./result/bypass_net/bypass_net_" + str(core_num) + "core"

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

        coreInfoPath = os.path.join(baseDir, "core_params.json")
        with open(coreInfoPath, "r", encoding="utf8") as fp:
            all_core_params = json.load(fp)
        for chip_addr in all_core_params:
            core_params = all_core_params[chip_addr]
        print(f"core_nums: {len(core_params)}")
        test_num = 1
        for i in range(test_num):

            input_spike_list = [
                np.eye(timestep, dtype=np.int8) for _ in range(len(core_params))
            ]
            # input_spike_list = [np.random.randint(0, 2, (timestep, timestep)) for _ in range(len(core_params))]

            t1 = time.time()

            output_spike_list = []
            output_spike_dict = snn(input_spike_list)
            for output_spike in output_spike_dict:
                output_spike_list.append(output_spike_dict[output_spike])

            t2 = time.time()
            snn.record_time(t2 - t1)

            snn.paicore_status()

            for list_id in range(len(output_spike_list)):
                assert np.equal(
                    input_spike_list[list_id], output_spike_list[list_id]
                ).all(), list_id
            print(f"Test {i} passed.")

        snn.perf(test_num)
