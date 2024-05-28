import __init__

import numpy as np
import time
import os

import example.datasets.load_mnist as load_mnist
from example.Encoder.encoder import PoissonEncoder

from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

if __name__ == "__main__":
    timestep = 8
    layer_num = 1
    baseDir = "./result/fc_net/03.1_fcnet_2layer_dual_port/"
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.config(oFrmNum=40)

    dataset_root = os.path.join(os.path.expanduser("~"), "work/99_datasets/MNIST/raw")
    testdata, testlabels = load_mnist.CreatData(dataset_root)
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    print()
    img_num = 10
    correct = 0

    for i in range(img_num):
        input_data = testdata[i].reshape(1, 28, 28)
        input_spike = PoissonEncoder(x=input_data, timesteps=timestep).reshape(
            timestep, 784
        )

        t1 = time.time()
        output_spike_dict = snn(
            [input_spike, input_spike], TimeMeasure=True
        )
        t2 = time.time()

        snn.record_time(t2 - t1)
        output_spike1 = output_spike_dict["layer2_dual_port_o1"]
        output_spike2 = output_spike_dict["layer2_dual_port_o2"]
        output_spike = output_spike1
        if output_spike is not None:
            pred = np.argmax(output_spike.sum(axis=0))
        if pred == testlabels[i]:
            correct += 1

    print(f"{img_num}/{img_num} Acc:{round(correct / img_num, 4)}")

    snn.perf(img_num)
