import __init__
import os
import json
import numpy as np
import time
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

import example.datasets.load_mnist as load_mnist
from example.Encoder.encoder import ConvSpikeEncoder

if __name__ == "__main__":

    mnist_example_list = [
        "conv_2layer",
        "conv_2layer_bn",
        "conv_avgpool",
        "ResNet",
    ]

    for name in mnist_example_list:
        print()
        print("------------------ " + name + "  ------------------")
        baseDir = "./result/paiflow_example/mnist/" + name

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

        dataset_root = os.path.join(
            os.path.expanduser("~"), "work/99_datasets/MNIST/raw"
        )
        testdata, testlabels = load_mnist.CreatData(dataset_root)
        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

        weight = np.load(os.path.join(baseDir, "spike_encode/weight_conv1.npy"))
        bias = np.load(os.path.join(baseDir, "spike_encode/bias_conv1.npy"))
        Vthr = np.load(os.path.join(baseDir, "spike_encode/Vthr_conv1.npy"))
        spikeEncoder = ConvSpikeEncoder(
            weight, bias, Vthr, padding=[0, 0], stride=[1, 1]
        )
        print()

        img_num = 1
        correct = 0

        for i in range(img_num):
            input_data = testdata[i].reshape(1, 28, 28)
            input_spike = spikeEncoder.SpikeEncode(x=input_data, timesteps=timestep)

            t1 = time.time()
            output_spike = snn(input_spike)
            t2 = time.time()

            snn.record_time(t2 - t1)
            # print(output_spike.sum(axis=0))
            if output_spike is not None:
                pred = np.argmax(output_spike.sum(axis=0))
            if pred == testlabels[i]:
                correct += 1
            # snn.paicore_status()
        print(f"{img_num}/{img_num} Acc:{round(correct / img_num, 4)}")

        snn.perf(img_num)
