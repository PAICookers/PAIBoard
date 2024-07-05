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
    batch = 32
    layer_num = 1
    baseDir = "./result/base_example/fcnet_2layer/"
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num, batch_size=batch)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num, batch_size=batch)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num, batch_size=batch)

    snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
    snn.config(oFrmNum=timestep * batch * 10)

    dataset_root = os.path.join(os.path.expanduser("~"), "work/99_datasets/MNIST/raw")
    testdata, testlabels = load_mnist.CreatData(dataset_root)
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    print()
    img_num = 100
    iter_num = int((img_num - 1) / batch) + 1 if img_num > 0 else 0
    correct = 0
    for i in range(iter_num):
        # prepare input data by batch
        t1 = time.time()

        img_index = i * batch
        if img_index + batch > img_num:
            batch = img_num - img_index
        input_data = testdata[img_index : img_index + batch].reshape(batch, 1, 28, 28)
        input_spike = PoissonEncoder(x=input_data, timesteps=timestep)
        input_spike = input_spike.transpose(1, 0, 2, 3, 4).reshape(
            batch * timestep, 1, 28, 28
        )

        output_spike = snn(input_spike)

        output_spike = output_spike.reshape(batch, timestep, -1)
        pred = np.argmax(output_spike.sum(axis=1), axis=1)

        t2 = time.time()

        snn.record_time(t2 - t1)

        correct += np.sum(
            pred == testlabels[img_index : min(img_index + batch, img_num)]
        )

    print(f"{img_num}/{img_num} Acc:{round(correct / img_num, 4)}")

    snn.perf(img_num)
