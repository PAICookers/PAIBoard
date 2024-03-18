import __init__

import numpy as np
import time

import example.datasets.load_mnist as load_mnist
from example.Encoder.encoder import PoissonEncoder

from paiboard.pcie.paiboard_pcie import PAIBoard_PCIe
from paiboard.simulator.paiboard_sim import PAIBoard_SIM

if __name__ == "__main__":
    timestep = 8
    layer_num = 0
    baseDir = "./result/03_mnist_1layer/"
    snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    snn.config(oFrmNum=20)

    dataset_root = "./example/datasets/MNIST/raw"
    testdata, testlabels = load_mnist.CreatData(dataset_root)
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    print()
    pbsim_data = np.load(baseDir + "output_save.npy")
    img_num = 100
    correct = 0
    for i in range(img_num):
        input_data = testdata[i].reshape(1, 28, 28)

        input_spike = PoissonEncoder(x=input_data, timesteps=timestep)
        output_spike = snn(input_spike, TimeMeasure=True)

        if output_spike is not None:
            if i < 100:
                if not (output_spike == pbsim_data[i]).all():
                    # todo find out why different from paibox simulation
                    print(f"DIFF: {i}")
                    print(output_spike)
                    print(pbsim_data[i])
                    pass
            pred = np.argmax(output_spike.sum(axis=0))
        else:
            continue
        if pred == testlabels[i]:
            correct += 1

    print(f"{img_num}/{img_num} Acc:{round(correct / img_num, 4)}")
