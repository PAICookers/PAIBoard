import __init__
import os
import numpy as np
import time
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

import torch
from spike_tensor import SpikeTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def genInputData(data, timeStep, inputMode):
    if inputMode == 'snn':
        data = SpikeTensor(data, timeStep, 1)
    else:
        data = data * 255
        data.scale = 1
    return [[data]]

if __name__ == "__main__":

    dataset_root = os.path.join(os.path.expanduser('~'), "work/99_datasets/")
    dataset_name = 'CIFAR10'

    dataset_fn=datasets.CIFAR10

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                      std=[0.2470, 0.2435, 0.2616])
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                # normalize,
                ])  

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_set = dataset_fn(dataset_root, download=False, train=False, transform=transform_test)
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)


    timestep = 64
    layer_num = 0
    baseDir = "./result/CIFAR10_SNN_ENCODE"

    # snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num, backend="PAIFLOW")
    snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num, backend="PAIFLOW")
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num, backend="PAIFLOW")
    snn.config(oFrmNum=200)

    correct = 0
    cnt = 0
    num = 1000
    mode = 'snn'

    for data,target in test_loader:
        cnt += 1
        t1 = time.time()
        data_rep = data.repeat(timestep,1,1,1)
        inputdata = genInputData(data_rep, timestep, mode)

        outputSpike = snn(inputdata[0], TimeMeasure=False)

        pred = np.argmax(outputSpike.sum(axis=0))
        t2 = time.time()
        snn.record_time(t2 - t1)
        # print("      target : ",classes[target])
        # print("        pred : ",classes[pred])
        # print("")
        if(target == pred):
            correct +=1

        
        if cnt == num:
            break
        print("       "+ str(cnt) + "/" + str(num) + " Acc:" + str(round(correct/(cnt),4)) + '\r',end='')
    print("----------------------------------")
    print('                     \r' + str(num) + "/" + str(num) + " Acc:" + str(round(correct/num,4)))
    print("----------------------------------\n")

    snn.perf(num)