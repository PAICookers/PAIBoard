import torch
from spike_tensor import SpikeTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PAIBoard import PAIBoard
from dma_utils import *
from timeMeasure import *
from global_hw_params import *

def genInputData(data, timeStep, inputMode):
    if inputMode == 'snn':
        data = SpikeTensor(data, timeStep, 1)
    else:
        data = data * 255
        data.scale = 1
    return [[data]]

if __name__ == "__main__":

    timeStep = 64
    coreType = 'offline'
    mode = 'snn'
    appName = 'CIFAR10_SNN_ENCODE'

    import os
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

    snn = PAIBoard(appName, timeStep, coreType, sim=True, Verbose=False)
    snn.config(globalSignalDelay = 92, oFrmNum = 100, oen=oen, channel_mask=channel_mask)

    correct = 0
    cnt = 0
    num = 100
    print("")
    indata = []
    intarget = []
    for data,target in test_loader:
        indata.append(data)
        intarget.append(target)
        cnt += 1
        if cnt == num:
            break

    for i in range(num):
        data_rep = indata[i].repeat(timeStep,1,1,1)
        inputdata = genInputData(data_rep, timeStep, mode)

        import time
        t1 = time.time()

        pred = snn(inputdata[0], REG_RECV=False, TimeMeasure=True)

        t2 = time.time()
        
        # val = read_bypass(REGFILE_BASE + TLAST_CNT)
        # time_dict['FULL INFERENCE'] = time_dict['FULL INFERENCE'] + (t2-t1)*1000*1000
        # time_dict['CORE INFERENCE'] = time_dict['CORE INFERENCE'] + int(val/1)

        # print("\r",end="") # 不加这句测出来的时间不准
        # if(i == num-1):
        #     print("----------------------------------")
        #     for key in time_dict:
        #         print(key + " TIME : {:.1f} us".format(time_dict[key]/num))
        #     print("----------------------------------")

        print("      target : ",classes[intarget[i]])
        print("        pred : ",classes[pred])
        print("")

        if(intarget[i] == pred):
            correct +=1

        cnt += 1
        if cnt == num:
            break
        print(str(i+1) + "/" + str(num) + " Acc:" + str(round(correct/(i+1),4)) + '\r',end='')
    print("----------------------------------")
    print(appName)
    print('                     \r' + str(num) + "/" + str(num) + " Acc:" + str(round(correct/num,4)))
    print("----------------------------------\n")