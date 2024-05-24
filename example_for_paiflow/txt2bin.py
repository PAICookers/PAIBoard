import os
import numpy as np

def txtframe2bin(configPath):
    config_frames = np.loadtxt(configPath, str)
    config_num = config_frames.size
    config_buffer = np.zeros((config_num,), dtype=np.uint64)
    for i in range(0, config_num):
        config_buffer[i] = int(config_frames[i], 2)
    config_frames = config_buffer
    configPath = configPath[:-4] +  ".bin"
    print(configPath)
    config_frames.tofile(configPath)

baseDir = "./result/CIFAR10_SNN_ENCODE/output/config_cores_all.txt"
txtframe2bin(baseDir)
