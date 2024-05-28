import os
import numpy as np

def frame_np2txt(frameBuffer, txt_path, frameSplit = True):
    with open(txt_path, 'w') as f:
        for i in range(frameBuffer.shape[0]):
            if frameSplit:
                frameStr = "{:064b}".format(frameBuffer[i])
                dataLen = [4,10,10,10,3,11,8,8]
                for j in range(len(dataLen)):
                    f.write(frameStr[sum(dataLen[:j]):sum(dataLen[:j+1])] + " ")
                f.write("\n")
            else:
                f.write("{:064b}\n".format(frameBuffer[i]))


def txtframe2bin(txt_path):
    config_frames = np.loadtxt(txt_path, str)
    config_num = config_frames.size
    config_buffer = np.zeros((config_num,), dtype=np.uint64)
    for i in range(0, config_num):
        config_buffer[i] = int(config_frames[i], 2)
    config_frames = config_buffer
    configPath = txt_path[:-4] +  ".bin"
    config_frames.tofile(configPath)

if __name__ == "__main__":
    baseDir = "./result/CIFAR10_SNN_ENCODE/output/config_cores_all.txt"
    txtframe2bin(baseDir)
