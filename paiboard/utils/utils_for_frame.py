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