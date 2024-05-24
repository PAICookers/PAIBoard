import numpy as np
from snn_utils import *
import os
from runtime import *
from dma_utils import *

class PAIBoard(object):
    def __init__(self, appName: str , timeStep: int, coreType: str, sim = False, Verbose = False):
        self.appName = appName

        current_path = os.path.abspath(__file__)
        self.father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + "..")

        self.baseDir  = self.father_path + '/result/' + self.appName + '/output/'
        self.timeStep = timeStep
        self.coreType = coreType
        self.learn_mode    = 0 if(coreType == 'offline') else 1

        auxNetDir = os.path.join(self.baseDir, "auxNet")
        self.preNet, self.postNet = loadAuxNet(auxNetDir)

        self.frameFormats, self.frameNums, self.inputNames = loadInputFormats(self.baseDir)
        self.initFrames,self.syncFrames = GEN_INIT_SYNC(self.frameFormats, self.frameNums)
        vectorized_binary_to_uint64 = np.vectorize(binary_to_uint64)
        self.frameFormats = vectorized_binary_to_uint64(self.frameFormats[self.frameNums[0]:-1])
        
        self.outDict, self.shapeDict, self.scaleDict, self.mapper = loadOutputFormats(self.baseDir)
    
        self.sim = sim
        self.Verbose = Verbose

    def config(self,globalSignalDelay = 92, oFrmNum = 10000, oen = 0b1110, channel_mask = 0b1000):

        if(self.sim):
            configPath = os.path.join(self.baseDir, "frames/config.txt")
            self.simulator = setOnChipNetwork(configPath)
        else:
            print("")
            if serialConfig(globalSignalDelay = globalSignalDelay) :
                print("[Error] : Uart can not send, Open and Reset PAICORE.")
                exit()

            self.oFrmNum = oFrmNum
            dma_init(self.oFrmNum, oen, channel_mask)

            # resetPath = "/home/cjailab/work/PAIBoard_PCIe/APP/result/RESET_FRAME.bin"
            # resetFrames = np.fromfile(resetPath, dtype='<u8')
            # SendFrame(resetFrames)

            # initPath = "/home/cjailab/work/PAIBoard_PCIe/APP/result/INIT_FRAME.bin"
            # self.initFrames = np.fromfile(initPath, dtype='<u8')

            configPath = os.path.join(self.baseDir, "frames/config.bin")
            configFrames = np.fromfile(configPath, dtype='<u8')
            print("----------------------------------")
            print("----------PAICORE CONFIG----------")
            SendFrameWrap(configFrames)
            print("----------------------------------")

    def Init(self):
        Init(self.initFrames)

    def __call__(self, x , raw_out = False, REG_RECV = False, TimeMeasure = False):

        x = runPreNetWrap(self.preNet, self.inputNames, TimeMeasure, *x)
        dataFrames = Tensor2FrameWrap(x, self.frameFormats,self.inputNames, TimeMeasure)

        if(self.sim):
            inputFrames = np.concatenate((self.initFrames,dataFrames,self.syncFrames))
            outputFrames = runSimulator(self.simulator, inputFrames)
        else:
            inputFrames = np.concatenate((dataFrames,self.syncFrames))
            outputFrames_np = WorkOnce(self.initFrames, inputFrames, self.oFrmNum, REG_RECV=REG_RECV, TimeMeasure = TimeMeasure)
            outputFrames = []
            
            for i in range(len(outputFrames_np)):
                out_str = bin(outputFrames_np[i])
                outputFrames.append(out_str[2:])

        outData, outData_spk = Frame2TensorWrap(outputFrames, self.outDict, self.shapeDict, self.scaleDict, self.mapper, self.timeStep, self.coreType, TimeMeasure)
        if(raw_out):
            return outData
        else:
            pred = np.argmax(outData)
            return pred
