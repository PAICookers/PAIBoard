import torch
import os
import numpy as np
from collections import OrderedDict
from frame import Frame
from hwSim import Simulator, GLOBAL_CORE_ID, STARID
from hwSim import ISSYNC, ISEND
from timeMeasure import time_calc_addText, get_original_function

'''-----------------------------------------------------------------------'''
'''                             RUN NETWORK                               '''
'''-----------------------------------------------------------------------'''

def loadNet(netPath):
    net = torch.load(netPath,map_location="cpu")
    net = net.cpu()
    net.eval()
    return net

def loadAuxNet(auxNetDir):
    preNetPath = os.path.join(auxNetDir, "preNet.pth")
    postNetPath = os.path.join(auxNetDir, "postNet.pth")
    return loadNet(preNetPath), loadNet(postNetPath)

@time_calc_addText("runPreNet     ")
def runPreNet(preNet, inputNames, *data):
    if len(preNet.inputs_nodes) > 0:
        preNet(*data)
        return preNet.nodes
    else:
        outData = dict()
        for i in range(len(inputNames)):
            outData[inputNames[i]] = data[i]
        return outData

def runPreNetWrap(preNet, inputNames, TimeMeasure, *data):
    if(TimeMeasure):
        return runPreNet(preNet, inputNames, *data)
    else:
        original_runPreNet = get_original_function(runPreNet)
        return original_runPreNet(preNet, inputNames, *data)

def runSimulator(simulator, dataFrames, txtFrame = False):
    if(txtFrame):
        dataFrames = getTxTData(dataFrames)
    else:
        dataFrames = getSimuData(dataFrames)

    dataFrames = [Frame.toInt(frame) for frame in dataFrames]
    frameList, helpInfo = framePartition(dataFrames)
    
    dataNum = 0
    for h in helpInfo:
        dataNum += h
    assert dataNum == 1, dataNum

    outputFrames = list()
    for frames, h in zip(frameList, helpInfo):
        simulator.setInputs(frames)
        simulator.begin()
        if h:
            outputFrames.append(
                [Frame.toString(frame) for frame in simulator.outputBuffer]
            )

    return outputFrames[0]  # maybe right
    # return outputFrames  


'''-----------------------------------------------------------------------'''
'''                        load & store frames                            '''
'''-----------------------------------------------------------------------'''

def getSimuData(dataFrames):
    frames = []
    for i in range(dataFrames.shape[0]):
        frames.append("{:064b}\n".format(dataFrames[i]))

    frames = [frame.strip() for frame in frames if not frame.startswith("0000")]
    dataFrames = [frame.strip() for frame in frames]
    return dataFrames

def getTxTData(dataPath):
    with open(dataPath,'r') as f:
        frames = f.readlines()
    frames = [frame.strip() for frame in frames if not frame.startswith("0000")]
    return [frame.strip() for frame in frames]

'''-----------------------------------------------------------------------'''
'''     parse config frames (support both online and offline cores)       '''
'''-----------------------------------------------------------------------'''

def parseConfig(configPath):
    
    '''---------------------------------------------------------------'''
    '''                           offline                             '''
    '''---------------------------------------------------------------'''
    def parseConfig1(frameGroup, coreId):
        return []

    def parseConfig2(frameGroup, coreId):
        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame,2) & ((1 << 30) - 1)
            load = (load << 30) + payLoad
        load = load >> 23
        config = [0] * 11
        dataLen = [2,4,1,1,13,1,15,15,1,4,10]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        for i in range(10, -1, -1):
            config[i] = load & dataMask[i]
            load >>= dataLen[i]
        return config
    
    def parseConfig3(frameGroup, coreId):
        intFrame0 = int(frameGroup[0],2)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        load = 0
        config = dict()
        dataLen = [8, 11, 5, 5, 5, 5, 5, 5, 2, 30, 1, 5, 1, 29, 29, 1, 1, 30, 1, 5, 30]
        signedData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        memBase = 0

        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 4 == 3:
                tmpConfig = [0] * 21
                for j in range(20, -1, -1):
                    tmpConfig[j] = load & dataMask[j]
                    if signedData[j] and (tmpConfig[j] & (1 << (dataLen[j] - 1)) != 0):
                        tmpConfig[j] -= 1 << dataLen[j]
                    load >>= dataLen[j]
                config[neuronId] = tmpConfig
                neuronId += 1
                memBase = 0
        return  config
    
    def parseConfig4_param(frameGroup, coreId):
        intFrame0 = int(frameGroup[0],2)
        # neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        neuronId = 0
        memBase = 0
        config = dict()
        dataLen = [8, 11, 5, 5, 5, 5, 5, 5, 2, 30, 1, 5, 1, 29, 29, 1, 1, 30, 1, 5, 30]
        signedData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        load = 0
        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 18 == 17:
                memBase = 0
                for j in range(5):
                    tmpConfig = [0] * 21
                    for k in range(len(dataLen) - 1, -1, -1):
                        tmpConfig[k] = load & dataMask[k]
                        if signedData[k] and (tmpConfig[k] & (1 << (dataLen[k] - 1)) != 0):
                            tmpConfig[k] -= 1 << dataLen[k]
                        load >>= dataLen[k]
                    config[neuronId] = tmpConfig
                    neuronId += 1
                    # load >>= 214

        return config
    
    def parseConfig4_weight(frameGroup, coreId, isSNN):
        intFrame0 = int(frameGroup[0],2)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        if not isSNN:
            neuronId *= 8
        frameNum = intFrame0 & ((1 << 19) - 1)
        assert starId == 0
        memBase = 0
        config = OrderedDict()
        load = 0
        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 18 == 17:
                memBase = 0
                if isSNN:
                    weight = np.zeros(1152)
                    for i in range(1152):
                        weight[i] = load & 1
                        load >>= 1
                    config[neuronId] = weight
                    neuronId += 1
                else:
                    for j in range(8):
                        weight = np.zeros(144)
                        for i in range(144):
                            weight[i] = load & 1
                            load >>= 1
                        config[neuronId] = weight
                        neuronId += 1
                load = 0
        return config

    '''---------------------------------------------------------------'''
    '''                            online                             '''
    '''---------------------------------------------------------------'''

    def parseConfigLUT(frameGroup, coreId):

        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame[-30:], 2)
            load = (load << 30) + payLoad
        LUT = np.zeros(60, dtype=int)
        mask = (1 << 8) - 1
        for i in range(59,-1,-1):
            LUT[i] = int(load & mask)
            if LUT[i] & (1 << 7):
                LUT[i] -= (1 << 8)
            load >>= 8
        return  LUT
    
    def parseConfig2ON(frameGroup, coreId):

        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame,2) & ((1 << 30) - 1)
            load = (load << 30) + payLoad
        load = load >> 30
        config = [0] * 18

        dataLen = [
             2,  2, 32, 8,  8,  8, 
            10, 10,  5, 5, 15, 15, 
            60,  1,  1, 1, 10, 16
        ]

        dataMask = [(1 << (i)) - 1 for i in dataLen]

        signedData = [
            0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]

        for i in range(len(config) - 1, -1, -1):
            config[i] = load & dataMask[i]
            load >>= dataLen[i]
            if i == 13:
                config[i] = load & dataMask[i]
                load >>= 1
            if signedData[i]:
                if config[i] & ((1 << (dataLen[i] - 1))):
                    config[i] -= (1 << dataLen[i])
        return config

    def parseConfig3ON(frameGroup, coreId, bitWidth):
        intFrame0 = Frame.toInt(frameGroup[0])
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        load = 0
        config = dict()

        if bitWidth == 1:
            dataLen = [
                15, 15, 7, 6, 6, 15,  3,  5,
                5,  5, 5, 5, 5, 11, 10, 10,
            ]
        else:
            dataLen = [ 
                32, 32, 32, 32, 32, 32,  3,  5,
                5,  5,  5,  5,  5, 11, 10, 10
            ]

        signedData = [
            1, 1, 1, 1, 1, 1, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        dataMask = [(1 << (i)) - 1 for i in dataLen]

        if bitWidth == 1:
            frameUnit = 2
            frameEnd = 1
            neuronUnit = 1
        else:
            frameUnit = 4
            frameEnd = 3
            neuronUnit = 2

        for i, frame in enumerate(frameGroup[1:]):
            frame = Frame.toInt(frame)
            payLoad = frame & ((1 << 64) - 1)
            load = (load<<64) + payLoad

            if i % frameUnit == frameEnd:
                tmpConfig = [0] * len(dataLen)
                for j in range(len(tmpConfig) - 1, -1, -1):
                    tmpConfig[j] = load & dataMask[j]
                    if signedData[j] and (tmpConfig[j] & (1 << (dataLen[j] - 1)) != 0):
                        tmpConfig[j] -= 1 << dataLen[j]
                    load >>= dataLen[j]
                config[neuronId] = tmpConfig
                neuronId += neuronUnit
        return config
    
    def parseConfig4ON(frameGroup, coreId, bitWidth):
        # intFrame0 = int(frameGroup[0],2)
        intFrame0 = Frame.toInt(frameGroup[0])
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)

        frameNum = intFrame0 & ((1 << 19) - 1)
        config = dict()
        load = 0
        weightVec = list()
        OneComplete = 16 * bitWidth
        weightMask = (1 << bitWidth) - 1
        for i, frame in enumerate(frameGroup[1:]):
            # frame = int(frame,2)
            frame = Frame.toInt(frame)
            payLoad = frame & ((1 << 64) - 1)
            load = (load << 64) + payLoad
            if i % 16 == 15:
                vecNum = 1024 // bitWidth
                for j in range(vecNum):
                    shiftLen = (vecNum - 1 -j) * bitWidth
                    oneWeight = (load >> shiftLen) & weightMask
                    if bitWidth > 1 and oneWeight & (1<<(bitWidth - 1)):
                        oneWeight -= 1 << bitWidth
                    weightVec.append(oneWeight)
                load = 0
            if i % OneComplete == OneComplete - 1:
                weight = np.array(weightVec)
                config[neuronId] = weight
                weightVec.clear()
                neuronId += bitWidth
        return config

    '''--------------------------------------------------------------'''
    '''                         parse frames                         '''
    '''--------------------------------------------------------------'''

    with open(configPath,'r') as f:
        frames = f.readlines()
    frameNum = len(frames)
    configs =  dict()
    i = 0
    
    
    while i < frameNum:
        frame = frames[i].strip()
        # intFrame = int(frame,2)
        intFrame = Frame.toInt(frame)
        frameHead = intFrame >> 60
        coreId = GLOBAL_CORE_ID(intFrame)
        starId = STARID(intFrame)
        assert starId == 0, f"{i}: {frame}"
        # assert coreId & 31 <= 15, coreId
        # assert (coreId >> 5) <= 15, coreId
        # assert coreId & int("1110011100",2) != int("1110011100",2), coreId
        # assert frameHeadM <= frameHead, f"{frameHeadM}: {frames[i]}"
        # frameHeadM = frameHead
        if frameHead == 0:
            k = 0
            while i + k < frameNum:
                tmpIntFrame = Frame.toInt(frames[i+k])
                tmpFrameHead = tmpIntFrame >> 60
                if tmpFrameHead == 0 and GLOBAL_CORE_ID(tmpIntFrame) == coreId:
                    k += 1
                else:
                    break
            if k == 3: # offline
                LUT = parseConfig1(frames[i:i+k], coreId)
            else:
                LUT = parseConfigLUT(frames[i:i+k], coreId)
            if coreId in configs:
                configs[coreId]['LUT'] = LUT
            else:
                configs[coreId] = {'LUT':LUT}
            i += k

        elif frameHead == 1:
            assert coreId in configs, f"{coreId}\n {sorted(configs.keys())}"
            if len(configs[coreId]['LUT']) == 0:
                end = i + 3
            else:
                end = i + 8
            frameGroup = [frames[j].strip() for j in range(i, end)]
            if len(configs[coreId]['LUT']) == 0:
                config = parseConfig2(frameGroup, coreId)
            else:
                config = parseConfig2ON(frameGroup, coreId)
            configs[coreId]['core'] = config
            i = end

        elif frameHead == 2:
            num = intFrame & ((1 << 19) - 1)
            end = i + num + 1
            neuronNum = configs[coreId]['core'][4]
            frameGroup = [frames[j].strip() for j in range(i, end)]
            if 'neuron' not in configs[coreId]:
                configs[coreId]['neuron'] = dict()

            if len(configs[coreId]['LUT']) == 0:
                config = parseConfig3(frameGroup,coreId)
                isSNN = (configs[coreId]['core'][2] == 0)
                neuronUnit = (1 << configs[coreId]['core'][0]) * (1 << configs[coreId]['core'][1])
                if not isSNN:
                    for neuronId, neuronConfig in config.items():
                        if neuronId * neuronUnit >= neuronNum:
                            continue
                        for j in range(neuronUnit):
                            if neuronId * neuronUnit + j not in configs[coreId]['neuron']:
                                configs[coreId]['neuron'][neuronId * neuronUnit + j] = {
                                    'parameter':None, 
                                    'weight':None
                                }
                            configs[coreId]['neuron'][neuronId * neuronUnit + j]['parameter'] = neuronConfig
                else:
                    for neuronId, neuronConfig in config.items():
                        if neuronId  not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][neuronId] = {'parameter':None, 'weight':None}
                        configs[coreId]['neuron'][neuronId]['parameter'] = neuronConfig
            else:
                bitWidth = 1 << configs[coreId]['core'][0]
                config = parseConfig3ON(frameGroup, coreId, bitWidth)
                for neuronId, neuronConfig in config.items():
                    if neuronId % bitWidth != 0:
                        continue
                    for j in range(bitWidth):
                        newId = neuronId + j
                        if newId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][newId] = {
                                'parameter': None,
                                'weight': None
                            }
                        configs[coreId]['neuron'][newId]['parameter'] = neuronConfig

            

            i = end

        elif frameHead == 3:
            num = intFrame & ((1 << 19) - 1)
            neuronId = (intFrame >> 20) & ((1 << 10) - 1)
            end = i + num + 1
            frameGroup = [frames[j].strip() for j in range(i, end)]
            if 'neuron' not in configs[coreId]:
                configs[coreId]['neuron'] = dict()
            if len(configs[coreId]['LUT']) == 0:
                isSNN = (configs[coreId]['core'][2] == 0)
                neuronUnit = configs[coreId]['core'][0] * (1 << configs[coreId]['core'][1])
                neuronNum = configs[coreId]['core'][4]
                if isSNN:
                    config = parseConfig4_weight(frameGroup, coreId, isSNN)
                    for neuronId, neuronConfig in config.items():
                        if neuronId >= neuronNum:
                            continue
                        if neuronId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][neuronId] = {'paramter':None, 'weight': None}
                        configs[coreId]['neuron'][neuronId]['weight'] = neuronConfig
                else:
                    if neuronId == 0:
                        config = parseConfig4_weight(frameGroup, coreId, isSNN)
                        for neuronId, neuronConfig in config.items():
                            if neuronId >= neuronNum:
                                continue
                            if neuronId not in configs[coreId]['neuron']:
                                configs[coreId]['neuron'][neuronId] = {'paramter':None, 'weight': None}
                            configs[coreId]['neuron'][neuronId]['weight'] = neuronConfig
                    else:
                        # assert False
                        config = parseConfig4_param(frameGroup, coreId)
                        neuronUnit = 1<< configs[coreId]['core'][0]     #bitWidth
                        neuronUnit *= (1 << configs[coreId]['core'][1]) #LCN
                        neuronBase = int(512 * neuronUnit)
                        for neuronId, neuronConfig in config.items():
                            base = neuronBase + neuronId * neuronUnit
                            for i in range(neuronUnit):
                                if base >= neuronNum:
                                    continue
                                if base not in configs[coreId]['neuron']:
                                    configs[coreId]['neuron'][base] = {'paramter':None, 'weight': None}
                                configs[coreId]['neuron'][base]['parameter'] = neuronConfig
                                base += 1
            else:
                bitWidth = 1 << configs[coreId]['core'][0]
                config = parseConfig4ON(frameGroup, coreId, bitWidth)
                for neuronId, neuronConfig in config.items():
                    for i in range(bitWidth):
                        newId = neuronId + i
                        # newConfig = (neuronConfig >> (bitWidth - i - 1)) & 1
                        if newId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][newId] = {
                                'parameter': None,
                                'weight': None
                            }
                        configs[coreId]['neuron'][newId]['weight'] = neuronConfig
            i = end
        
        else:
            assert False, frameHead
    
    return configs


'''-----------------------------------------------------------------------'''
'''     partition frames in multiple groups:                              '''
'''         if a group has True helpInfo, it has data frames              '''
'''         else it is just end frames for online cores                   '''
'''-----------------------------------------------------------------------'''

def framePartition(frames):
    beforeSync = True
    onSync = False
    afterSync = False
    preFrames = list()
    postFrames = list()
    frameList = list()
    helpInfo = list()
    for i, frame in enumerate(frames):
        if ISSYNC(frame):
            beforeSync = False
            onSync = True
        else:
            if onSync:
                onSync = False
                afterSync = True
                frameList.append(preFrames)
                helpInfo.append(1)
                preFrames = list()
        if afterSync:
            if ISEND(frame):
                postFrames.append(frame)
            else:
                frameList.append(postFrames)
                helpInfo.append(0)
                postFrames = list()
                beforeSync = True
                onSync = False
                afterSync = False
        else:
            preFrames.append(frame)
    if len(preFrames) > 0:
        frameList.append(preFrames)
        helpInfo.append(1)
    if len(postFrames) > 0:
        frameList.append(postFrames)
        helpInfo.append(0)
    return frameList, helpInfo

def setOnChipNetwork(configPath):
    configs = parseConfig(configPath)
    simulator = Simulator()
    simulator.setConfig(configs)
    return simulator

