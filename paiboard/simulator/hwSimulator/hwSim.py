from copy import deepcopy
import time
from tqdm import tqdm
import numpy as np
from .hwConfig import Hardware
from .frame import Frame, FrameKind, MASK

PRE_COUNTER_MAX = 31
POST_COUNTER_MAX = 31
LUTNUM = 60
LUTBIAS = 29

def multiCast(coreBase, starId, coreBit, mapping, chipCoreset=None):
    cores = set()
    cores.add(coreBase)
    for i in range(coreBit):
        if (starId >> i) & 1:
            tmpCores = deepcopy(cores)
            star = 1 << i
            for core in tmpCores:
                if core ^ star not in list(chipCoreset.keys()):
                    continue
                cores.add(core ^ star)
    if mapping is not None:
        newCores = set()
        for core in cores:
            newCores.add(mapping[core])
        return newCores
    else:
        return cores

def CHIPID(intFrame):
    return Frame.getChipId(intFrame)

def COREID(intFrame):
    return Frame.getCoreId(intFrame)

def STARID(intFrame):
    return Frame.getStarId(intFrame)

def AXONID(intFrame):
    return Frame.getAxonId(intFrame)

def SLOTID(intFrame):
    return Frame.getSlotId(intFrame)
    
def DATAID(intFrame):
    return Frame.getData(intFrame)

def PAYLOAD(intFrame):
    return Frame.getPayload(intFrame)

def GLOBAL_CORE_ID(intFrame):
    return Frame.getGlobalId(intFrame)

def ISDATA(intFrame):
    return Frame.isKind(intFrame, FrameKind.DATA)

def ISSYNC(intFrame):
    return Frame.isKind(intFrame,FrameKind.SYNC)

def ISCLEAR(intFrame):
    return Frame.isKind(intFrame, FrameKind.CLEAR)

def ISINIT(intFrame):
    return Frame.isKind(intFrame, FrameKind.INIT)

def ISSTART(intFrame):
    return Frame.isKind(intFrame, FrameKind.START)

def ISEND(intFrame):
    return Frame.isKind(intFrame, FrameKind.END)

def ISLATERAL(intFrame):
    return Frame.isKind(intFrame, FrameKind.LATERAL)

class Neuron:
    def __init__(self, chipId, coreId, neuronId):
        self.chipId = chipId
        self.coreId = coreId
        self.neuronId = neuronId
        return 
    def setInit(self):
        return

class OfflineNeuron(Neuron):
    def __init__(self, chipId, coreId, neuronId, paras, weights, bitWidth, LCN):
        super().__init__(chipId, coreId, neuronId)
        para = paras[0]
        self.tickReltive = para[0]
        destAxon = para[1]
        destCore = (para[2] << Hardware.COREY) + para[3]
        destStar = (para[4] << Hardware.COREY) + para[5]
        destChip = (para[6] << Hardware.CHIPY) + para[7]
        cores = {destCore}
        for i in range(10):
            if (destStar >> i) & 1:
                star = 1 << i
                tmp = deepcopy(cores)
                for c in tmp:
                    cores.add(c ^ star)
        
        self.spikeFormats = list()
        for core in cores:
            spikeFormat = \
                (8 << 60) + (destChip << 50) + (core << 40) + \
                + (destAxon << 16)
            assert spikeFormat < (3 << 62), f"{destChip},{destCore},{destStar},{destAxon}"
            self.spikeFormats.append(spikeFormat)
        self.resetMode = para[8]
        self.resetV = para[9]
        self.leakPost = para[10]
        self.threMaskCtrl = para[11]
        self.threNegMode = para[12]
        self.thresholdNeg = para[13]
        self.thresholdPos = para[14]
        self.leakReFlag = para[15]
        self.leakDetStoch = para[16]
        self.leakV = para[17]
        self.weightDetStoch = para[18]
        self.bitTrunc = para[19]
        self.vjtPre = int(para[20])
        self.weights = [[] for i in range(LCN)]
        self.weightPos = [[] for i in range(LCN)]
        end = self.bitTrunc
        beg = max(0, self.bitTrunc - 8)
        self.shiftR = beg
        self.truncEnd = 1 << end
        self.truncMask = (1 << (end - beg)) - 1
        self.shiftL = 8 - (end - beg)
        for i in range(LCN):
            for j, w in enumerate(weights[i]):
                if w != 0:
                    self.weights[i].append(w)
                    self.weightPos[i].append(j)
                
    def truncRelu(self):
        if self.vjtPre <= 0 or self.vjtPre < self.thresholdPos:
            return 0
        if self.vjtPre >= self.truncEnd:
            return (1 << 8) - 1
        return min((int(self.vjtPre) >> self.shiftR), self.truncMask) << self.shiftL
    
    def compute(
        self, timeStep, buffer, LCN, SNN_EN, spikeWidth, slotBase, maxPool
    ):

        if not maxPool:
            output = 0
        else:
            output = -(1 << 20)
        for i in range(LCN):
            for w, pos in zip(self.weights[i], self.weightPos[i]):
                if not maxPool:
                    output += w * buffer[i,pos]
                else:
                    output = max(output, buffer[i,pos])
        self.vjtPre += output
        if self.leakPost == 0:
            self.vjtPre += self.leakV
        outputSpike = 0
        if spikeWidth == 0:
            if self.vjtPre >= self.thresholdPos:
                if self.resetMode == 0:
                    self.vjtPre = self.resetV
                elif self.resetMode == 1:
                    self.vjtPre -= self.thresholdPos
                outputSpike = 1
            elif self.vjtPre <= self.thresholdNeg:
                if self.threNegMode == 0:
                    self.vjtPre = self.resetV
                elif self.threNegMode == 1:
                    self.vjtPre = self.thresholdNeg
            if self.leakPost == 1:
                self.vjtPre += self.leakV
        else:
            outputSpike = self.truncRelu()
        if SNN_EN == 0:
            self.vjtPre = 0
        spikes = list()
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", True)
        if outputSpike != 0:
            for spikeFormat in self.spikeFormats:
                slotId = (self.tickReltive + slotBase) % hardwareSlotNum
                spike = (slotId << 8) + spikeFormat + outputSpike
                spikes.append(spike)
        return spikes

    def setInit(self):
        self.vjtPre = 0
    
    def dumpWeight(self, LCN_id, axonId):
        raise NotImplementedError()

    def debugNeuronParas(self, paras):
        para = paras[0]
        assert self.tickReltive == para[0]
        destAxon = para[1]
        destCore = (para[2] << Hardware.COREY) + para[3]
        destStar = (para[4] << Hardware.COREY) + para[5]
        destChip = (para[6] << Hardware.CHIPY) + para[7]
        cores = {destCore}
        for i in range(10):
            if (destStar >> i) & 1:
                star = 1 << i
                tmp = deepcopy(cores)
                for c in tmp:
                    cores.add(c ^ star)
        
        spikeFormats = list()
        for core in cores:
            spikeFormat = \
                (8 << 60) + (destChip << 50) + (core << 40) + \
                + (destAxon << 16)
            assert spikeFormat < (3 << 62), f"{destChip},{destCore},{destStar},{destAxon}"
            spikeFormats.append(spikeFormat)
        for spike1, spike2 in zip(spikeFormats, self.spikeFormats):
            assert spike1 == spike2, f"{spikeFormats}\n{self.spikeFormats}\n"

        assert self.resetMode == para[8], f"{self.resetMode} <--> {para[8]}"
        assert self.resetV == para[9], f"{self.resetV} <--> {para[9]}"
        assert self.leakPost == para[10], f"{self.leakPost} <--> {para[10]}"
        assert self.threMaskCtrl == para[11], f"{self.threMaskCtrl} <--> {para[11]}"
        assert self.threNegMode == para[12], f"{self.threNegMode} <--> {para[12]}"
        assert self.thresholdNeg == para[13], f"{self.thresholdNeg} <--> {para[13]}"
        assert self.thresholdPos == para[14], f"{self.thresholdPos} <--> {para[14]}"
        assert self.leakReFlag == para[15], f"{self.leakReFlag} <--> {para[15]}"
        assert self.leakDetStoch == para[16], f"{self.leakDetStoch} <--> {para[16]}"
        assert self.leakV == para[17], f"{self.leakV } <--> { para[17]}"
        assert self.weightDetStoch == para[18], f"{self.weightDetStoch} <--> {para[18]}"
        assert self.bitTrunc == para[19], f"{self.bitTrunc} <--> {para[19]}"
        assert self.vjtPre == int(para[20]), f"{self.vjtPre} <--> {int(para[20])}"

class OnlineNeuron(Neuron):
    def __init__(
        self, chipId, coreId, neuronId, paras, weights, bitWidth, LCN,
    ):
        super().__init__(chipId, coreId, neuronId)

        para = paras[0]
        self.leakV = para[0]
        self.thresholdPos = para[1]
        self.floorV = para[2]
        self.resetV = para[3]
        self.initV = para[4]
        self.vjtPre = para[5]
        self.tickRelative = para[6]

        destChip = (para[7] << Hardware.CHIPY) | para[8]
        destCore = (para[9] << Hardware.COREY) | para[10]
        destStar = (para[11] << Hardware.COREY) | para[12]
        destAxon = para[13]

        
        destCores = multiCast(destCore, destStar, Hardware.COREBIT, None)

        self.dataFrameFormats = list()
        for coreId in destCores:
            self.dataFrameFormats.append(
                Frame.genDTSim(
                    destChip, coreId, destStar, destAxon
                )
            )
        

        self.plascityBegs = list()
        self.plascityEnds = list()
        neuronNum = len(paras)
        for i in range(0, neuronNum, bitWidth):
            self.plascityBegs.append(paras[i][14])
            self.plascityEnds.append(paras[i][15])

        self.weights = np.array(weights).astype(int)
        assert self.weights is not None, f"{self.coreId}, {self.neuronId}, {weights}"

        self.postTrace = 0
        self.postFlag = False

        self.spikes = 0

        return
    
    def printNeuron(self):
        print(f"------------------[{self.chipId},{self.coreId},{self.neuronId}]---------------------")
        # print(f"weight = {self.weights[:,:16]}")
        # print(f"{np.argmax(self.weights)}")
        print(f"leak = {self.leakV}")
        print(f"threshold = {self.thresholdPos}")
        print(f"memFloor = {self.floorV}")
        print("------------------------------------------------------------------------------------------")
    
    def compute(
        self, timeStep, buffer, LCN, slotBase, lateralVal, onlineMode,
        preTrace, LUT, lowerWeight, upperWeight
    ):
        
        update = np.sum(self.weights * buffer) + self.leakV
        self.vjtPre += update - lateralVal
        self.vjtPre = max(self.vjtPre, self.floorV)
        isSpike = self.vjtPre >= self.thresholdPos
        outputs = list()
        if isSpike:
            self.spikes += 1
            self.vjtPre = self.resetV
            # if onlineMode:
            #     self.vjtPre = self.resetV
            # else:
            #     self.vjtPre -= self.thresholdPos
            slotNum = Hardware.getAttr("SLOTNUM", False)
            slotId = (slotBase + self.tickRelative) % slotNum
            outputs = [
                Frame.genDF(t, slotId, 1) for t in self.dataFrameFormats
            ]
            if onlineMode:
                self.postFlag = True
                self.postTrace = 0
        elif onlineMode:
            self.postTrace = min(self.postTrace + 1, POST_COUNTER_MAX)
        if onlineMode:
            self.updateWeight(preTrace, LUT, lowerWeight, upperWeight, isSpike)
    
        return outputs

    def updateWeight(self,preTrace, LUT, lowerWeight, upperWeight, postSpike):
        if postSpike:
            weightDelta = LUT[
                (preTrace + LUTBIAS).astype(int).clip(0, 59)
            ].reshape(self.weights.shape).astype(int)
        else:
            weightDelta = (LUT[
                (preTrace - self.postTrace + LUTBIAS).astype(int).clip(0, 59)
            ] * (preTrace == 0)).reshape(self.weights.shape).astype(int)

        LCN = len(self.weights)
        for i in range(LCN):
            beg = self.plascityBegs[i]
            end = self.plascityEnds[i] + 1
            self.weights[i, beg : end] += weightDelta[i, beg : end]
        self.weights = self.weights.clip(lowerWeight, upperWeight)
        return

    def doWeightDecay(self, preFlag, weightDecay, lowerWeight, upperWeight):
        if (not self.postFlag):
            return
        LCN = len(self.weights)
        tmpWeightDecay = (weightDecay * (preFlag == 0)).reshape(self.weights.shape)
        for i in range(LCN):
            beg = self.plascityBegs[i]
            end = self.plascityEnds[i] + 1
            self.weights[i, beg: end] -= tmpWeightDecay[i, beg:end]
        self.weights = self.weights.clip(lowerWeight, upperWeight)
    
    def setInit(self):
        self.vjtPre = 0

    def setBeg(self):
        self.vjtPre = self.initV
        self.postFlag = False
        self.postTrace = 0

    def setEnd(self, preFlag, weightDecay, lowerWeight, upperWeight):
        self.doWeightDecay(preFlag, weightDecay, lowerWeight, upperWeight)
    
    def dumpWeight(self, LCN_id, axonId):
        return self.weights[LCN_id, axonId]

class Core:
    def __init__(self, chipId, coreId, *coreConfig):
        #coreConfig: 
        #   timeNum, axonNum, tickWaitStart, tickWaitEnd
        #   
        self.chipId = chipId
        self.coreId = coreId

        timeNum, axonNum, tickWaitStart, tickWaitEnd = coreConfig

        self.tickWaitStart = tickWaitStart
        self.tickWaitEnd = tickWaitEnd

        self.inputBuffer = np.zeros([timeNum,axonNum],dtype=int)
        self.neurons = list()
        self.timeStep = 0
        self.outputBuffer = list()
        

        return

    def checkActive(self, timeId):
        return (self.tickWaitStart > 0) and \
            (timeId >= self.tickWaitStart and (self.timeStep < self.tickWaitEnd or self.tickWaitEnd == 0))
    
    def compute(self, timeId):
        assert False, f"Your Core class must override func compute(self, timeId).\n"
    
    def initNeurons(self, neuronConfigs):
        assert False, f"Your Core class must override func initNeurons(self, neuronConfigs).\n"

    def updateState(self, spike):
        assert False, f"Your Core class must override func updateState(self, spike).\n"

    def setInit(self):
        self.inputBuffer[:,:] = 0
        self.outputBuffer.clear()
        self.timeStep = 0
        for neuron in self.neurons:
            neuron.setInit()

    def advanceTime(self, timeId):
        if self.checkActive(timeId):
            self.timeStep+=1

    def dumpWeight(self, completeNeuronId, LCN_id, axonId):
        return self.neurons[completeNeuronId].dumpWeight(LCN_id, axonId)

class OnlineCore(Core):

    def __init__(self, chipId, coreId, configs):
        timeStepNum = 8
        axonNum = 1024
        coreConfig = [
            timeStepNum, 
            axonNum, 
            configs['core'][10], 
            configs['core'][11]
        ]
        super().__init__(chipId, coreId, *coreConfig)
        self.weightWidth  = 1 << configs['core'][0]
        self.LCN          = 1 << configs['core'][1]
        self.lateralInhi  = configs['core'][2]
        self.weightDecay  = configs['core'][3]
        self.upperWeight  = configs['core'][4]
        self.lowerWeight  = configs['core'][5]
        self.neuronStart  = configs['core'][6]
        self.neuronEnd    = configs['core'][7]
        self.inhiCoreStar = (configs['core'][8] << 5) + configs['core'][9]
        self.leakageOrder = configs['core'][14]
        self.onlineMode   = configs['core'][15]
        # self.onlineMode   = 1
        
        self.testChipAddr = configs['core'][16]
        self.LUT          = configs['LUT']

        self.preTraces  = np.zeros(axonNum * self.LCN, dtype=int)
        self.preFlag = np.zeros(axonNum * self.LCN, dtype=bool)
        self.lateralOn = False

        self.lateralFrames = list()
        lateralCores = multiCast(coreId, self.inhiCoreStar, Hardware.COREBIT, None)
        for core in lateralCores:
            globalCoreId = Hardware.getgPlusCoreId2(chipId, core)
            self.lateralFrames.append(
                Frame.makeLateralFrame(globalCoreId, 0)
            )
        if 'neuron' in configs:
            self.initNeurons(configs['neuron'])
        else:
            assert self.tickWaitStart == 0
    
    def initNeurons(self, neuronConfigs):
        completeNum = len(neuronConfigs) // (self.LCN * self.weightWidth)
        neuronId = 0
        weightBase = 2 ** np.arange(self.weightWidth)
        if self.weightWidth > 1:
            weightBase[self.weightWidth - 1] = -weightBase[self.weightWidth - 1]
        segLen = Hardware.getAttr("AXONNUM", False)

        for i in range(completeNum):
            assert 'parameter' in neuronConfigs[neuronId], \
                f"{neuronId} {self.coreId} {self.chipId}"
            para = neuronConfigs[neuronId]['parameter']
            weights = list()
            paras = list()
            for j in range(self.LCN):
                weight = neuronConfigs[neuronId]['weight']
                paras.append(neuronConfigs[neuronId]['parameter'])
                neuronId += self.weightWidth
                # weight = 0
                # for k in range(self.weightWidth):
                #     for t in range(segLen):
                #         assert neuronConfigs[neuronId]['weight'] is not None, \
                #             f"{self.chipId},{self.coreId},{neuronId}"
                #         weight += np.array(neuronConfigs[neuronId]['weight']) * weightBase[k]
                #     neuronId += 1
                weights.append(weight)
            self.neurons.append(
                OnlineNeuron(
                    self.chipId, self.coreId, i, paras, weights,self.weightWidth, self.LCN
                )
            )
    
    def getInput(self, timeId):
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", False)
        slotBeg = (self.LCN * self.timeStep) % hardwareSlotNum
        slotEnd = (slotBeg + self.LCN)
        return self.inputBuffer[slotBeg: slotEnd, :]

    def compute(self, timeId):
        self.outputBuffer.clear()
        if not self.checkActive(timeId):
            return 
        else:
            inputBuffer = self.getInput(timeId)
            self.updatePreCounter(inputBuffer)

            slotBase = self.LCN * self.timeStep
            lateralVal = self.lateralInhi * self.lateralOn
            for i, neuron in enumerate(self.neurons):
                outputs = neuron.compute(
                    self.timeStep, inputBuffer,  
                    self.LCN, slotBase, lateralVal, self.onlineMode,
                    self.preTraces, self.LUT, self.lowerWeight, self.upperWeight
                )
                self.outputBuffer += outputs
            
            inputBuffer[:,:] = 0
            if self.onlineMode and len(self.outputBuffer) > 0:
                self.outputBuffer += deepcopy(self.lateralFrames)
    
            self.lateralOn = False
        
    def updatePreCounter(self, inputBuffer):
        if self.onlineMode:
            slotNum = inputBuffer.shape[0]
            axonNum = inputBuffer.shape[1]
            hasInput = (inputBuffer != 0).reshape(slotNum * axonNum)
            self.preFlag[hasInput] = True
            self.preTraces[hasInput] = 0
            self.preTraces[~hasInput] += 1
            self.preTraces = self.preTraces.clip(0, PRE_COUNTER_MAX)

    def setInit(self):
        super().setInit()
        return
    
    def setBeg(self):
        if self.onlineMode:
            self.preTraces[:] = 0
            self.preFlag[:] = 0
            self.lateralOn = False
            for neuron in self.neurons:
                neuron.setBeg()
    
    def setEnd(self):
        if self.onlineMode:
            for neuron in self.neurons:
                neuron.setEnd(self.preFlag, self.weightDecay, self.lowerWeight, self.upperWeight)

    def setLateral(self):
        if self.onlineMode:
            self.lateralOn = True
    
    def receive(self, intFrame):
        if ISSTART(intFrame):
            self.setBeg()
        elif ISEND(intFrame):
            self.setEnd()
        elif ISLATERAL(intFrame):
            self.setLateral()
        elif ISDATA(intFrame):
            axon = AXONID(intFrame)
            slot = SLOTID(intFrame)
            data = DATAID(intFrame)
            coreId = COREID(intFrame)
            starId = STARID(intFrame)
            if ((coreId ^ self.coreId) | starId) == starId:
                self.inputBuffer[slot, axon] = data
        else:
            self.updateState(intFrame)

class OfflineCore(Core):
    def __init__(self, chipId, coreId, configs):
        inputWidth = configs['core'][2]
        validAxonNum = 1152 if inputWidth == 0 else 144
        coreConfig = [
            256,
            validAxonNum,
            configs['core'][6],
            configs['core'][7]
        ]
        super().__init__(chipId, coreId, *coreConfig)

        self.weightWidth = 1 << configs['core'][0]
        self.LCN = 1 << configs['core'][1]
        self.inputWidth = configs['core'][2]
        self.spikeWidth = configs['core'][3]
        self.neuronNum = configs['core'][4]
        self.poolMax = configs['core'][5]
        self.SNN_EN = configs['core'][8]
        self.targetLCN = 1<<configs['core'][9]
        self.testChipAddr = configs['core'][10]
        self.LUT = []
        if self.tickWaitStart > 0:
            assert 'neuron' in configs
            self.initNeurons(configs['neuron'])

    def initNeurons(self, neuronConfigs):
        completeNum = len(neuronConfigs) // (self.LCN * self.weightWidth)
        neuronId = 0
        weightBase = 2 ** np.arange(self.weightWidth)
        if self.weightWidth > 1:
            weightBase[self.weightWidth - 1] = -weightBase[self.weightWidth - 1]

        for i in range(completeNum):
            assert 'parameter' in neuronConfigs[neuronId], \
                f"{neuronId} {self.coreId} {self.chipId}"
            para = neuronConfigs[neuronId]['parameter']
            weights = list()
            for j in range(self.LCN):
                weight = 0
                for k in range(self.weightWidth):
                    assert neuronConfigs[neuronId]['weight'] is not None, \
                        f"{self.chipId},{self.coreId},{neuronId}"
                    weight += np.array(neuronConfigs[neuronId]['weight']) * weightBase[k]
                    neuronId += 1
                weights.append(weight)
            self.neurons.append(
                OfflineNeuron(
                    self.chipId, self.coreId, i, [para], weights,self.weightWidth, self.LCN
                )
            )

    def setInit(self):
        super().setInit()
        return

    def compute(self, timeId):
        self.outputBuffer.clear()
        if not self.checkActive(timeId):
            return 
        else:
            hardwareSlotNum = Hardware.getAttr("SLOTNUM", True)
            slotBeg = (self.LCN * self.timeStep) % hardwareSlotNum
            # slotBeg = (self.LCN) % hardwareSlotNum - 1
            slotEnd = (slotBeg + self.LCN)
            # if not all zero
            # if self.inputBuffer[slotBeg:slotEnd,:].any():
            #     print(1)
            # else:
            #     print(0)
            # print(self.inputBuffer[slotBeg:slotEnd,:] )
            # print(f'slotBeg: {slotBeg}, slotEnd: {slotEnd}')
            # todo : 需要考虑tick wait start的情况
            slotBase = self.targetLCN * self.timeStep
            for i, neuron in enumerate(self.neurons):
                outputs = neuron.compute(
                    self.timeStep, self.inputBuffer[slotBeg:slotEnd,:],  
                    self.LCN, self.SNN_EN, self.spikeWidth, slotBase, self.poolMax
                )
                self.outputBuffer += outputs
            
            # paiflow origin
            self.inputBuffer[slotBeg : slotEnd, :] = 0
            # update
            # self.inputBuffer[:-1*self.LCN,:] = self.inputBuffer[self.LCN:,:]
            
    def receive(self, intFrame):
        if ISDATA(intFrame):
            axon = AXONID(intFrame)
            if self.inputWidth == 1:
                assert axon & 7 == 0
                axon = axon // 8
            slot = SLOTID(intFrame)
            data = DATAID(intFrame)
            coreId = COREID(intFrame)
            starId = STARID(intFrame)
            if ((coreId ^ self.coreId) | starId) == starId:
                self.inputBuffer[slot, axon] = data
        else:
            self.updateState(intFrame)

    def debugNeuronParas(self, realCoreId, neuronConfigs):
        completeNum = len(neuronConfigs) // (self.LCN * self.weightWidth)
        # completeNum = len(self.neurons)
        step = (self.LCN * self.weightWidth)
        neuronId= 0
        weightBase = 2 ** np.arange(self.weightWidth)
        if self.weightWidth > 1:
            weightBase[self.weightWidth - 1] = -weightBase[self.weightWidth - 1]

        for i in range(completeNum):
            assert 'parameter' in neuronConfigs[neuronId], \
                f"{neuronId} {self.coreId} {self.chipId}"
            para = neuronConfigs[neuronId]['parameter']
            if i < len(self.neurons):
                self.neurons[i].debugNeuronParas([para])
                print(f"[in]: {i}/{self.neuronNum} {para}")
            else:
                print(f"[out]: {i}/{self.neuronNum} {para}")
                # assert i >= self.neuronNum, f"{self.coreId}: {i} {self.neuronNum}\n{len(self.neurons)}"
            neuronId += step
            # weights = list()
            # for j in range(self.LCN):
            #     weight = 0
            #     for k in range(self.weightWidth):
            #         assert neuronConfigs[neuronId]['weight'] is not None, \
            #             f"{self.chipId},{self.coreId},{neuronId}"
            #         weight += np.array(neuronConfigs[neuronId]['weight']) * weightBase[k]
            #         neuronId += 1
            #     weights.append(weight)


class Chip:
    def __init__(self, chipId):
        self.chipId = chipId
        self.syncTimes = -1
        self.init = 0
        self.cores = dict()
        self.buffer = list()
        self.debugTime = -1
        self.debugInfo = dict()

    def setConfig(self, coreId, config):
        assert coreId not in self.cores
        if len(config['LUT']) == 0:
            self.cores[coreId] = OfflineCore(self.chipId, coreId, config)
        else:
            self.cores[coreId] = OnlineCore(self.chipId, coreId, config)
        # self.cores[coreId] = Core(self.chipId, coreId, config)

    def sendSpikes(self):
        offChipSpikes = list()
        num = 0
        for spike in self.buffer:
            if CHIPID(spike) != self.chipId:
                offChipSpikes.append(spike)
            else:
                coreId = COREID(spike)
                num += 1
                self.cores[coreId].receive(spike)
        # print(num)
        self.buffer.clear()
        return offChipSpikes

    def receiveSpikes(self, spikes):
        self.buffer.clear()
        for spike in spikes:
            assert CHIPID(spike) == self.chipId
            if ISSYNC(spike):
                syncTime = PAYLOAD(spike)
                self.setSyncTimes(syncTime)
            elif ISCLEAR(spike):
                self.setClear()
            elif ISINIT(spike):
                self.setInit()
            else:
                coreId = COREID(spike)
                starId = STARID(spike)
                cores = multiCast(coreId, starId, Hardware.COREBIT, None, self.cores)
                # print(cores)
                for core in cores:
                    # assert core in self.cores, f"{coreId}, {starId}, {cores} {self.cores.keys()}"
                    self.cores[core].receive(spike)

        return 
    
    def checkActive(self, timeId):
        if timeId > self.syncTimes and self.syncTimes > 0:
            return False
        active = False
        for coreId, core in self.cores.items():
            active = active or core.checkActive(timeId)
        return active
    
    def compute(self, timeId):
        if timeId > self.syncTimes and self.syncTimes > 0:
            return
        for coreId, core in self.cores.items():
            core.compute(timeId)
            self.buffer += core.outputBuffer
        if timeId == self.debugTime:
            print("[info] begin to debug")
            self.debug()
    
    def advanceTime(self, timeId):
        for coreId, core in self.cores.items():
            core.advanceTime(timeId)

    def setSyncTimes(self, syncTime):
        self.syncTimes = syncTime
    
    def setClear(self):
        return

    def setInit(self):
        for core in self.cores.values():
            core.setInit()
    
    def dumpWeight(self, coreId, completeNeuronId, LCN_id, axonId):
        val = self.cores[coreId].dumpWeight(
            completeNeuronId, LCN_id, axonId
        )
        return val
    
    def debug(self):
        for coreId, config in tqdm(self.debugInfo.items(), "debug core parameters"):
            self.cores[coreId].debugNeuronParas(coreId, config['neuron'])

    def debugNeuronParas(self, coreId, config, timeStep):
        assert coreId in self.cores
        assert self.debugTime == -1 or timeStep == self.debugTime
        self.debugTime = timeStep
        assert coreId not in self.debugInfo
        self.debugInfo[coreId] = config
        
    
class Simulator:
    def __init__(self, TimestepVerbose):
        self.chips = dict()
        self.buffer = list()
        self.outputBuffer = list()
        self.debugFrameDir = None
        self.TimestepVerbose = TimestepVerbose

    def setConfig(self, configs):
        print("")
        print("[Configure]")
        for coreId, config in tqdm(configs.items(), "configure cores"):
            chipId = coreId >> 10
            realCoreId = coreId & MASK.COREMASK
            if chipId not in self.chips:
                self.chips[chipId] = Chip(chipId)
            self.chips[chipId].setConfig(realCoreId, config)
        cores = list(configs.keys())
        # draw(cores,"tmp.png")
    
    def setDebug(self, debugFrameDir):
        self.debugFrameDir = debugFrameDir

    def debugNeuronParas(self, testFramePath, timeStep):
        configs = parseTestFrames(testFramePath)
        for coreId, config in configs.items():
            chipId = coreId >> 10
            realCoreId = coreId & MASK.COREMASK
            # assert chipId in self.chips, chipId
            # self.chips[chipId].debugNeuronParas(realCoreId, config, timeStep)
            self.chips[0].debugNeuronParas(realCoreId, config, timeStep)

    def setInputs(self, spikes):
        spikesDict = dict()
        pos = 0

        for pos, spike in enumerate(spikes):
            chipId = CHIPID(spike)
            if chipId not in spikesDict:
                spikesDict[chipId] = list()
            spikesDict[chipId].append(spike)

        for chipId, s in spikesDict.items():
            # assert chipId in self.chips, f"{chipId}:{self.chips.keys()}"
            if chipId in self.chips:
                self.chips[chipId].receiveSpikes(s)
            else:
                self.outputBuffer += s
        # print("setinputs")
    def setInputs2(self, spikes):
        spikesDict = dict()
        pos = 0
        for pos, spike in enumerate(spikes):
            # spike = int(spike,2)
            chipId = CHIPID(spike)
            if ISSYNC(spike):
                self.chips[chipId].setSyncTimes(PAYLOAD(spike))
            elif ISINIT(spike):
                self.chips[chipId].setInit()
            elif ISSTART(spike):
                coreId = COREID(spike)
                starId = STARID(spike)
                cores = multiCast(coreId, starId, Hardware.COREBIT, None)
                for core in cores:
                    self.chips[chipId].setBeg(core)
            elif ISEND(spike):
                coreId = COREID(spike)
                starId = STARID(spike)
                cores = multiCast(coreId, starId, Hardware.COREBIT, None)
                for core in cores:
                    self.chips[chipId].setEnd(core)
            elif ISLATERAL(spike):
                assert False
                coreId = COREID(spike)
                starId = STARID(spike)
                cores = multiCast(coreId, starId, Hardware.COREBIT, None)
                for core in cores:
                    self.chips[chipId].setLateral(core)
            else:
                assert ISDATA(spike)
                if chipId not in spikesDict:
                    spikesDict[chipId] = list()
                spikesDict[chipId].append(spike)
        assert False

        for chipId, s in spikesDict.items():
            # assert chipId in self.chips, f"{chipId}:{self.chips.keys()}"
            if chipId in self.chips:
                self.chips[chipId].receiveSpikes(s)
            else:
                self.outputBuffer.append(spike)

        # assert False
        
    def advanceTime(self, timeId):
        for chipId, chip in self.chips.items():
            chip.advanceTime(timeId)

    def begin(self):
        self.outputBuffer.clear()
        active = True
        timeStep = 1
        # print("begin")
        if(self.TimestepVerbose):
            print("")
        while active:

            for chipId, chip in self.chips.items():
                chip.compute(timeStep)
                spikes = chip.sendSpikes()
                self.buffer += spikes
            # priont()
            self.setInputs(self.buffer)

            self.buffer.clear()
            
            # update clock
            self.advanceTime(timeStep)
            timeStep += 1
            
            if(self.TimestepVerbose):
                print("TimeStep:" + str(timeStep - 1) + "\r",end="")
            #determine if all the chips stop or not
            active = False
            for chipId, chip in self.chips.items():
                active = active or chip.checkActive(timeStep)

        return

    def dumpWeight(self, globalCoreId, completeNeuronId, LCN_id, axonId):
        chipId = globalCoreId >> Hardware.COREBIT
        coreId = globalCoreId & (MASK.COREMASK)
        val = self.chips[chipId].dumpWeight(
            coreId, completeNeuronId, LCN_id, axonId
        )
        return val

def draw(cores, path):
    # import matplotlib.pyplot as plt
    # xlist = list()
    # ylist = list()
    # for core in cores:
    #     x = (core >> 5) & 31
    #     y = core & 31
    #     xlist.append([x,x+1, x + 1, x])
    #     ylist.append([y, y, y+1, y+1])
    # ax = plt.axes([0.075,0.075,0.9,0.9])

    # ax.set_xlim(0,32)
    # ax.set_ylim(0,32)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    # # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    # # ax.grid(which='minor', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    # ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    # # ax.grid(which='minor', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    # # ax.set_xticklabels([])
    # # ax.set_yticklabels([])
    # i = 0
    # res = set()
    # for x,y in zip(xlist,ylist):
    #     assert (x[0] * 32 + y[0]) not in res
    #     res.add(x[0] * 32 + y[0])
    #     plt.fill(x,y,color='black')

    # plt.savefig(path)
    pass

# only for FPGA version and offline cores, for debug
def parseTestFrames(testFramePath):

    def parseTest3(frameGroup):
        intFrame0 = int(frameGroup[0],2)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        begWith0 = neuronId == 0
        if not begWith0:
            assert neuronId == 1
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
                config[neuronId] = {'parameter':tmpConfig}
                neuronId += 1
                if begWith0 and neuronId == 1:
                    neuronId += 1
                memBase = 0
        return config

    with open(testFramePath,'r') as f:
        frames = f.readlines()
    frameNum = len(frames)
    configs =  dict()
    i = 0
    while i < frameNum:
        frame = frames[i].strip()
        # intFrame = int(frame,2)
        intFrame = Frame.toInt(frame)
        coreId = GLOBAL_CORE_ID(intFrame)
        starId = STARID(intFrame)
        assert starId == 0
        assert Frame.isKind(intFrame, FrameKind.TEST3_OUT)
        frameNum1 = Frame.getFrameNum(intFrame) 
        end = i + frameNum1 + 1
        config = parseTest3(frames[i:end])

        if coreId not in configs:
            configs[coreId] = {'neuron': config}
        else:
            configs[coreId]['neuron'].update(config)
        i = end

    return configs