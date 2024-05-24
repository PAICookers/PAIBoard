import os
import pickle
import json
import numpy as np
from hwConfig import Hardware
import binascii,time
from timeMeasure import time_calc_addText, get_original_function

def createFile(filePath):
    if os.path.exists(filePath):
        return 1
    else:
        try:
            os.mkdir(filePath)
        except Exception as e:
            os.makedirs(filePath)
        return 0

# Simulator
def loadInputFormats(baseDir):
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(formatDir, "formats.pkl"),"rb") as formatInputFile:
        inputFormats = pickle.load(formatInputFile)
    with open(os.path.join(formatDir, "numbers.pkl"),"rb") as numberFile:
        numbers = pickle.load(numberFile)
    with open(os.path.join(formatDir, "inputNames.pkl"),"rb") as nameInputFile:
        inputNames = pickle.load(nameInputFile)
    return inputFormats, numbers, inputNames

def loadOutputFormats(baseDir):
    infoDir = os.path.join(baseDir, "info")
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(infoDir, "outDict.json"),"r") as f:
        outDict = json.load(f)
    with open(os.path.join(infoDir, "shape.json"),"r") as f:
        shapeDict = json.load(f)
    with open(os.path.join(infoDir, "scale.json"),"r") as f:
        scaleDict = json.load(f)
    with open(os.path.join(formatDir, "mapper.txt"),"r") as f:
        mapper = json.load(f)
    intMapper = dict()
    for name, [tensorName, pos] in mapper.items():
        intMapper[int(name)] = [tensorName, int(pos)]
    return outDict, shapeDict, scaleDict, intMapper

def serialConfig(globalSignalDelay = 92):
    import serial
    ser = serial.Serial("/dev/ttyUSB0", 9600)
    if ser.isOpen():                        # 判断串口是否成功打开
        print("[Info]  : Serial Open.")
    else:
        print("[Error] : Serial Not Open.")
        return 1

    # b = hex(globalSignalDelay)[2:]
    b = '{:02x}'.format(globalSignalDelay)
    uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 64 05 90 00 ' + b +' F8 C8')   # 312M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 6C 05 B0 00 ' + b +' F8 C8')   # 336M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 38 00 E0 00 ' + b +' F8 C8')   # 360M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 3C 00 F0 00 ' + b +' F8 C8')   # 384M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 40 00 10 00 ' + b +' F8 C8')   # 408M error
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 50 01 40 00 ' + b +' F8 C8')   # 504M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 60 01 80 00 ' + b +' F8 C8')   # 600M
    write_len=ser.write(uarthex)

    time.sleep(0.2)
    count = ser.inWaiting()

    data = None
    if count > 0:
        data=ser.read(count)
        if data!=b'':
            dataStr = str(binascii.b2a_hex(data))[2:-1]
            print("receive:",dataStr)
        else:
            return 2
        # if dataStr != 'fffffffffffffffe640590005cf8c8':
        #     return 3
    if data == None:
        return 4
    
    ser.close()
    if ser.isOpen():
        print("[Error] : Serial Not Close.")
    else:
        print("[Info]  : Serial Close. Uart send Done!")
    
    return 0

# Tensor2Frame
def binary_to_uint64(binary_string):
    return np.uint64(int(binary_string, 2))

def GEN_INIT_SYNC(frameFormats,frameNums):
    init_frames_nums = frameNums[0]
    # print("init_frames_nums",init_frames_nums)

    vectorized_binary_to_uint64 = np.vectorize(binary_to_uint64)

    # for init frames
    initFrames_str = frameFormats[:init_frames_nums]
    initFrames_np = vectorized_binary_to_uint64(initFrames_str)

    # for sync frames
    syncFrames_str = [frameFormats[-1]]
    syncFrames_np = vectorized_binary_to_uint64(syncFrames_str)

    return initFrames_np, syncFrames_np

@time_calc_addText("Tensor2Frame  ")
def Tensor2Frame(dataDict, frameFormats, nameLists):
    for name in nameLists:
        if hasattr(dataDict[name], "timesteps"):
            data = dataDict[name].data
            data = data.detach().numpy().reshape(-1)
        else:
            data = dataDict[name].detach().numpy().reshape(-1)
    # data = dataDict.reshape(-1).astype(np.uint64)
    data = data.astype(np.uint64)
    indexes  = np.nonzero(data)
    data = data[indexes]
    frameFormats = frameFormats[indexes]
    dataFrames = frameFormats << np.uint64(8) | data 
    return dataFrames


def Tensor2FrameWrap(dataDict, frameFormats, nameLists, TimeMeasure):
    if(TimeMeasure):
        return Tensor2Frame(dataDict, frameFormats, nameLists)
    else:
        original_Tensor2Frame = get_original_function(Tensor2Frame)
        return original_Tensor2Frame(dataDict, frameFormats, nameLists)


@time_calc_addText("Frame2Tensor  ")
def Frame2Tensor(
    dataFrames, outputDict, shapeDict, 
    scaleDict, mapper, timeStep, coreType):
    dataDict = dict()
    shapeLen = dict()
    timeSteps = dict()
    for name, shape in shapeDict.items():
        shapeLen[name] = np.prod(shape)
        dataDict[name] = np.zeros(shapeLen[name] * timeStep)
        timeSteps[name] = 0
    
    #for both online and offline  
    hardwareAxonBit = Hardware.getAttr("AXONBIT", True)
    if coreType == 'offline':
        timeSlotNum = 256
    else:
        timeSlotNum = 8
    for frameId, frame in enumerate(dataFrames):
        if frame == '':
            break
        pos = (int(frame[4:24],2) << hardwareAxonBit) + int(frame[37:48],2)
        data = int(frame[-8:],2)
        newTimeStep = int(frame[48:56],2)
        name, tensorPos = mapper[pos]
        if timeSteps[name] % timeSlotNum <= newTimeStep:
            timeSteps[name] += (newTimeStep - timeSteps[name] % timeSlotNum)
            if dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]]:
                timeSteps[name] += (timeSlotNum - (timeSteps[name] % timeSlotNum)) + newTimeStep
        else:
            timeSteps[name] += (timeSlotNum - (timeSteps[name] % timeSlotNum)) + newTimeStep
        if(timeSteps[name] == 256):
            timeSteps[name] = 0
        # assert dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]] ==0, f"{timeSteps[name]}"
        dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]] = data

    for name, outputs in outputDict.items():
        pos = 0
        if len(outputs) == 1 and outputs[0] == name:
            continue
        for output in outputs:
            dataDict[name][pos: (pos + shapeLen[output] * timeStep)] = dataDict[output][:]
            pos += shapeLen[output] * timeStep
    realDataDict = dict()
    realDataSpikeDict = dict()
    for name in outputDict.keys():
        realDataDict[name] = dataDict[name].reshape(timeStep, *shapeDict[name])
    for name in outputDict.keys():
        shape = shapeDict[name]
        scale = np.array(scaleDict[name])
        realDataSpikeDict[name] = realDataDict[name].reshape(timeStep, *shape)
        realDataDict[name] = realDataDict[name].reshape(timeStep, *shape).mean(0) * scale

    # Dict -> Data
    for name, shape in shapeDict.items():
        outData = realDataDict[name]
        outData_spk = realDataSpikeDict[name]

    return outData, outData_spk

def Frame2TensorWrap(
    dataFrames, outputDict, shapeDict, 
    scaleDict, mapper, timeStep, coreType , TimeMeasure):
    if(TimeMeasure):
        return Frame2Tensor(dataFrames, outputDict, shapeDict, 
                            scaleDict, mapper, timeStep, coreType)
    else:
        original_Frame2Tensor = get_original_function(Frame2Tensor)
        return original_Frame2Tensor(dataFrames, outputDict, shapeDict, 
                                     scaleDict, mapper, timeStep, coreType)


def npFrame2txt(dataPath,inputFrames):
    with open(dataPath, 'w') as f:
        for i in range(inputFrames.shape[0]):
            f.write("{:064b}\n".format(inputFrames[i]))

def strFrame2txt(dataPath,inputFrames):
    with open(dataPath, 'w') as f:
        for i in range(len(inputFrames)):
            f.write(inputFrames[i] + "\n")

def binFrame2Txt(configPath):
    configFrames = np.fromfile(configPath, dtype='<u8')
    fName, _ = os.path.splitext(configPath)
    configTxtPath = fName + '.txt'
    npFrame2txt(configTxtPath,configFrames)
    print(f"[generate] Generate frames as txt file")

def txtFrame2Bin(configTxtPath):
    config_frames = np.loadtxt(configTxtPath, str)
    config_num = config_frames.size
    config_buffer = np.zeros((config_num,), dtype=np.uint64)
    for i in range(0, config_num):
        config_buffer[i] = int(config_frames[i], 2)
    config_frames = config_buffer
    fName, _ = os.path.splitext(configTxtPath)
    configPath = fName + '.bin'
    config_frames.tofile(configPath)
    print(f"[generate] Generate frames as bin file")


if __name__ == "__main__":
    serialConfig(globalSignalDelay = 92)