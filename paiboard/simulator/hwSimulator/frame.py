class MASK:

    HEADBEGIN = 60

    CHIPBEGIN = 50
    CHIPMASK = ((1 << 10) - 1)

    COREBEGIN = 40
    COREMASK = ((1 << 10) - 1)

    GLOBALMASK = ((1 << 20) - 1)

    STARBEGIN = 30
    STARMASK = ((1 << 10) - 1)
    
    AXONBEGIN = 16
    AXONMASK = ((1 << 11) - 1)

    SLOTBEGIN = 8
    SLOTMASK = ((1 << 8) - 1)

    DATABEGIN = 0
    DATAMASK = ((1 << 8) - 1)
    FULLPAYLOAD = (1 << 30) - 1

    SRAMBEGIN = 20
    SRAMMASK = ( 1 << 10) - 1

    FRAMENUM_MASK = (1 << 19) - 1

class FrameKind:
    SYNC = [
        (9 << 60), (15 << 60)
    ]
    CLEAR = [
        (10 << 60), (15 << 60)
    ]
    INIT = [
        (11 << 60), (15 << 60)
    ]
    
    START = [
        (8 << 60) + (4 << 27), (15 << 60) + (7 << 27)
    ]
    END =  [
        (8 << 60) + (2 << 27), (15 << 60) + (7 << 27)
    ]
    LATERAL = [
        (8 << 60) + (1 << 27), (15 << 60) + (7 << 27)
    ]
    DATA = [
        (8 << 60), (15 << 60)
    ]
    TEST3_IN = [
        (6 << 60) + (1 << 19), (15 << 60) + (1 << 19)
    ]
    TEST3_OUT = [
        (6 << 60) + (0 << 19), (15 << 60) + (1 << 19)
    ]
    TEST4_IN = [
        (7 << 60) + (1 << 19), (15 << 60) + (1 << 19)
    ]
    TEST4_OUT = [
        (7 << 60) + (1 << 19), (15 << 60) + (1 << 19)
    ]

class Frame:
            
    @staticmethod
    def toString(intFrame):
        return "{:064b}".format(intFrame)
    
    @staticmethod
    def toInt(strFrame):
        return int(strFrame,2)
    
    @staticmethod
    def getChipId(intFrame):
        return (intFrame >> MASK.CHIPBEGIN) & MASK.CHIPMASK
    
    @staticmethod
    def getCoreId(intFrame):
        return (intFrame >> MASK.COREBEGIN) & MASK.COREMASK
    
    @staticmethod
    def getGlobalId(intFrame):
        return (intFrame >> MASK.COREBEGIN) & MASK.GLOBALMASK
    
    @staticmethod
    def getStarId(intFrame):
        return (intFrame >> MASK.STARBEGIN) & MASK.STARMASK

    @staticmethod
    def getAxonId(intFrame):
        return (intFrame >> MASK.AXONBEGIN) & MASK.AXONMASK

    @staticmethod
    def getSlotId(intFrame):
        return (intFrame >> MASK.SLOTBEGIN) & MASK.SLOTMASK
    
    @staticmethod
    def getData(intFrame):
        return (intFrame >> MASK.DATABEGIN) & MASK.DATAMASK

    @staticmethod
    def getPayload(intFrame):
        return (intFrame >> MASK.DATABEGIN) & MASK.FULLPAYLOAD

    @staticmethod
    def getFrameNum(intFrame):
        return intFrame & MASK.FRAMENUM_MASK
    
    @staticmethod
    def isKind(intFrame, frameKind):
        return (intFrame & frameKind[1]) == frameKind[0]
    
    @staticmethod
    def makeSyncFrame(chipId, t):
        return FrameKind.SYNC[0] | (chipId << MASK.CHIPBEGIN) | t

    @staticmethod
    def makeClearFrame(chipId):
        return FrameKind.CLEAR[0] | (chipId << MASK.CHIPBEGIN)

    @staticmethod
    def makeInitFrame(chipId):
        return FrameKind.INIT[0] | (chipId << MASK.CHIPBEGIN)

    @staticmethod
    def makeWorkFrame(FrameTemplate, globalCoreId, starId):
        return FrameTemplate | (globalCoreId << MASK.COREBEGIN) | (starId << MASK.STARBEGIN)

    @staticmethod
    def makeStartFrame(globalCoreId, starId):
        return Frame.makeWorkFrame(FrameKind.START[0], globalCoreId, starId)
    
    @staticmethod
    def makeEndFrame(globalCoreId, starId):
        return Frame.makeWorkFrame(FrameKind.END[0], globalCoreId, starId)

    @staticmethod
    def makeLateralFrame(globalCoreId, starId):
        return Frame.makeWorkFrame(FrameKind.LATERAL[0], globalCoreId, starId)
    
    @staticmethod
    def makeDataFrame(globalCoreId, starId, axonId, slotId, data):
        return FrameKind.DATA[0] | (globalCoreId << MASK.COREBEGIN) \
            | (axonId << MASK.AXONBEGIN) | (starId << MASK.STARBEGIN) | (slotId << MASK.SLOTBEGIN) | data
    
    @staticmethod
    def makeTest3InFrame(globalCoreId, starId, sram, frameNum):
        return FrameKind.TEST3_IN[0] | (globalCoreId << MASK.COREBEGIN) | (starId << MASK.STARBEGIN) \
            | (sram << MASK.SRAMBEGIN) | frameNum

    @staticmethod
    def makeTest4InFrame(globalCoreId, starId, sram, frameNum):
        return FrameKind.TEST4_IN[0] | (globalCoreId << MASK.COREBEGIN) | (starId << MASK.STARBEGIN) \
            | (sram << MASK.SRAMBEGIN) | frameNum
    
    @staticmethod
    def makePosFrame(globalCoreId, starId, axonId, slotId):
        return (globalCoreId << MASK.COREBEGIN) | (starId << MASK.STARBEGIN) \
            | (axonId << MASK.AXONBEGIN) | (slotId << MASK.SLOTBEGIN)

    @staticmethod
    def makeInputFormat(globalCoreId, starId, payload):
        return Frame.makeDataFrame(
            globalCoreId, starId, 0, 0, payload) >> 8


    @staticmethod
    def genDTSim(chipId, coreId, starId, axonId): #gen data template for simulation
        #without time slot and data
        return FrameKind.DATA[0] | (chipId << MASK.CHIPBEGIN) \
            |(coreId << MASK.COREBEGIN) | (starId << MASK.STARBEGIN) \
            | (axonId << MASK.AXONBEGIN)

    @staticmethod
    def genDF(DTSim, slotId, data): #gen data frame for simulation
        return DTSim | (slotId << MASK.SLOTBEGIN) | (data << MASK.DATABEGIN)
