import numpy as np
from timeMeasure import time_calc_addText, get_original_function

try:
    from example import _add_arrays_1d
    from example import pcie_init,read_bypass,write_bypass,write_control
    from example import send_dma,read_dma
    from example import send_dma_np, read_dma_np
except:
    ImportError
    
REGFILE_BASE = 0x00000
BP_LED_BASE  = 0x10000

CT_LED0_BASE = 0x00000
CT_LED1_BASE = 0x10000

RX_STATE        = 0  * 4
TX_STATE        = 1  * 4
CPU2FIFO_CNT    = 2  * 4
FIFO2SNN_CNT    = 3  * 4
SNN2FIFO_CNT    = 4  * 4
FIFO2CPU_CNT    = 5  * 4
WDATA_1         = 6  * 4
WDATA_2         = 7  * 4
RDATA_1         = 8  * 4
RDATA_2         = 9  * 4
DATA_CNT        = 10 * 4
TLAST_CNT       = 11 * 4

SEND_LEN        = 20 * 4
CTRL_REG        = 21 * 4
oFrmNum_REG     = 22 * 4
DP_RSTN         = 23 * 4
SINGLE_CHANNEL  = 24 * 4
CHANNEL_MASK    = 25 * 4
OEN             = 26 * 4


def show_reg_status():
    print()
    print("RX_STATE      : %d" % read_bypass(REGFILE_BASE + RX_STATE      ))
    print("TX_STATE      : %d" % read_bypass(REGFILE_BASE + TX_STATE      ))
    print("CPU2FIFO_CNT  : %d" % read_bypass(REGFILE_BASE + CPU2FIFO_CNT  ))
    print("FIFO2SNN_CNT  : %d" % read_bypass(REGFILE_BASE + FIFO2SNN_CNT  ))
    print("SNN2FIFO_CNT  : %d" % read_bypass(REGFILE_BASE + SNN2FIFO_CNT  ))
    print("FIFO2CPU_CNT  : %d" % read_bypass(REGFILE_BASE + FIFO2CPU_CNT  ))
    print("SEND_LEN      : %d" % read_bypass(REGFILE_BASE + SEND_LEN      ))
    print("CTRL_REG      : %d" % read_bypass(REGFILE_BASE + CTRL_REG      ))
    val1 = read_bypass(REGFILE_BASE + WDATA_1)
    val2 = read_bypass(REGFILE_BASE + WDATA_2)
    print("WDATA         : 0x%016x" % (val2 << 32 | val1))
    val1 = read_bypass(REGFILE_BASE + RDATA_1)
    val2 = read_bypass(REGFILE_BASE + RDATA_2)
    print("RDATA         : 0x%016x" % (val2 << 32 | val1))
    print("DATA_CNT      : %d" % read_bypass(REGFILE_BASE + DATA_CNT      ))
    val = read_bypass(REGFILE_BASE + TLAST_CNT)
    print("TLAST_IN_CNT  : %d" % (val & 0x0000FFFF))
    print("TLAST_OUT_CNT : %d" % (val >> 16))
    print()


def dma_init(oFrmNum, oen = 0b1110, channel_mask = 0b1000):

    if(pcie_init() < 0):
        print("pcie_init error")

    write_bypass(REGFILE_BASE + CPU2FIFO_CNT, 0)
    write_bypass(REGFILE_BASE + FIFO2SNN_CNT, 0)
    write_bypass(REGFILE_BASE + SNN2FIFO_CNT, 0)
    write_bypass(REGFILE_BASE + FIFO2CPU_CNT, 0)

    write_bypass(REGFILE_BASE + DP_RSTN, 0)
    write_bypass(REGFILE_BASE + DP_RSTN, 1)

    write_bypass(REGFILE_BASE + oFrmNum_REG, oFrmNum)

    write_bypass(REGFILE_BASE + OEN, oen)
    write_bypass(REGFILE_BASE + CHANNEL_MASK, channel_mask)

@time_calc_addText("SendFrame     ")
def SendFrame(send_data, multi_channel_enable = False):
    if(multi_channel_enable):
        write_bypass(REGFILE_BASE + SINGLE_CHANNEL, 0)
    else:
        write_bypass(REGFILE_BASE + SINGLE_CHANNEL, 1)
    write_byte_nums = send_data.size << 3 # byte_nums
    write_bypass(REGFILE_BASE + SEND_LEN, write_byte_nums >> 3)
    rc = send_dma_np(send_data,write_byte_nums)
    # print("send %d bytes." % rc)

    val = 0
    while(val == 0):
        val = read_bypass(REGFILE_BASE + TX_STATE)
    write_bypass(REGFILE_BASE + TX_STATE,0)


def SendFrameWrap(send_data, multi_channel_enable = False, TimeMeasure = False):
    if(TimeMeasure):
        return SendFrame(send_data, multi_channel_enable) # work
    else:
        original_Tensor2Frame = get_original_function(SendFrame)
        return original_Tensor2Frame(send_data, multi_channel_enable)


def RecvFrameStream(oFrmNum):
    write_bypass(REGFILE_BASE + RX_STATE,1)
    rc ,outputFrames = read_dma_np(oFrmNum << 3)
    outputFrames = np.delete(outputFrames,np.where(outputFrames == 0))
    outputFrames = np.delete(outputFrames,np.where(outputFrames == 18446744073709551615))
    # print("read %d bytes." % rc)

    val = 1
    while(val == 1):
        val = read_bypass(REGFILE_BASE + RX_STATE)
    write_bypass(REGFILE_BASE + RX_STATE,0)
    return outputFrames


def RecvFrameReg(oFrmNum):
    write_bypass(REGFILE_BASE + RX_STATE,1)
    val = 1
    while(val == 1):
        val = read_bypass(REGFILE_BASE + RX_STATE)
    write_bypass(REGFILE_BASE + RX_STATE,0)

    outputFrames = np.zeros((oFrmNum,), dtype='<u8')
    for i in range(oFrmNum):
        outDataLow  = read_bypass(RECV_DAT_BASE + i * 8)
        outDataHigh = read_bypass(RECV_DAT_BASE + i * 8 + 4)
        outputFrames[i] = outDataHigh << 32 | outDataLow
    outputFrames = np.delete(outputFrames,np.where(outputFrames == 0))
    outputFrames = np.delete(outputFrames,np.where(outputFrames == 18446744073709551615))

    return outputFrames

@time_calc_addText("RecvFrame     ")
def RecvFrame(oFrmNum, REG_RECV):
    if(REG_RECV):
        outputFrames = RecvFrameReg(oFrmNum)
    else:
        outputFrames = RecvFrameStream(oFrmNum)
    return outputFrames
    
@time_calc_addText("Init          ")
def Init(initFrames):
    write_bypass(REGFILE_BASE + CTRL_REG, 4)
    SendFrameWrap(initFrames)
    write_bypass(REGFILE_BASE + CTRL_REG, 0)

def WorkOnce(initFrames, inputFrames, oFrmNum, REG_RECV = False, TimeMeasure = False):

    if(TimeMeasure):
        Init(initFrames)
        SendFrame(inputFrames, multi_channel_enable=True) # work
        outputFrames = RecvFrame(oFrmNum, REG_RECV)
    else:
        original_Init = get_original_function(Init)
        original_Init(initFrames)

        original_SendFrame = get_original_function(SendFrame)
        original_SendFrame(inputFrames, multi_channel_enable=True)

        original_RecvFrame = get_original_function(RecvFrame)
        outputFrames = original_RecvFrame(oFrmNum, REG_RECV)

    return outputFrames