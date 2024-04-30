import numpy as np

from paiboard.dma.base import DMA_base 
try:
    from paiboard.pcie.example import pcie_init,read_bypass,write_bypass
    from paiboard.pcie.example import send_dma_np, read_dma_np
except:
    ImportError
from paiboard.utils.timeMeasure import time_calc_addText, get_original_function

class DMA_PCIe(DMA_base):
    def __init__(self, oen: int, channel_mask: int) -> None:
        super().__init__()

        if(pcie_init() < 0):
            print("pcie_init error")

        self.REGFILE_BASE = 0x00000
        write_bypass(self.REGFILE_BASE + self.CPU2FIFO_CNT, 0)
        write_bypass(self.REGFILE_BASE + self.FIFO2SNN_CNT, 0)
        write_bypass(self.REGFILE_BASE + self.SNN2FIFO_CNT, 0)
        write_bypass(self.REGFILE_BASE + self.FIFO2CPU_CNT, 0)

        write_bypass(self.REGFILE_BASE + self.DP_RSTN, 0)
        write_bypass(self.REGFILE_BASE + self.DP_RSTN, 1)

        write_bypass(self.REGFILE_BASE + self.OEN, oen)
        write_bypass(self.REGFILE_BASE + self.CHANNEL_MASK, channel_mask)

    def read_reg(self, addr):
        return read_bypass(addr)

    def write_reg(self, addr, data):
        write_bypass(addr, data)

    @time_calc_addText("SendFrame     ")
    def send_frame(self, send_data, multi_channel_enable = False):
        if(multi_channel_enable):
            write_bypass(self.REGFILE_BASE + self.SINGLE_CHANNEL, 0)
        else:
            write_bypass(self.REGFILE_BASE + self.SINGLE_CHANNEL, 1)
        write_byte_nums = send_data.size << 3
        write_bypass(self.REGFILE_BASE + self.SEND_LEN, write_byte_nums >> 3)
        rc = send_dma_np(send_data,write_byte_nums)

        val = 0
        while(val == 0):
            val = read_bypass(self.REGFILE_BASE + self.TX_STATE)
        write_bypass(self.REGFILE_BASE + self.TX_STATE,0)

    def send_config_frame(self, send_data):
        self.send_frame(send_data, multi_channel_enable = False)

    @time_calc_addText("RecvFrame     ")
    def recv_frame(self, oFrmNum):
        write_bypass(self.REGFILE_BASE + self.RX_STATE,1)
        
        rc ,outputFrames = read_dma_np(oFrmNum << 3)
        outputFrames = np.delete(outputFrames,np.where(outputFrames == 0))
        outputFrames = np.delete(outputFrames,np.where(outputFrames == 18446744073709551615))

        val = 1
        while(val == 1):
            val = read_bypass(self.REGFILE_BASE + self.RX_STATE)
        write_bypass(self.REGFILE_BASE + self.RX_STATE,0)
        return outputFrames