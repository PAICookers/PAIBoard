import numpy as np

from paiboard.dma.base import DMA_base
from paiboard.ethernet.utils_for_ethernet import *
from paiboard.utils.timeMeasure import time_calc_addText, get_original_function

class DMA_Ethernet(DMA_base):
    def __init__(self) -> None:
        super().__init__()

        ip = "192.168.31.100"
        port = 8889
        self.addr = (ip, port)
        buffer_bytes = 4096
        self.buffer_num = int(buffer_bytes / 8)  # for uint64
        self.tcpCliSock = Ethernet_config(self.addr)

        self.REGFILE_BASE = 0x00000
    
    def read_reg(self, addr):
        reg_addr = np.array([addr], dtype=np.uint64)
        return Ethernet_recv(self.tcpCliSock, self.buffer_num, oFrmNum=1,read_reg=True, reg_addr=reg_addr)[0]

    def write_reg(self, addr, data):

        configFrames = np.array([addr, data], dtype=np.uint64)
        Ethernet_send(self.tcpCliSock, "WRITE REG", configFrames, self.buffer_num)

    def send_config_frame(self, send_data):
        Ethernet_send(self.tcpCliSock, "SEND", send_data, self.buffer_num)

    @time_calc_addText("SendFrame     ")
    def send_frame(self, send_data):
        Ethernet_send(self.tcpCliSock, "SEND", send_data, self.buffer_num)

    @time_calc_addText("RecvFrame     ")
    def recv_frame(self, oFrmNum):
        outputFrames = Ethernet_recv(self.tcpCliSock, self.buffer_num, oFrmNum=oFrmNum)
        print(f"RecvFrame: {outputFrames.size}")
        return outputFrames

    def __del__(self):
        # print("Close connection")
        Ethernet_send(self.tcpCliSock, "QUIT", None, self.buffer_num)
        # self.tcpCliSock.close()
        pass