# import numpy as np

# from paiboard.dma.base import DMA_base 

# from paiboard.ethernet.utils_for_ethernet import *

# class DMA_Ethernet(DMA_base):
#     def __init__(self) -> None:
#         super().__init__()

#         ip = '192.168.31.100'
#         port = 8889
#         self.addr = (ip, port)
#         buffer_bytes = 4096
#         self.buffer_num = int(buffer_bytes/8) # for uint64

#     def read_reg(self, addr):
#         raise NotImplementedError

#     def write_reg(self, addr, data):
#         raise NotImplementedError

#     def send_frame(self, send_data):


#     def recv_frame(self, oFrmNum):
