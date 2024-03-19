import numpy as np
import os

from paiboard.base import PAIBoard

from paiboard.ethernet.utils_for_ethernet import *

class PAIBoard_Ethernet(PAIBoard):

    def __init__(
        self,
        baseDir: str,
        timestep: int,
        layer_num: int = 0,
        output_delay: int = 0,
        batch_size: int = 1,
    ):
        super().__init__(baseDir, timestep, layer_num, output_delay, batch_size)

    def config(self, oFrmNum: int = 10000):

        ip = '192.168.31.100'
        port = 8889
        self.addr = (ip, port)
        buffer_bytes = 4096
        self.buffer_num = int(buffer_bytes/8) # for uint64

        configPath = os.path.join(self.baseDir, "config_cores_all.bin")
        configFrames = np.fromfile(configPath, dtype="<u8")
        print("----------------------------------")
        print("----------PAICORE CONFIG----------")
        tcpCliSock = Ethernet_config(self.addr)
        Ethernet_send(tcpCliSock, "CONFIG", configFrames, self.buffer_num)
        tcpCliSock.close()
        print("----------------------------------")

    def inference(self, initFrames, inputFrames):

        workFrames = np.concatenate((initFrames, inputFrames))
        # workFrames = inputFrames

        tcpCliSock = Ethernet_config(self.addr)
        Ethernet_send(tcpCliSock, "WORK", workFrames, self.buffer_num)
        outputFrames =  Ethernet_recv(tcpCliSock, self.buffer_num)
        tcpCliSock.close()


        return outputFrames
