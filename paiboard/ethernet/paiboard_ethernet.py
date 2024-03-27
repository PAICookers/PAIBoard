import numpy as np
import os

from paiboard.base import PAIBoard
from paiboard.ethernet.dma_ethernet import DMA_Ethernet

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
        self.dma_inst = DMA_Ethernet()

    def config(self, oFrmNum: int = 10000):

        self.oFrmNum = oFrmNum
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.OFAME_NUM_REG, oFrmNum
        )

        print("----------------------------------")
        print("----------PAICORE CONFIG----------")
        self.dma_inst.send_frame(self.configFrames)
        print("----------------------------------")

    def paicore_init(self, initFrames):
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 4)
        self.dma_inst.send_frame(initFrames)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 0)

    def inference(self, initFrames, inputFrames):
        self.paicore_init(initFrames) # may be the bottleneck
        # inputFrames = np.concatenate((initFrames, inputFrames))

        self.dma_inst.send_frame(inputFrames)
        return self.dma_inst.recv_frame(self.oFrmNum)
