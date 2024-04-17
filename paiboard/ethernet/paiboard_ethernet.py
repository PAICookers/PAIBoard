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

        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CPU2FIFO_CNT, 0)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.FIFO2SNN_CNT, 0)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.SNN2FIFO_CNT, 0)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.FIFO2CPU_CNT, 0)

        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.DP_RSTN, 0)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.DP_RSTN, 1)

        # TODO : Need to check the value of oen and channel_mask
        # self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.OEN, 0)
        # self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CHANNEL_MASK, 0)

    def config(self, oFrmNum: int = 10000):

        self.oFrmNum = oFrmNum
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.OFAME_NUM_REG, oFrmNum
        )
        # print(len(self.configFrames))
        print("----------------------------------")
        print("----------PAICORE CONFIG----------")
        self.dma_inst.send_frame(self.configFrames)
        print("----------------------------------")


    def paicore_status(self):
        cpu2fifo_cnt = self.dma_inst.read_reg(self.dma_inst.CPU2FIFO_CNT)
        fifo2snn_cnt = self.dma_inst.read_reg(self.dma_inst.FIFO2SNN_CNT)
        snn2fifo_cnt = self.dma_inst.read_reg(self.dma_inst.SNN2FIFO_CNT)
        fifo2cpu_cnt = self.dma_inst.read_reg(self.dma_inst.FIFO2CPU_CNT)
        us_time_tick = self.dma_inst.read_reg(self.dma_inst.US_TIME_TICK)

        print("cpu2fifo_cnt = " + str(cpu2fifo_cnt))
        print("fifo2snn_cnt = " + str(fifo2snn_cnt))
        print("snn2fifo_cnt = " + str(snn2fifo_cnt))
        print("fifo2cpu_cnt = " + str(fifo2cpu_cnt))
        print("us_time_tick = " + str(us_time_tick))
        

    def paicore_init(self, initFrames):
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 4)
        self.dma_inst.send_frame(initFrames)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 0)

    def inference(self, initFrames, inputFrames):
        self.paicore_init(initFrames) # may be the bottleneck
        # inputFrames = np.concatenate((initFrames, inputFrames))

        self.dma_inst.send_frame(inputFrames)
        return self.dma_inst.recv_frame(self.oFrmNum)
