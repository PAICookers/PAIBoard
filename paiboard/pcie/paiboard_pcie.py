import numpy as np
import os

from paiboard.base import PAIBoard
from paiboard.pcie.dma_pcie import DMA_PCIe
from paiboard.pcie.global_hw_params import getBoard_data
from paiboard.utils.timeMeasure import time_calc_addText, get_original_function

from paiboard.utils.utils_for_uart import *

class PAIBoard_PCIe(PAIBoard):

    def __init__(
        self,
        baseDir: str,
        timestep: int,
        layer_num: int = 0,
        output_delay: int = 0,
        batch_size: int = 1,
    ):
        super().__init__(baseDir, timestep, layer_num, output_delay, batch_size)
        self.globalSignalDelay, self.oen, self.channel_mask = getBoard_data()
        self.dma_inst = DMA_PCIe(self.oen, self.channel_mask)

    def config(self, oFrmNum: int = 10000):
        print("")
        if serialConfig(globalSignalDelay=self.globalSignalDelay):
            print("[Error] : Uart can not send, Open and Reset PAICORE.")
            exit()

        self.oFrmNum = oFrmNum
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.OFAME_NUM_REG, oFrmNum
        )

        configPath = os.path.join(self.baseDir, "config_cores_all.bin")
        configFrames = np.fromfile(configPath, dtype="<u8")
        print("----------------------------------")
        print("----------PAICORE CONFIG----------")
        # SendFrameWrap(configFrames)
        self.dma_inst.send_config_frame(configFrames)
        print("----------------------------------")

    @time_calc_addText("Init          ")
    def paicore_init(self, initFrames):
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 4)
        self.dma_inst.send_frame(initFrames)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 0)

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

    def inference(self, initFrames, inputFrames):
        self.paicore_init(initFrames)
        self.dma_inst.send_frame(inputFrames, multi_channel_enable=True)
        # self.dma_inst.send_frame(inputFrames, multi_channel_enable=False)
        return self.dma_inst.recv_frame(self.oFrmNum)
