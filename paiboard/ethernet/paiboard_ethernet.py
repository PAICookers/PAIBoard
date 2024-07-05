import numpy as np
import os

from paiboard.base import PAIBoard
from paiboard.ethernet.dma_ethernet import DMA_Ethernet
from paiboard.ethernet.global_hw_params import getBoard_data
from paiboard.utils.timeMeasure import time_calc_addText, get_original_function

from paiboard.utils.utils_for_uart import *


class PAIBoard_Ethernet(PAIBoard):

    def __init__(
        self,
        baseDir: str,
        timestep: int,
        layer_num: int = 0,
        output_delay: int = 0,
        batch_size: int = 1,
        backend: str = "PAIBox",
    ):
        super().__init__(
            baseDir, timestep, layer_num, output_delay, batch_size, backend
        )
        self.dma_inst = DMA_Ethernet()

        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.CPU2FIFO_CNT, 0
        )
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.FIFO2SNN_CNT, 0
        )
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.SNN2FIFO_CNT, 0
        )
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.FIFO2CPU_CNT, 0
        )

        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.DP_RSTN, 0)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.DP_RSTN, 1)

        # TODO : Need to check the value of oen and channel_mask
        # self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.OEN, 0)
        # self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CHANNEL_MASK, 0)

    def chip_init(self, chip_id_list):
        self.dma_inst.chip_rst()
        for index, source_chip in enumerate(chip_id_list):
            uart_hex = uart_hex_gen(clk_freq=240, source_chip=source_chip)
            uart_np = uart_np_gen(uart_hex, index)
            self.dma_inst.chip_uart(uart_np)

    def config(self, oFrmNum: int = 10000, send=True):

        self.oFrmNum = oFrmNum
        self.dma_inst.write_reg(
            self.dma_inst.REGFILE_BASE + self.dma_inst.OFAME_NUM_REG, oFrmNum
        )
        # print(len(self.configFrames))
        print("----------------------------------")
        print("----------PAICORE CONFIG----------")
        if send:
            self.dma_inst.send_config_frame(self.configFrames)
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

    @time_calc_addText("Init          ")
    def paicore_init(self, initFrames):
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 4)
        self.dma_inst.send_config_frame(initFrames)
        self.dma_inst.write_reg(self.dma_inst.REGFILE_BASE + self.dma_inst.CTRL_REG, 0)

    def inference(self, initFrames, inputFrames, init):
        if init:
            self.paicore_init(initFrames)  # may be the bottleneck
        # from paiboard.utils.utils_for_frame import frame_np2txt
        # frame_np2txt(inputFrames, "inputFrames.txt",frameSplit=False)
        return self.dma_inst.recv_frame(inputFrames, self.oFrmNum)
