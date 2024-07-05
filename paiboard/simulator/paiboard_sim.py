import numpy as np
import os

from paiboard.base import PAIBoard
from paiboard.simulator.hwSimulator.hwSimulator import setOnChipNetwork, runSimulator


class PAIBoard_SIM(PAIBoard):

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

    def config(self, oFrmNum: int = 10000, TimestepVerbose: bool = False):
        configPath = os.path.join(self.baseDir, "config_all.bin")
        self.simulator = setOnChipNetwork(configPath, TimestepVerbose)

    def chip_init(self, chip_id_list):
        print("PAIBoard_SIM Not implemented Chip Init")

    def paicore_status(self):
        print("PAIBoard_SIM Not implemented Status")

    def inference(self, initFrames, inputFrames):
        workFrames = np.concatenate((initFrames, inputFrames))
        outputFrames = runSimulator(self.simulator, workFrames)
        return outputFrames
