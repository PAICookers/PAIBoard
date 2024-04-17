import numpy as np
import os

from paiboard.base import PAIBoard
from paiboard.simulator.hwSimulator.hwSimulator import *

class PAIBoard_SIM(PAIBoard):

    def __init__(
        self,
        baseDir: str,
        timestep: int,
        layer_num: int = 0,
        output_delay: int = 0,
        batch_size: int = 1,
    ):
        super().__init__(baseDir, timestep, layer_num, output_delay, batch_size)
    
    def config(self, oFrmNum: int = 10000, TimestepVerbose: bool = False):
        configPath = os.path.join(self.baseDir, "config_cores_all.txt")
        self.simulator = setOnChipNetwork(configPath, TimestepVerbose)

    def paicore_status(self):
        print("PAIBoard_SIM Not implemented Status")
        
    def inference(self, initFrames, inputFrames):
        workFrames = np.concatenate((initFrames, inputFrames))
        outputFrames = runSimulator(self.simulator, workFrames)
        return outputFrames
