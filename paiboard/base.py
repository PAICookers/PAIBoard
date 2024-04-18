import numpy as np
import os
import json

from paiboard.PAIBoxRuntime.PAIBoxRuntime import PAIBoxRuntime
from paiboard.utils.timeMeasure import *
from paiboard.utils.utils_for_frame import frame_np2txt

class PAIBoard(object):
    def __init__(
        self,
        baseDir: str,
        timestep: int,
        layer_num: int = 0,
        output_delay: int = 0,
        batch_size: int = 1,
    ):
        self.baseDir = baseDir
        assert timestep * batch_size <= 256 # batch inference limit
        self.timestep = timestep
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.output_delay = output_delay

        coreInfoPath = os.path.join(self.baseDir, "core_params.json")
        self.initFrames = PAIBoxRuntime.gen_init_frame(coreInfoPath)
        self.syncFrames = PAIBoxRuntime.gen_sync_frame(self.timestep + self.layer_num)

        inputInfoPath = os.path.join(self.baseDir, "input_proj_info.json")
        with open(inputInfoPath, "r", encoding="utf8") as fp:
            input_proj_info = json.load(fp)
        self.input_frames_info = PAIBoxRuntime.gen_input_frames_info(
            timestep=self.timestep * self.batch_size, input_proj_info=input_proj_info
        )

        outputInfoPath = os.path.join(self.baseDir, "output_dest_info.json")
        with open(outputInfoPath, "r", encoding="utf8") as fp:
            self.output_dest_info = json.load(fp)
        self.output_frames_info = PAIBoxRuntime.gen_output_frames_info(
            timestep=self.timestep * self.batch_size,
            delay=self.output_delay,
            output_dest_info=self.output_dest_info,
        )

        configPath = os.path.join(self.baseDir, "config_cores_all.bin")
        self.configFrames = np.fromfile(configPath, dtype="<u8")

    def config(self, *args, **kwargs):
        # dma init & send config frame
        raise NotImplementedError
    
    def inference(self, *args, **kwargs):
        # send work frame and recvive output frame
        raise NotImplementedError

    def genSpikeFrame(self, input_spike, ifi):
        if isinstance(input_spike, list):
            if self.batch_size > 1:
                # todo: batch_size for multiple input port
                for i in range(len(input_spike)):
                    ifi[i] = ifi[i][0 : input_spike[i].size]
                # todo what happen if two(or more) input's batch different?
                self.batch_ts = input_spike[0].shape[0]
                assert self.batch_ts == input_spike[i].shape[0]
                self.syncFrames = PAIBoxRuntime.gen_sync_frame(
                    self.batch_ts + self.layer_num
                )
            spikeFrame = np.array([], dtype=np.uint64)
            for i in range(len(input_spike)):
                sf = PAIBoxRuntime.encode(input_spike[i], ifi[i])
                spikeFrame = np.concatenate((spikeFrame, sf))
        else:
            if self.batch_size > 1:
                ifi = [ifi[0][0 : input_spike.size]]
                # 需要实时判断当前batch_size的长度，因为最后一个batch_size可能不满足batch_size
                # 但在生成input_frames_info时，需要用最大的batch_size来生成
                self.batch_ts = input_spike.shape[0]
                assert (
                    self.batch_ts <= self.timestep * self.batch_size
                    and self.batch_ts % self.timestep == 0
                )
                self.syncFrames = PAIBoxRuntime.gen_sync_frame(
                    self.batch_ts + self.layer_num
                )
            spikeFrame = PAIBoxRuntime.encode(input_spike, ifi[0])
        return spikeFrame

    def genOutputSpike(self, outputFrames, ofi, ts):
        if len(ofi) == 1 and self.batch_size > 1:
            ofi = [
                ofi[0]
                .reshape(-1, self.timestep * self.batch_size)[:, 0 : self.batch_ts]
                .flatten()
            ]

            ts = self.batch_ts
        elif len(ofi) > 1 and self.batch_size > 1:
            # todo
            for i in range(len(ofi)):
                ofi[i] = (
                    ofi[i]
                    .reshape(-1, self.timestep * self.batch_size)[:, 0 : self.batch_ts]
                    .flatten()
                )
            ts = self.batch_ts
        outputSpike = PAIBoxRuntime.decode_spike_less1152(
            timestep=ts,
            oframes=outputFrames,
            oframe_infos=ofi,
            flatten=False,
        )
        if len(ofi) == 1:
            outputSpike = outputSpike[0]
        elif len(ofi) > 1:
            outputSpike_dict = {}
            for index, output_proj_name in enumerate(self.output_dest_info):
                # print(f"output_proj_name: {output_proj_name}")
                outputSpike_dict[output_proj_name] = outputSpike[index]
            return outputSpike_dict
        return outputSpike

    def __call__(self, input_spike, TimeMeasure=False):
        t1 = time.time()
        spikeFrame = self.genSpikeFrame(input_spike, self.input_frames_info)
        t2 = time.time()
        time_dict["genSpikeFrame "] = (
            time_dict["genSpikeFrame "] + (t2 - t1) * 1000 * 1000
        )

        inputFrames = np.concatenate((spikeFrame, self.syncFrames))

        outputFrames = self.inference(
            self.initFrames,
            inputFrames
        )
        # frame_np2txt(outputFrames, self.baseDir + "/outputFrames.txt")

        t3 = time.time()
        outputSpike = self.genOutputSpike(
            outputFrames, self.output_frames_info, self.timestep
        )
        t4 = time.time()
        time_dict["genOutputSpike"] = (
            time_dict["genOutputSpike"] + (t4 - t3) * 1000 * 1000
        )
        # pred = np.argmax(spike_out)

        return outputSpike

    def record_time(self, full_time):
        # TODO : read register to get core_time
        core_time = self.dma_inst.read_reg(self.dma_inst.US_TIME_TICK)
        recore_time(core_time, full_time)

    def perf(self, img_num):
        print_time(img_num)