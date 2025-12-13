from abc import ABC, abstractmethod

from paiboard.common.regfile_def import RegFile
from paiboard.global_cfg import DEFAULT_N_OUTPUT_FRAMES
from paiboard.types import FrameArrayType


class HostCtrlInterface(ABC):
    """General host control interface for every platform."""

    running: bool

    @abstractmethod
    def open(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def reset_regfile(self) -> None: ...

    @abstractmethod
    def reset_chip(self, *chip_idx: int) -> None: ...

    @abstractmethod
    def write_reg(self, addr: int, value: int) -> None: ...

    @abstractmethod
    def read_reg(self, addr: int) -> int: ...

    @abstractmethod
    def send_frames(self, frames: FrameArrayType, **kwargs) -> int: ...

    @abstractmethod
    def send_and_recv_frames(
        self, frames: FrameArrayType, recv_size: int | None = None, **kwargs
    ) -> FrameArrayType: ...

    def set_n_max_oframe(self, value: int | None = None) -> None:
        """Set the maximum number of output frames to be received from the chip.

        Args:
            n_max_oframe: maximum number of output frames, uint32.
        """
        if value is None:
            value = DEFAULT_N_OUTPUT_FRAMES

        assert value > 0
        self.n_max_oframe = value & 0xFFFF_FFFF
        self.write_reg(RegFile.OFAME_NUM_REG, self.n_max_oframe)

    def get_regfile_status(self) -> None:
        """Read and print the current status of the regfile registers in a formatted way."""
        wdata_1 = self.read_reg(RegFile.WDATA_1)
        wdata_2 = self.read_reg(RegFile.WDATA_2)
        rdata_1 = self.read_reg(RegFile.RDATA_1)
        rdata_2 = self.read_reg(RegFile.RDATA_2)
        tlast_cnt = self.read_reg(RegFile.TLAST_CNT)

        status = {
            "RX_STATE": self.read_reg(RegFile.RX_STATE),
            "TX_STATE": self.read_reg(RegFile.TX_STATE),
            "CPU2FIFO_CNT": self.read_reg(RegFile.CPU2FIFO_CNT),
            "FIFO2SNN_CNT": self.read_reg(RegFile.FIFO2SNN_CNT),
            "SNN2FIFO_CNT": self.read_reg(RegFile.SNN2FIFO_CNT),
            "FIFO2CPU_CNT": self.read_reg(RegFile.FIFO2CPU_CNT),
            "WDATA": wdata_2 << 32 | wdata_1,
            "RDATA": rdata_2 << 32 | rdata_1,
            "DATA_CNT": self.read_reg(RegFile.DATA_CNT),
            "TLAST_IN_CNT": tlast_cnt & 0xFFFF,
            "TLAST_OUT_CNT": tlast_cnt >> 16,
            "US_TIME_TICK": self.read_reg(RegFile.US_TIME_TICK),
            "SEND_LEN": self.read_reg(RegFile.SEND_LEN),
            "CHIP_TOP_CTRL": self.read_reg(RegFile.CHIP_TOP_CTRL),
            "OFAME_NUM_REG": self.read_reg(RegFile.OFAME_NUM_REG),
            "DATAPATH_RSTN": self.read_reg(RegFile.DATAPATH_RSTN),
            "SINGLE_CHANNEL": self.read_reg(RegFile.SINGLE_CHANNEL),
            "CHANNEL_MASK": self.read_reg(RegFile.CHANNEL_MASK),
            "OEN": self.read_reg(RegFile.OEN),
        }

        print("Regfile Status:")
        print("-" * 30)
        for name, v in status.items():
            print(f"{name:<20}: 0x{v:08X} ({v})")

        print("-" * 30)
