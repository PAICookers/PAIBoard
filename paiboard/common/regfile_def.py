from dataclasses import dataclass
import numpy as np


REGFILE_DTYPE = np.uint32


@dataclass(frozen=True)
class RegFile:
    """Please refer to the hardware regfile definitions for the address offset of each register.

    In XDMA implementation, the regfile is connected with M_AXI_BYPASS channel.
    """

    N_BYTE = np.dtype(REGFILE_DTYPE).itemsize
    # The offset of each register
    RX_STATE = 0
    TX_STATE = 1 * N_BYTE
    CPU2FIFO_CNT = 2 * N_BYTE
    FIFO2SNN_CNT = 3 * N_BYTE
    SNN2FIFO_CNT = 4 * N_BYTE
    FIFO2CPU_CNT = 5 * N_BYTE
    WDATA_1 = 6 * N_BYTE
    WDATA_2 = 7 * N_BYTE
    RDATA_1 = 8 * N_BYTE
    RDATA_2 = 9 * N_BYTE
    DATA_CNT = 10 * N_BYTE
    TLAST_CNT = 11 * N_BYTE
    US_TIME_TICK = 12 * N_BYTE
    SEND_LEN = 20 * N_BYTE
    CHIP_TOP_CTRL = 21 * N_BYTE
    OFAME_NUM_REG = 22 * N_BYTE
    DATAPATH_RSTN = 23 * N_BYTE
    SINGLE_CHANNEL = 24 * N_BYTE
    CHANNEL_MASK = 25 * N_BYTE
    OEN = 26 * N_BYTE

    # Bit of CHIP_TOP_CTRL
    CHIP_TOP_CLEAR_BIT = 0
    CHIP_TOP_SYNC_BIT = 1
    CHIP_TOP_INIT_BIT = 2
