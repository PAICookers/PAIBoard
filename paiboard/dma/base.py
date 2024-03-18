
class DMA_base(object):

    def __init__(self) -> None:

        self.RX_STATE        = 0  * 4
        self.TX_STATE        = 1  * 4
        self.CPU2FIFO_CNT    = 2  * 4
        self.FIFO2SNN_CNT    = 3  * 4
        self.SNN2FIFO_CNT    = 4  * 4
        self.FIFO2CPU_CNT    = 5  * 4
        self.WDATA_1         = 6  * 4
        self.WDATA_2         = 7  * 4
        self.RDATA_1         = 8  * 4
        self.RDATA_2         = 9  * 4
        self.DATA_CNT        = 10 * 4
        self.TLAST_CNT       = 11 * 4
        self.US_TIME_TICK    = 12 * 4

        self.SEND_LEN        = 20 * 4
        self.CTRL_REG        = 21 * 4
        self.OFAME_NUM_REG   = 22 * 4
        self.DP_RSTN         = 23 * 4
        self.SINGLE_CHANNEL  = 24 * 4
        self.CHANNEL_MASK    = 25 * 4
        self.OEN             = 26 * 4

    def read_reg(self, addr):
        raise NotImplementedError
    
    def write_reg(self, addr, data):
        raise NotImplementedError

    def send_frame(self, *args, **kwargs):
        raise NotImplementedError
    
    def recv_frame(self, *args, **kwargs):
        raise NotImplementedError

    def show_reg_status(self):
        regfile_base_addr = self.REGFILE_BASE
        print()
        print("RX_STATE      : %d" % self.read_reg(regfile_base_addr + self.RX_STATE      ))
        print("TX_STATE      : %d" % self.read_reg(regfile_base_addr + self.TX_STATE      ))
        print("CPU2FIFO_CNT  : %d" % self.read_reg(regfile_base_addr + self.CPU2FIFO_CNT  ))
        print("FIFO2SNN_CNT  : %d" % self.read_reg(regfile_base_addr + self.FIFO2SNN_CNT  ))
        print("SNN2FIFO_CNT  : %d" % self.read_reg(regfile_base_addr + self.SNN2FIFO_CNT  ))
        print("FIFO2CPU_CNT  : %d" % self.read_reg(regfile_base_addr + self.FIFO2CPU_CNT  ))
        print("SEND_LEN      : %d" % self.read_reg(regfile_base_addr + self.SEND_LEN      ))
        print("CTRL_REG      : %d" % self.read_reg(regfile_base_addr + self.CTRL_REG      ))
        val1 = self.read_reg(regfile_base_addr + self.WDATA_1)
        val2 = self.read_reg(regfile_base_addr + self.WDATA_2)
        print("WDATA         : 0x%016x" % (val2 << 32 | val1))
        val1 = self.read_reg(regfile_base_addr + self.RDATA_1)
        val2 = self.read_reg(regfile_base_addr + self.RDATA_2)
        print("RDATA         : 0x%016x" % (val2 << 32 | val1))
        print("DATA_CNT      : %d" % self.read_reg(regfile_base_addr + self.DATA_CNT      ))
        val = self.read_reg(regfile_base_addr + self.TLAST_CNT)
        print("TLAST_IN_CNT  : %d" % (val & 0x0000FFFF))
        print("TLAST_OUT_CNT : %d" % (val >> 16))
        print()

