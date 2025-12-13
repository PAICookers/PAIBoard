import numpy as np
import time
from typing import overload
from numpy.typing import NDArray
from paicorelib.coordinate import to_coord, CoordLike

try:
    import serial
    from serial.tools import list_ports

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

UART_CMD_DTYPE = np.uint8
UartCmdArrayType = NDArray[UART_CMD_DTYPE]


class ChipUartCfg:
    CLK_FREQ_PLL_BASE = 24  # MHz
    CLK_FREQ_PLL_PARAMS = {
        # From MSB to LSB:
        # CLKF(6) CLKR(4) CLKOD(4) BWADJ(6), BWADJ = CLKF
        22.5: (0xE, 0x0, 0xF, 0xE),
        24: (0xE, 0x0, 0xE, 0xE),
        48: (0xF, 0x0, 0x7, 0xF),
        72: (0xE, 0x0, 0x4, 0xE),
        96: (0xF, 0x0, 0x3, 0xF),
        120: (0xE, 0x0, 0x2, 0xE),
        144: (0x11, 0x0, 0x2, 0x11),
        168: (0x14, 0x0, 0x2, 0x14),
        192: (0xF, 0x0, 0x1, 0xF),
        216: (0x11, 0x0, 0x1, 0x11),
        240: (0x13, 0x0, 0x1, 0x13),
        264: (0x15, 0x0, 0x1, 0x15),
        288: (0x17, 0x0, 0x1, 0x17),
        312: (0x19, 0x0, 0x1, 0x19),
        336: (0x1B, 0x0, 0x1, 0x1B),
        360: (0xE, 0x0, 0x0, 0xE),
        384: (0xF, 0x0, 0x0, 0xF),
        408: (0x10, 0x0, 0x0, 0x10),
        432: (0x11, 0x0, 0x0, 0x11),
        456: (0x12, 0x0, 0x0, 0x12),
        480: (0x13, 0x0, 0x0, 0x13),
        504: (0x14, 0x0, 0x0, 0x14),
        528: (0x15, 0x0, 0x0, 0x15),
        552: (0x16, 0x0, 0x0, 0x16),
        576: (0x17, 0x0, 0x0, 0x17),
        600: (0x18, 0x0, 0x0, 0x18),
    }

    CLK_EN_L2_ALL_ON = [0xFF] * 8
    CLK_EN_L2_ONLY_ONLINE_DISABLED = [0xFF] * 7 + [0xFE]
    CLK_EN_L2_DEFAULT = CLK_EN_L2_ONLY_ONLINE_DISABLED

    @overload
    @classmethod
    def gen_uart_cmd(
        cls,
        chip_coord: CoordLike,
        clk_freq: int = 240,
        *,
        clk_en_L2: list[int] | None = None,
        debug: bool = False,
        global_signal_delay: int = 0,
        global_signal_width: int = 31,
        global_signal_busy_mask: int = 100,
    ) -> bytes: ...

    @overload
    @classmethod
    def gen_uart_cmd(
        cls,
        chip_coord: CoordLike,
        *,
        clk_en_L2: list[int] | None = None,
        debug: bool = False,
        global_signal_delay: int = 0,
        global_signal_width: int = 31,
        global_signal_busy_mask: int = 100,
        PLL_NF: int,
        PLL_NR: int,
        PLL_OD: int,
    ) -> bytes: ...

    @classmethod
    def gen_uart_cmd(
        cls,
        chip_coord: CoordLike,
        clk_freq: int | None = 240,
        *,
        clk_en_L2: list[int] | None = None,
        debug: bool = False,
        global_signal_delay: int = 0,
        global_signal_width: int = 31,
        global_signal_busy_mask: int = 100,
        PLL_NF: int | None = None,
        PLL_NR: int | None = None,
        PLL_OD: int | None = None,
    ) -> bytes:
        """Generate UART command.

        Args:
            chip_coord (CoordLike): The chip coordinate.
            clk_freq (Optional, int): The clock frequency in MHz. Defaults to 240.
            clk_en_L2 (list[int]): the opened clocks of L2 clusters of the chip.
            debug (bool): debug mode. Defaults to false.
            global_signal_delay (int): the global signal delay. Defaults to 0.
            global_signal_width (int): the global signal width. Defaults to 31.
            global_signal_busy_mask (int): the mask of busy global signal. Defaults to 100.
            PLL_NF/PLL_NR/PLL_OD (Optional, int): specify the parameters of PLL to set the clock frequency  \
                if `clk_freq` is not provided.
        """
        # From MSB to LSB:
        # CLKF(6) CLKR(4) CLKOD(4) BWADJ(6)
        # F = 24MHz * (CLKF+1) / (CLKR+1) / (CLK0D+1)
        # BWADJ = CLKF
        if clk_freq is None:
            assert PLL_NF is not None and 1 <= PLL_NF <= 64
            assert PLL_NR is not None and 1 <= PLL_NR <= 16
            assert PLL_OD is not None and 1 <= PLL_OD <= 16
            if (clk := cls.CLK_FREQ_PLL_BASE * PLL_NF / PLL_NR) < 360:
                raise ValueError(
                    f"PLL output frequency is too low, CLK*NF/NR = {clk} < 360MHz"
                )

            clkf = (PLL_NF - 1) & 0x3F
            clkr = (PLL_NR - 1) & 0xF
            clkod = (PLL_OD - 1) & 0xF
            bwadj = clkf
        elif clk_freq in cls.CLK_FREQ_PLL_PARAMS:
            clkf, clkr, clkod, bwadj = cls.CLK_FREQ_PLL_PARAMS[clk_freq]
        else:
            supported_clk_freq = ", ".join(str(clk) for clk in cls.CLK_FREQ_PLL_PARAMS)
            raise ValueError(
                f"{clk_freq} is not supported. Supported clock MHz: {supported_clk_freq}"
            )

        cfg_bytes_lst = [0] * 7
        # #9 byte[7:2] = CLKF, byte[1:0] = CLKR[3:2]
        cfg_bytes_lst[0] = ((clkf & 0x3F) << 2) | ((clkr & 0xC) >> 2)
        # #10 byte[7:6] = CLKR[1:0], [5:2] = CLKOD, [1:0] = BWADJ[5:4]
        cfg_bytes_lst[1] = (
            ((clkr & 0x3) << 6) | ((clkod & 0xF) << 2) | ((bwadj & 0x30) >> 4)
        )
        # #11 byte[7:4] = BWADJ[3:0], [3:0] = chip_x[4:1]
        chip_coord = to_coord(chip_coord)
        cfg_bytes_lst[2] = (bwadj & 0xF) << 4 | ((chip_coord.x & 0x1E) >> 1)
        # #12 byte[7] = chip_x[0], [6:2] = chip_y, [1:0] = delay_global_signal[9:8]
        cfg_bytes_lst[3] = (
            ((chip_coord.x & 0x1) << 7)
            | ((chip_coord.y & 0x1F) << 2)
            | ((global_signal_delay & 0x300) >> 8)
        )
        # #13 byte[7:0] = delay_global_signal[7:0]
        cfg_bytes_lst[4] = global_signal_delay & 0xFF
        # #14 byte[7:3] = width_global_signal, [2:0] = busy_mask_global_signal[9:7]
        cfg_bytes_lst[5] = ((global_signal_width & 0x1F) << 3) | (
            (global_signal_busy_mask & 0x380) >> 7
        )
        # #15 byte[7:1] = busy_mask_global_signal[6:0], [0] = debug
        cfg_bytes_lst[6] = ((global_signal_busy_mask & 0x7F) << 1) | (int(debug) & 0x1)

        # Generate command bytes (8+7 bytes)
        if clk_en_L2 is None:
            clk_en_L2 = cls.CLK_EN_L2_DEFAULT

        return bytes(clk_en_L2 + cfg_bytes_lst)

    @classmethod
    def is_uart_debug_enable(cls, uart_cmd: bytes) -> bool:
        # #15 byte[7:1] = busy_mask_global_signal[6:0], [0] = debug
        return bool(uart_cmd[-1] & 0x1)

    if HAS_SERIAL:

        @classmethod
        def serial_config(
            cls,
            port: str | None = "/dev/ttyUSB0",
            baudrate: int = 9600,
            chip_coord: CoordLike = (0, 0),
            clk_freq: int = 312,
            *,
            timeout: float | None = None,
            clk_en_L2: list[int] | None = None,
            debug: bool = False,
            global_signal_delay: int = 92,
            global_signal_width: int = 31,
            global_signal_busy_mask: int = 100,
        ) -> bytes:
            if port is None:
                ports = get_available_serial_ports()
                if len(ports) == 0:
                    raise OSError("no serial port found.")

                for p in ports:
                    try:
                        ser = _try_open_serial_port(p, baudrate, timeout)
                        break
                    except serial.SerialException:
                        continue
            else:
                ser = _try_open_serial_port(port, baudrate, timeout)

            uart_cmd = cls.gen_uart_cmd(
                chip_coord,
                clk_freq,
                clk_en_L2=clk_en_L2,
                debug=debug,
                global_signal_delay=global_signal_delay,
                global_signal_width=global_signal_width,
                global_signal_busy_mask=global_signal_busy_mask,
            )

            # Send command
            ser.write(uart_cmd)
            ser.flush()
            time.sleep(0.2)

            waiting = ser.in_waiting
            if waiting > 0:
                data = ser.read(waiting)
                print(f"received data: {data}, len {len(data)}")

            ser.close()
            if ser.is_open:
                raise OSError("serial port is not closed.")

            return data


if HAS_SERIAL:
    import sys

    def get_available_serial_ports() -> list[str]:
        """Get available serial ports."""
        if sys.platform == "win32":
            name = "COM"
        else:
            name = "ttyUSB"

        return [p.device for p in list_ports.comports() if name in p.device]

    def _try_open_serial_port(
        port: str, baudrate: int, timeout: float | None
    ) -> serial.Serial:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        if not ser.is_open:
            raise serial.PortNotOpenError()
        else:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"Connected to {ser.name}")

        return ser

    # serialConfig(92, (0, 0))
    # uart_hex = gen_uart_cmd(
    #     None,
    #     clk_freq=312,
    #     #在这里如果想要工具链方生成的是阵列板的config 可以设置为（1，0) 如果生成的是单板的config 需要设置为（0,0)
    #     #这是往单板里发的串口数据生成代码 要让PAICORE芯片知道自己是谁
    #     source_chip=(1, 0),
    #     global_signal_delay=92,
    #     global_signal_width=31,
    #     global_signal_busy_mask=100,
    #     Debug_en=0,
    # )
    # uart_hex = gen_uart_cmd(
    #     None,
    #     clk_freq=240,
    #     # 在这里如果想要工具链方生成的是阵列板的config 可以设置为（1，0) 如果生成的是单板的config 需要设置为（0,0)
    #     # 这是往单板里发的串口数据生成代码 要让PAICORE芯片知道自己是谁
    #     source_chip=(1, 0),
    #     global_signal_delay=0,
    #     global_signal_width=31,
    #     global_signal_busy_mask=100,
    #     Debug_en=0,
    # )
    # print(uart_hex)

    # uart_hex = "FFFFFFFFFFFFFFFE640590005CF8C8"
    # uart_hex_list = []
    # for i in range(0, len(uart_hex), 2):
    #     uart_hex_list.append(int("0x"+uart_hex[i:i + 2],16))
    # print(uart_hex_list)


# FFFFFFFFFFFFFFFE 64 05 9 000 5CF8C8


# 22.5:FFFFFFFFFFFFFFFE383CE0006450C9
# 24  :FFFFFFFFFFFFFFFE3838E0006450C9
# 48  :FFFFFFFFFFFFFFFE3C1CF0006450C9
# 72  :FFFFFFFFFFFFFFFE3810E0006450C9
# 96  :FFFFFFFFFFFFFFFE3C0CF0006450C9
# 120 :FFFFFFFFFFFFFFFE3808E0006450C9
# 144 :FFFFFFFFFFFFFFFE440910006450C9
# 168 :FFFFFFFFFFFFFFFE500940006450C9
# 192 :FFFFFFFFFFFFFFFE3C04F0006450C9
# 216 :FFFFFFFFFFFFFFFE440510006450C9
# 240 :FFFFFFFFFFFFFFFE4C0530006450C9
# 264 :FFFFFFFFFFFFFFFE540550006450C9
# 288 :FFFFFFFFFFFFFFFE5C0570006450C9
# 312 :FFFFFFFFFFFFFFFE640590006450C9
# 336 :FFFFFFFFFFFFFFFE6C05B0006450C9
# 360 :FFFFFFFFFFFFFFFE3800E0006450C9
# 384 :FFFFFFFFFFFFFFFE3C00F0006450C9
# 408 :FFFFFFFFFFFFFFFE400100006450C9
# 432 :FFFFFFFFFFFFFFFE440110006450C9
# 456 :FFFFFFFFFFFFFFFE480120006450C9
# 480 :FFFFFFFFFFFFFFFE4C0130006450C9
# 504 :FFFFFFFFFFFFFFFE500140006450C9
# 528 :FFFFFFFFFFFFFFFE540150006450C9
# 552 :FFFFFFFFFFFFFFFE580160006450C9
# 576 :FFFFFFFFFFFFFFFE5C0170006450C9
# 600 :FFFFFFFFFFFFFFFE600180006450C9
# 624 :FFFFFFFFFFFFFFFE640190006450C9
# 648 :FFFFFFFFFFFFFFFE6801A0006450C8
# 672 :FFFFFFFFFFFFFFFE6C01B0006450C8
# 696 :FFFFFFFFFFFFFFFE7001C0006450C8
# 720 :FFFFFFFFFFFFFFFE7401D0006450C8
# 744 :FFFFFFFFFFFFFFFE7801E0006450C8
# 768 :FFFFFFFFFFFFFFFE7C01F0006450C8
# 792 :FFFFFFFFFFFFFFFE800200006450C8
# 816 :FFFFFFFFFFFFFFFE840210006450C8
# 840 :FFFFFFFFFFFFFFFE880220006450C8
# 864 :FFFFFFFFFFFFFFFE8C0230006450C8
# 888 :FFFFFFFFFFFFFFFE900240006450C8
# 912 :FFFFFFFFFFFFFFFE940250006450C8
# 936 :FFFFFFFFFFFFFFFE980260006450C8
# 960 :FFFFFFFFFFFFFFFE9C0270006450C8
# 984 :FFFFFFFFFFFFFFFEA00280006450C8
# 1008:FFFFFFFFFFFFFFFEA40290006450C8
# 1032:FFFFFFFFFFFFFFFEA802A0006450C8
# 1056:FFFFFFFFFFFFFFFEAC02B0006450C8
# 1080:FFFFFFFFFFFFFFFEB002C0006450C8
# 1104:FFFFFFFFFFFFFFFEB402D0006450C8
# 1128:FFFFFFFFFFFFFFFEB802E0006450C8
# 1152:FFFFFFFFFFFFFFFEBC02F0006450C8
# 1176:FFFFFFFFFFFFFFFEC00300006450C8
# 1200:FFFFFFFFFFFFFFFEC40310006450C8
