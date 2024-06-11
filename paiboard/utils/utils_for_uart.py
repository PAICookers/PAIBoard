import binascii, time


def hex2bin(hex_str):
    return bin(int(hex_str, 16))[2:]


def bin2hex(bin_str):
    return hex(int(bin_str, 2))[2:].upper()


def uart_hex_gen(
    core_info,
    clk_freq,
    source_chip,
    globalSignalDelay,
    globalSignalWidth,
    globalSignalBusyMask,
    Debug_en,
):
    clk_en = "FFFFFFFFFFFFFFFE"
    if clk_freq == 22.5:
        CLK_PARA = "383CE"
    elif clk_freq == 24:
        CLK_PARA = "3838E"
    elif clk_freq == 48:
        CLK_PARA = "3C1CF"
    elif clk_freq == 72:
        CLK_PARA = "3810E"
    elif clk_freq == 96:
        CLK_PARA = "3C0CF"
    elif clk_freq == 120:
        CLK_PARA = "3808E"
    elif clk_freq == 144:
        CLK_PARA = "44091"
    elif clk_freq == 168:
        CLK_PARA = "50094"
    elif clk_freq == 192:
        CLK_PARA = "3C04F"
    elif clk_freq == 216:
        CLK_PARA = "44051"
    elif clk_freq == 240:
        CLK_PARA = "4C053"
    elif clk_freq == 264:
        CLK_PARA = "54055"
    elif clk_freq == 288:
        CLK_PARA = "5C057"
    elif clk_freq == 312:
        CLK_PARA = "64059"
    elif clk_freq == 336:
        CLK_PARA = "6C05B"
    elif clk_freq == 360:
        CLK_PARA = "3800E"
    elif clk_freq == 384:
        CLK_PARA = "3C00F"
    elif clk_freq == 408:
        CLK_PARA = "40010"
    elif clk_freq == 432:
        CLK_PARA = "44011"
    elif clk_freq == 456:
        CLK_PARA = "48012"
    elif clk_freq == 480:
        CLK_PARA = "4C013"
    elif clk_freq == 504:
        CLK_PARA = "50014"
    elif clk_freq == 528:
        CLK_PARA = "54015"
    elif clk_freq == 552:
        CLK_PARA = "58016"
    elif clk_freq == 576:
        CLK_PARA = "5C017"
    elif clk_freq == 600:
        CLK_PARA = "60018"
    else:
        raise ValueError("Invalid clk_freq")

    chip_x = bin(source_chip[0])[2:].zfill(5)
    chip_y = bin(source_chip[1])[2:].zfill(5)

    delay_global_signal = bin(globalSignalDelay)[2:].zfill(10)
    width_global_signal = bin(globalSignalWidth)[2:].zfill(5)
    busy_mask_global_signal = bin(globalSignalBusyMask)[2:].zfill(10)
    Debug = bin(Debug_en)[2:].zfill(1)

    res_bin = (
        chip_x
        + chip_y
        + delay_global_signal
        + width_global_signal
        + busy_mask_global_signal
        + Debug
    )
    return clk_en + CLK_PARA + bin2hex(res_bin).zfill(9)


def serialConfig(clk_freq=312, globalSignalDelay=92, source_chip=(0, 0)):
    import serial

    ser = serial.Serial("/dev/ttyUSB0", 9600)
    if ser.isOpen():  # 判断串口是否成功打开
        print("[Info]  : Serial Open.")
    else:
        print("[Error] : Serial Not Open.")
        return 1

    uart_hex = uart_hex_gen(None, 312, source_chip, globalSignalDelay, 31, 100, 0)
    uart_bytes = bytes.fromhex(uart_hex)
    write_len = ser.write(uart_bytes)

    time.sleep(0.2)
    count = ser.inWaiting()

    data = None
    if count > 0:
        data = ser.read(count)
        if data != b"":
            dataStr = str(binascii.b2a_hex(data))[2:-1]
            print("receive:", dataStr)
        else:
            return 2
        # if dataStr != 'fffffffffffffffe640590005cf8c8':
        #     return 3
    if data == None:
        return 4

    ser.close()
    if ser.isOpen():
        print("[Error] : Serial Not Close.")
    else:
        print("[Info]  : Serial Close. Uart send Done!")

    return 0


if __name__ == "__main__":
    # serialConfig(92, (0, 0))
    uart_hex = uart_hex_gen(None, 192, (0,0), 92, 31, 100, 0)
    print(uart_hex)

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
