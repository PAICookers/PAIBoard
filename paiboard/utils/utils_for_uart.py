import binascii,time
import serial

def serialConfig(globalSignalDelay = 92):
    
    ser = serial.Serial("/dev/ttyUSB0", 9600)
    if ser.isOpen():                        # 判断串口是否成功打开
        print("[Info]  : Serial Open.")
    else:
        print("[Error] : Serial Not Open.")
        return 1

    # b = hex(globalSignalDelay)[2:]
    b = '{:02x}'.format(globalSignalDelay)
    uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 64 05 90 00 ' + b +' F8 C8')   # 312M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 6C 05 B0 00 ' + b +' F8 C8')   # 336M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 38 00 E0 00 ' + b +' F8 C8')   # 360M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 3C 00 F0 00 ' + b +' F8 C8')   # 384M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 40 00 10 00 ' + b +' F8 C8')   # 408M error
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 50 01 40 00 ' + b +' F8 C8')   # 504M
    # uarthex = bytes.fromhex('FF FF FF FF FF FF FF FE 60 01 80 00 ' + b +' F8 C8')   # 600M
    write_len=ser.write(uarthex)

    time.sleep(0.2)
    count = ser.inWaiting()

    data = None
    if count > 0:
        data=ser.read(count)
        if data!=b'':
            dataStr = str(binascii.b2a_hex(data))[2:-1]
            print("receive:",dataStr)
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