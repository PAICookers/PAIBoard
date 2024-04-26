
import socket
import time
import numpy as np
from pynq import Overlay
from pynq import MMIO
from pynq import Xlnk

LAST_FRAME = 18446744073709551615

# reg define
USER_REG = 0x40010000
ADDR_RANGE = 0x400
RX_STATE        = 0  * 4
TX_STATE        = 1  * 4
CPU2FIFO_CNT    = 2  * 4
FIFO2SNN_CNT    = 3  * 4
SNN2FIFO_CNT    = 4  * 4
FIFO2CPU_CNT    = 5  * 4
WDATA_1         = 6  * 4
WDATA_2         = 7  * 4
RDATA_1         = 8  * 4
RDATA_2         = 9  * 4
DATA_CNT        = 10 * 4
TLAST_CNT       = 11 * 4
US_TIME_TICK    = 12 * 4

SEND_LEN        = 20 * 4
CTRL_REG        = 21 * 4
OFAME_NUM_REG   = 22 * 4
DP_RSTN         = 23 * 4
SINGLE_CHANNEL  = 24 * 4
CHANNEL_MASK    = 25 * 4
OEN             = 26 * 4

def npFrameSplit(inputFrame, buffer_num):
    new_col = int((inputFrame.size - 1) / buffer_num)
    new_shape = (new_col, buffer_num)
    first = inputFrame[0 : buffer_num * new_col].reshape(new_shape)
    second = inputFrame[buffer_num * new_col :]
    all_one = np.array([[18446744073709551615] * buffer_num], dtype=np.uint64)
    all_one[0, 0 : second.size] = second
    split_frame = np.concatenate((first, all_one))
    return split_frame

def sendFrame(frames_bin):
    frames_num = frames_bin.size
    buffer = xlnk.cma_array(shape=(frames_num,), dtype=np.uint64)
    np.copyto(buffer,frames_bin)
    
    mmio.write(SEND_LEN, frames_num)
    dma_send.transfer(buffer)
    while True:
        tx_state = mmio.read(TX_STATE)
        if tx_state !=0:
            break
    mmio.write(TX_STATE, 0)

def recvFrame(oFrmNum):
    buffer = xlnk.cma_array(shape=(oFrmNum,), dtype=np.uint64)
    dma_recv.transfer(buffer)
    mmio.write(RX_STATE, 1)
    while True:
        rx_state = mmio.read(RX_STATE)
        if rx_state != 1:
            break
    mmio.write(RX_STATE, 0)
    dma_recv.wait()

    buffer = buffer[0:oFrmNum]

    return buffer

def status():

    rx_state     = mmio.read(RX_STATE)
    tx_state     = mmio.read(TX_STATE)
    cpu2fifo_cnt = mmio.read(CPU2FIFO_CNT)
    fifo2snn_cnt = mmio.read(FIFO2SNN_CNT)
    snn2fifo_cnt = mmio.read(SNN2FIFO_CNT)
    fifo2cpu_cnt = mmio.read(FIFO2CPU_CNT)
    us_time_tick = mmio.read(US_TIME_TICK)

    print("rx_state     = " + str(rx_state    ))
    print("tx_state     = " + str(tx_state    ))
    print("cpu2fifo_cnt = " + str(cpu2fifo_cnt))
    print("fifo2snn_cnt = " + str(fifo2snn_cnt))
    print("snn2fifo_cnt = " + str(snn2fifo_cnt))
    print("fifo2cpu_cnt = " + str(fifo2cpu_cnt))
    print("us_time_tick = " + str(us_time_tick))
    
    status_frame = np.array([rx_state,tx_state,cpu2fifo_cnt,fifo2snn_cnt,snn2fifo_cnt,fifo2cpu_cnt,us_time_tick], dtype=np.uint64)
    return status_frame

def read_reg(addr):
    reg_data = mmio.read(addr)
    reg_data_frame = np.array([reg_data], dtype=np.uint64)
    return reg_data_frame

def delete_last(np_array, delete_num):
    return np.delete(np_array, [i for i in range(np_array.size - delete_num, np_array.size)])

overlay = Overlay("./hw_file/design_2.bit")
dma_send = overlay.pl_datapath.axi_dma_0.sendchannel
dma_recv = overlay.pl_datapath.axi_dma_0.recvchannel

xlnk = Xlnk()
xlnk.xlnk_reset()

oFrmNum = 10000

mmio = MMIO(USER_REG, ADDR_RANGE)

mmio.write(CPU2FIFO_CNT, 0)
mmio.write(FIFO2SNN_CNT, 0)
mmio.write(SNN2FIFO_CNT, 0)
mmio.write(FIFO2CPU_CNT, 0)
mmio.write(OFAME_NUM_REG, oFrmNum)
mmio.write(OEN, 0x01)
mmio.write(CHANNEL_MASK, 0x01)

ip = '192.168.31.100'
port = 8889
addr = (ip, port)

buff_size = 4096 #消息的最大长度
buffer_num = int(buff_size / 8)  # for uint64

tcpSerSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcpSerSock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
tcpSerSock.bind(addr)
tcpSerSock.listen(2)

print('Wating...')
tcpCliSock, addr = tcpSerSock.accept()
print('Connected')

res = np.array([], dtype=np.uint64)

while True:

    header_get = False
    
    
    while True:
        if len(res) == 0:
            recv_buffer = tcpCliSock.recv(buff_size)
            while len(recv_buffer) == 0:
                recv_buffer = tcpCliSock.recv(buff_size)
            while len(recv_buffer)%buff_size != 0:
#                 print("Incompleted Package!")
                tmp_buffer = tcpCliSock.recv(buff_size)
                recv_buffer = recv_buffer + tmp_buffer
            recv_data = np.frombuffer(recv_buffer, dtype='uint64')
            if len(recv_data) != int(buff_size/8):
                res = recv_data[int(buff_size/8):]
                recv_data = recv_data[0:int(buff_size/8)]
                print("res",len(res))
        else:
            recv_data = res[0:int(buff_size/8)]
            res = res[int(buff_size/8):]
            print("res",len(res))
        if not header_get:
            all_num = 0
            if recv_data[0] == 0:
                work_mode = "SEND"
            elif recv_data[0] == 1:
                work_mode = "RECV"
            elif recv_data[0] == 2:
                work_mode = "WRITE REG"
                break
            elif recv_data[0] == 3:
                work_mode = "READ REG"
                break
            elif recv_data[0] == 4:
                work_mode = "QUIT"
                break

            send_num = recv_data[1]
            all_num += recv_data.size
            if(all_num >= send_num + 2):
                header_get = False
                recv_data = np.delete(recv_data, 0) # delete header
                recv_data = np.delete(recv_data, 0) # delete send_num
                recv_data = delete_last(recv_data, int(all_num - send_num - 2))
                sendFrame(recv_data)
                break
            else:
                header_get = True

            recv_data = np.delete(recv_data, 0) # delete header
            recv_data = np.delete(recv_data, 0) # delete send_num
            sendFrame(recv_data)
        else:
            all_num += recv_data.size
            if(all_num >= send_num + 2):
                recv_data = delete_last(recv_data, int(all_num - send_num - 2))
                sendFrame(recv_data)
                break
            else:
                sendFrame(recv_data)

    if work_mode == "SEND":
        pass
    elif work_mode == "RECV":
        recv_frame = recvFrame(oFrmNum)
        if len(recv_frame)== 0:
            print("No outputframe for this work.")
            recv_frame = np.array([18446744073709551615], dtype=np.uint64)
        # TODO: too much frame
        if oFrmNum > buffer_num:
            send_frame = npFrameSplit(recv_frame, buffer_num)  # split and add 0xFFFFFFFFFFFFFFFF
            for i in range(send_frame.shape[0]):
                send_buffer = send_frame[i].tobytes()
                rc = tcpCliSock.send(send_buffer)
        else:
            send_buffer = recv_frame.tobytes()
            tcpCliSock.sendall(send_buffer)
    elif work_mode == "WRITE REG":
        if(int(recv_data[2]) == OFAME_NUM_REG):
           oFrmNum = int(recv_data[3])
        mmio.write(int(recv_data[2]), int(recv_data[3]))
    elif work_mode == "READ REG":
        reg_data_frame = read_reg(int(recv_data[2]))
        send_buffer = reg_data_frame.tobytes()
        tcpCliSock.sendall(send_buffer)        
    elif work_mode == "QUIT":
        print('Wating...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('Connected')

status()

