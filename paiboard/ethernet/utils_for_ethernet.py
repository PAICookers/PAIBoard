import numpy as np
from socket import *


def npFrameSplit(inputFrame, buffer_num):
    new_col = int((inputFrame.size - 1) / buffer_num)
    new_shape = (new_col, buffer_num)
    first = inputFrame[0 : buffer_num * new_col].reshape(new_shape)
    second = inputFrame[buffer_num * new_col :]
    all_one = np.array([[18446744073709551615] * buffer_num], dtype=np.uint64)
    all_one[0, 0 : second.size] = second
    split_frame = np.concatenate((first, all_one))
    return split_frame


def Ethernet_config(addr):
    tcpCliSock = socket(AF_INET, SOCK_STREAM)
    tcpCliSock.connect(addr)
    return tcpCliSock


def Ethernet_send(tcpCliSock, mode, send_frame, buffer_num):
    if mode == "SEND":
        header_frame = np.array([0], dtype=np.uint64)
    elif mode == "RECV":
        header_frame = np.array([1] * buffer_num, dtype=np.uint64)
        send_buffer = header_frame.tobytes()
        tcpCliSock.sendall(send_buffer)
        return
    elif mode == "WRITE REG":
        header_frame = np.array([2], dtype=np.uint64)
    elif mode == "READ REG":
        header_frame = np.array([3], dtype=np.uint64)
    elif mode == "CHIP RST":
        header_frame = np.array([4] * buffer_num, dtype=np.uint64)
        send_buffer = header_frame.tobytes()
        tcpCliSock.sendall(send_buffer)
        return
    elif mode == "CHIP UART":
        header_frame = np.array([5], dtype=np.uint64)
    elif mode == "QUIT":
        header_frame = np.array([99] * buffer_num, dtype=np.uint64)
        send_buffer = header_frame.tobytes()
        tcpCliSock.sendall(send_buffer)
        return
    else:
        print("ERROR")
        raise ValueError
    
    num_frame = np.array([send_frame.size], dtype=np.uint64)
    send_frame = np.concatenate((header_frame, num_frame, send_frame))
    send_frame = npFrameSplit(
        send_frame, buffer_num
    )  # split and add 0xFFFFFFFFFFFFFFFF

    for i in range(send_frame.shape[0]):
        send_buffer = send_frame[i].tobytes()
        rc = tcpCliSock.send(send_buffer)

def Ethernet_recv(tcpCliSock, send_frame, buffer_num, oFrmNum, read_reg=False, reg_addr=None):
    if read_reg:
        Ethernet_send(tcpCliSock, "READ REG", reg_addr, buffer_num)
    else:
        header_frame = np.array([1], dtype=np.uint64)
        num_frame = np.array([send_frame.size], dtype=np.uint64)
        send_frame = np.concatenate((header_frame, num_frame, send_frame))
        send_frame = npFrameSplit(
            send_frame, buffer_num
        )  # split and add 0xFFFFFFFFFFFFFFFF

        for i in range(send_frame.shape[0]):
            send_buffer = send_frame[i].tobytes()
            rc = tcpCliSock.send(send_buffer)

    # TODO: too much frame
    recv_buffer = tcpCliSock.recv(buffer_num << 3)
    outputFrames = np.frombuffer(recv_buffer, dtype="uint64")
    if oFrmNum <= buffer_num:
        pass
    else:
        while(outputFrames.size < oFrmNum):
            recv_buffer = tcpCliSock.recv(buffer_num << 3)
            recv_data = np.frombuffer(recv_buffer, dtype='uint64')
            outputFrames = np.concatenate((outputFrames, recv_data))
            if outputFrames.size % buffer_num != 0:
                recv_buffer = tcpCliSock.recv(buffer_num << 3)
                recv_data = np.frombuffer(recv_buffer, dtype='uint64')
                outputFrames = np.concatenate((outputFrames, recv_data))
    if read_reg:
        pass
    else:
        outputFrames = np.delete(outputFrames, np.where(outputFrames == 0))
        outputFrames = np.delete(outputFrames, np.where(outputFrames == 18446744073709551615))
        # outputFrames = np.unique(outputFrames)
    return outputFrames
