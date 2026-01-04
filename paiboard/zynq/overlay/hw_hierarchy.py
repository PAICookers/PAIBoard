import time
from collections import UserList
from typing import ClassVar

import numpy as np
from paicorelib.framelib import FRAME_DTYPE, FrameArrayType
from pynq import MMIO, DefaultHierarchy, allocate  # pyright: ignore
from pynq.lib import AxiGPIO  # pyright: ignore

from paiboard.common import ChipUartCfg, RegFile
from paiboard.utils import wait_with_timeout

from .uartlite import UartAXI

N_BYTE_UART_DEBUG_EXTRA_RETURN = 19


class AXIUARTList(UserList[UartAXI]):
    pass


class HwDatapathHier(DefaultHierarchy):
    """This hierarchy packaged IP `pl_datapth` of the hardware design."""

    UART_IP_NAMES: ClassVar[list[str]] = [f"axi_uartlite_{i}" for i in range(4)]
    DMA_IP_NAME: ClassVar[str] = "axi_dma_0"
    GPIO_IP_NAME_CHIP_RST: ClassVar[str] = "axi_gpio_0"
    DEBUG_BRIDGE_IP_NAME: ClassVar[str] = "debug_bridge_0"

    REGFILE_BASE_ADDR: ClassVar[int] = 0x4001_0000
    REGFILE_ADDR_RANGE: ClassVar[int] = 0x10000  # 64K

    def __init__(self, description) -> None:
        super().__init__(description)
        # Regfile MMIO
        self.regfile_mmio = MMIO(self.REGFILE_BASE_ADDR, self.REGFILE_ADDR_RANGE)
        # DMA
        self.dma = getattr(self, self.DMA_IP_NAME)

        # GPIO chip reset
        axi_gpio_chip_rst_info = description["ip"][self.GPIO_IP_NAME_CHIP_RST]
        self.axi_gpio_chip_rst = AxiGPIO(axi_gpio_chip_rst_info).channel1

        # 4 UARTs
        self.axi_uarts = AXIUARTList()
        for uname in self.UART_IP_NAMES:
            uart_info = description["ip"][uname]
            base_addr, addr_range = uart_info["phys_addr"], uart_info["addr_range"]
            self.axi_uarts.append(UartAXI(base_addr, addr_range))

        for u in self.axi_uarts:
            u.setup_ctrl_reg()

        # If a debug bridge is implemented. Otherwise, comment out the following lines
        # bridge_desc = description["ip"][self.DEBUG_BRIDGE_IP_NAME]
        # self.db = DebugBridge(bridge_desc)

    def write_axi_gpio_chip_reset(self, value: int) -> None:
        # High reset, 4bit
        mask = 0xFFFF_FFFF
        self.axi_gpio_chip_rst.write(value, mask)
        time.sleep(0.02)
        self.axi_gpio_chip_rst.write(0, mask)

    def write_regfile(self, addr: int, value: int) -> None:
        self.regfile_mmio.write(addr, value)

    def read_regfile(self, addr: int, length: int = 4) -> int:  # uint32
        return self.regfile_mmio.read(addr, length)

    def write_dma(self, data: FrameArrayType) -> int:
        arr = np.ascontiguousarray(data)
        with allocate((arr.size,), dtype=FRAME_DTYPE) as buf:
            np.copyto(buf, arr)
            # Sync the buffer to ensure the updated data is visible to the PL
            buf.sync_to_device()

            self.write_regfile(RegFile.SEND_LEN, arr.size)  # #N of frames
            self.dma.sendchannel.transfer(buf)
            print("waiting DMA done")
            self.dma.sendchannel.wait()

            print("waiting TX set to 1")
            with wait_with_timeout(
                2, "Timeout waiting for TX_STATE to become 1"
            ) as timed_out:
                while self.read_regfile(RegFile.TX_STATE) == 0:
                    if timed_out():
                        break
                    time.sleep(0.0001)  # 100us

            self.write_regfile(RegFile.TX_STATE, 0)
            print("done")

        return arr.nbytes

    def read_dma(self, size: int) -> FrameArrayType:
        assert size > 0
        with allocate((size,), dtype=FRAME_DTYPE) as buf:
            self.write_regfile(RegFile.RX_STATE, 1)
            while True:
                rx_state = self.read_regfile(RegFile.RX_STATE)
                if rx_state != 1:
                    break
                time.sleep(0.0001)  # 100us

            self.write_regfile(RegFile.RX_STATE, 0)

            self.dma.recvchannel.transfer(buf)
            self.dma.recvchannel.wait()

            buf.sync_from_device()
            result = buf.copy()

        return result

    def send_uart_cmd(self, uart_idx: int, cmd: bytes) -> bytes:
        if uart_idx >= len(self.axi_uarts):
            raise ValueError(f"Invalid UART ID: {uart_idx}")

        u = self.axi_uarts[uart_idx]
        size = u.write_bytes(cmd)

        if ChipUartCfg.is_uart_debug_enable(cmd):
            # If debug is enabled, read the echo + extra 19 bytes
            recv = u.read_bytes(size + N_BYTE_UART_DEBUG_EXTRA_RETURN)
        else:
            recv = u.read_bytes(size)

        return recv

    def start_debug_bridge(self, port: int = 2542) -> None:
        self.db.start_xvc_server(
            bufferLen=4096,
            serverAddress="0.0.0.0",
            serverPort=port,
            reconnect=True,
            verbose=True,
        )

    def stop_debug_bridge(self) -> None:
        self.db.stop_xvc_server()

    @staticmethod
    def checkhierarchy(description) -> bool:
        required_ips = [
            HwDatapathHier.DMA_IP_NAME,
            HwDatapathHier.GPIO_IP_NAME_CHIP_RST,
            # HwDatapathHier.DEBUG_BRIDGE_IP_NAME,
        ] + HwDatapathHier.UART_IP_NAMES

        return all(ip in description["ip"] for ip in required_ips)
