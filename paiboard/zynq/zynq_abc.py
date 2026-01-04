from abc import abstractmethod

from paiboard.common.ctrl_intf_abc import HostCtrlInterface


class ZynqCtrlInterface(HostCtrlInterface):
    """General host control interface on Zynq platform."""

    @abstractmethod
    def config_uart_chip(self, uart_idx: int, cmd: bytes) -> bytes: ...
