import pytest
from paiboard import PAIBoardPCIe
from paiboard.common.uart_cfg import get_available_serial_ports


class TestPAIBoardPCIe:
    @pytest.mark.skipif(
        len(get_available_serial_ports()) == 0, reason="No serial port available"
    )
    @pytest.mark.parametrize("clk_freq", [192, 240, 288, 312, 360])
    def test_chip_uart_config(self, toolchain_build_dir, clk_freq):
        pcie = PAIBoardPCIe(toolchain_build_dir, 1, 1, xdma_dev_idx=0, xdma_channel=0)
        # uart debug on
        pcie.chip_uart_config(port=None, clk_freq=clk_freq, debug=True)
        # uart debug off
        pcie.chip_uart_config(port=None, clk_freq=clk_freq, debug=False)
