import pytest

from paiboard.common.uart_cfg import ChipUartCfg, get_available_serial_ports


class TestChipUartCfg:
    def test_CLK_FREQ_PLL_PARAMS_table(self):
        CLK_FREQ_PLL_PARAMS_OLD = {
            22.5: 0x383CE,
            24: 0x3838E,
            48: 0x3C1CF,
            72: 0x3810E,
            96: 0x3C0CF,
            120: 0x3808E,
            144: 0x44091,
            168: 0x50094,
            192: 0x3C04F,
            216: 0x44051,
            240: 0x4C053,
            264: 0x54055,
            288: 0x5C057,
            312: 0x64059,
            336: 0x6C05B,
            360: 0x3800E,
            384: 0x3C00F,
            408: 0x40010,
            432: 0x44011,
            456: 0x48012,
            480: 0x4C013,
            504: 0x50014,
            528: 0x54015,
            552: 0x58016,
            576: 0x5C017,
            600: 0x60018,
        }
        for clk, v in CLK_FREQ_PLL_PARAMS_OLD.items():
            clkf, clkr, clkod, bwadj = ChipUartCfg.CLK_FREQ_PLL_PARAMS[clk]

            assert clkf == (v >> 14) & 0x3F
            assert clkr == (v >> 10) & 0xF
            assert clkod == (v >> 6) & 0xF
            assert bwadj == v & 0x3F

    @pytest.mark.parametrize("clk_freq", [192, 240, 288, 312, 360])
    def test_get_uart_cmd(self, clk_freq):
        chip_coord = (0, 0)
        uart_cmd = ChipUartCfg.gen_uart_cmd(chip_coord, clk_freq=clk_freq)
        print([f"{b:02x}" for b in uart_cmd])

    @pytest.mark.skipif(
        len(get_available_serial_ports()) == 0, reason="No serial port available"
    )
    @pytest.mark.parametrize("clk_freq", [192, 240, 288, 312, 360])
    def test_serial_config(self, clk_freq):
        ChipUartCfg.serial_config(clk_freq=clk_freq)
