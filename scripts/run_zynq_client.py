import time

import click

from paiboard.common import ChipUartCfg
from paiboard.zynq import ZynqClient

DEFAULT_IP = "192.168.31.99"
DEFAULT_PORT = 8080


@click.command()
@click.option(
    "--ip", default=DEFAULT_IP, type=str, help=f"IP address (default: {DEFAULT_IP})"
)
@click.option(
    "-p",
    "--port",
    default=DEFAULT_PORT,
    type=int,
    help=f"Port number (default: {DEFAULT_PORT})",
)
def main(ip: str, port: int) -> None:
    client = ZynqClient(ip, port)
    assert client.running
    client.get_regfile_status()

    client.reset_chip()
    # client.chip_uart_debug_en(0, True)

    cmd1 = ChipUartCfg.gen_uart_cmd((0, 0), clk_freq=240, debug=False)
    client.config_uart_chip(0, cmd1)

    client.reset_chip()
    cmd2 = ChipUartCfg.gen_uart_cmd((0, 0), clk_freq=240, debug=True)
    client.config_uart_chip(0, cmd2)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
