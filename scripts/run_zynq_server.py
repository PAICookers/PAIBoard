import click

try:
    from paiboard.zynq import ZynqServer
except ImportError:
    raise RuntimeError("This script can only be run on zynq platform") from None

DEFAULT_PORT = 8080


@click.command()
@click.option(
    "-p",
    "--port",
    default=DEFAULT_PORT,
    type=int,
    help=f"Port number (default: {DEFAULT_PORT})",
)
@click.option(
    "--bitfile", "-b", type=str, help="Path to the file of bitstream", required=True
)
@click.option("--debug", is_flag=True, help="Start zynq debug bridge")
def main(port: int, bitfile: str, debug: bool) -> None:
    with ZynqServer("0.0.0.0", port, bitfile, start_debug_bridge=debug) as server:
        server.run()


if __name__ == "__main__":
    main()
