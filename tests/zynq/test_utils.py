from .conftest import FPGA_BITFILE_PATH


def test_default_fpge_bitfile_path_exists():
    bitfile = None
    for f in FPGA_BITFILE_PATH.glob("*.bit"):
        bitfile = f
        break

    assert bitfile is not None

    hwhfile = bitfile.with_suffix(".hwh")
    assert hwhfile.exists()
