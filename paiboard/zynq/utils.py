from pathlib import Path


def check_bitfile(path: Path | str) -> None:
    if not (bitfile := Path(path)).exists():
        raise FileNotFoundError(f"bitstream file not found: {bitfile}")

    hwhfile = bitfile.with_suffix(".hwh")
    if not hwhfile.exists():
        raise FileNotFoundError(f"hwh file not found: {hwhfile}")


def auto_search_bitfile(path: Path | str) -> Path:
    """Search FPGA bitstream & hwh file in the directory automatically, return the first one found."""
    p = Path(path).expanduser()

    if p.is_dir():
        bitfile = None
        for f in p.glob("*.bit"):
            bitfile = f
            break

        if bitfile is None:
            raise FileNotFoundError(f"bitstream file not found in directory: {p}")
    elif p.suffix == ".bit":
        if not p.exists():
            raise FileNotFoundError(f"bitstream file not found: {p}")
        bitfile = p
    else:
        raise ValueError(f"invalid path: {p}. Must be a directory or .bit file.")

    hwhfile = bitfile.with_suffix(".hwh")  # Same name as bitfile
    if not hwhfile.exists():
        raise FileNotFoundError(f"hwh file not found: {hwhfile}")

    return bitfile
