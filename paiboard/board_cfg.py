# oen for PAICORE

# oen = 0 means send_channel
# oen = 1 means receive_channel

# 4bit oen for FPGA

# oen = 1 means send_channel
# oen = 0 means receive_channel

# U3C2 U3C6 U2C2 U2C6

# BONDING Version
# W2 W6 E2 E6

# FLIP CHIP Version
# E2 E6 W2 W6


"""
Board configuration.
FIXME Need test
"""


from dataclasses import dataclass
from enum import Enum, unique
from typing import Literal

from .exceptions import BoardCfgError

SUPPORTED_BOARDS = ["FLIP8", "BONDING003", "BONDING004", "BONDING008", "BONDING8"]


@unique
class ChipSOMType(Enum):
    """The type of system-on-module of the chip."""

    SINGLE_BONDING = "single bonding"
    ARRAY_2X2 = "array 2x2"


@dataclass(frozen=True)
class BoardCfg:
    name: str
    chip_som_type: ChipSOMType
    n_max_channel: int
    global_signal_delay: int
    oen: int
    channel_mask: int


def get_board_cfg(
    board_name: Literal[
        "FLIP8", "BONDING003", "BONDING004", "BONDING008", "BONDING8"
    ] = "FLIP8",
    n_max_channel: int = 4,
):
    """Get the board configuration by given board name & the maximum number of channels."""
    if board_name not in SUPPORTED_BOARDS:
        raise BoardCfgError(
            f"board name '{board_name}' is not supported. Supported boards are: {SUPPORTED_BOARDS}"
        )

    if board_name == "FLIP8":
        if n_max_channel == 16:  # not implemented
            raise NotImplementedError
            # oen = "0" + "1" * 15
        elif n_max_channel == 4:
            oen = 0b0111
        else:
            raise ValueError(
                f"'n_max_channel' must be 4 or 16, but got {n_max_channel}"
            )
    elif board_name == "BONDING003":
        oen = 0b1000
    elif board_name == "BONDING004":
        oen = 0b1100
    elif board_name == "BONDING008":
        oen = 0b1110
    elif board_name == "BONDING8":
        oen = 0b1110

    if oen <= 0:
        raise BoardCfgError(f"invalid oen: {oen}")

    channel_mask = -1
    for i in range(n_max_channel):
        if (oen >> i) & 1:
            channel_mask = 1 << (n_max_channel - i - 1)
            break

    if channel_mask < 0:
        raise BoardCfgError(f"invalid oen: {oen}, channel_mask: {channel_mask}")

    print(f"using board : {board_name}")
    print(f"oen         : {oen}")
    print(f"channel_mask: {channel_mask}")

    global_signal_delay = 92
    return BoardCfg(
        board_name,
        ChipSOMType.SINGLE_BONDING,
        n_max_channel,
        global_signal_delay,
        oen,
        channel_mask,
    )

    # bonding
    # W6 W2 E6 E2

    # FLIP CHIP
    # E6 E2 W6 W2

    # 004 E2 bit error

    # 004 W2 W6 E6
