import ast
from typing import TypedDict

from paicorelib import OffCoreCfg, OnCoreCfg
from paicorelib.coordinate import ChipCoord, Coord, CoreType
from paicorelib.coordinate import ReplicationId as RId
from paicorelib.framelib.frame_defs import (
    OfflineWorkFrame1Format as Off_WF1F,
)
from paicorelib.framelib.frame_defs import (
    OnlineWorkFrame1Format_1 as On_WF1F_1,
)
from paicorelib.framelib.types import FrameArrayType
from paicorelib.routing_defs import get_multicast_cores

CoordStr = str
ChipCoordStr = CoordStr
NodeName = str
SrcCoordStr = CoordStr


def coordstr2coord(coord_str: CoordStr) -> Coord:
    """Convert a coordinate string to a tuple. The coordinate string may be a tuple or a string.

    NOTE: In older version, the coordinate string is "1" instead of "(0,1)".
    """
    try:
        c = ast.literal_eval(coord_str)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"invalid coordinate string: {coord_str}") from e

    if isinstance(c, tuple):
        return Coord(*c)

    return Coord.from_addr(c)


class NeuSegAddrAttrs(TypedDict):
    n_neuron: int
    """The number of neurons in the segment."""
    addr_offset: int
    """The address offset at which the segment starts in the neuron address space."""
    interval: int
    """The number of times the neuron in the segment is repeated."""
    idx_offset: int
    """The offset of the starting address of this neuron corresponding to the neuron node   \
        in which it is located."""


NeuPhyLocMap = dict[ChipCoordStr, dict[CoordStr, NeuSegAddrAttrs]]
CoreNeuSegLocMap = dict[Coord, NeuSegAddrAttrs]

# Type definitions of node attributes


class CoordAttrs(TypedDict):
    addr_chip_x: int
    addr_chip_y: int
    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    tick_relative: list[int]
    addr_axon: list[int]


class InputNodeAttrs(CoordAttrs):
    lcn: int


class OutputDestAttrs(CoordAttrs):
    pass


InputNodeAttrsMap = dict[NodeName, InputNodeAttrs]
OutputDestAttrsMap = dict[NodeName, dict[SrcCoordStr, OutputDestAttrs]]


def get_node_type_from_attrs(node_attrs: CoordAttrs) -> CoreType:
    """Extract whether the node is online core from its attributes."""
    _, target_coord, _ = attrs2coord(node_attrs)
    return target_coord.core_type


def is_node_online(node_attrs: CoordAttrs) -> bool:
    return get_node_type_from_attrs(node_attrs) == CoreType.ONLINE


def attrs2coord(attrs: CoordAttrs) -> tuple[ChipCoord, Coord, RId]:
    return (
        ChipCoord(attrs["addr_chip_x"], attrs["addr_chip_y"]),
        Coord(attrs["addr_core_x"], attrs["addr_core_y"]),
        RId(attrs["addr_core_x_ex"], attrs["addr_core_y_ex"]),
    )


class InputNodeRTCfg:
    """Runtime configurations for the input node."""

    def __init__(self, attrs: InputNodeAttrs, template: FrameArrayType) -> None:
        self.chip_coord, self.core_coord, self.rid = attrs2coord(attrs)
        self.lcn = attrs.get("lcn", 0)
        self.template = template
        """The template for encoding the input frames."""

        mcast_coords = get_multicast_cores(self.core_coord, self.rid)
        if not all(c.core_type == self.dest_type for c in mcast_coords):
            raise ValueError(
                "all destination cores of the input node must be of the same type"
            )

    @property
    def size(self) -> int:
        return self.template.size

    @property
    def dest_type(self) -> CoreType:
        return self.core_coord.core_type

    @property
    def is_online(self) -> bool:
        return self.dest_type == CoreType.ONLINE


class OutputDestRTCfg:
    """Runtime configurations for the output destination. Expected output shape is (timestep, actual_size).
    Since the output axon addresses of neurons have a limit, larger output sizes will be described using
    ('timestep', axon address). This 'timestep' is actually the fan-in extension factor (len_ex_factor) but
    used to extend the output size.
    """

    def __init__(
        self,
        source_coords: list[Coord],
        attrs: OutputDestAttrs,
        timestep: int,
        len_ex_factor: int,
        template: FrameArrayType,
    ) -> None:
        self.chip_coord, self.core_coord, self.rid = attrs2coord(attrs)
        assert self.rid == RId(0, 0)
        self.src_coords = source_coords
        self.timestep = timestep
        """The timestep of the output node"""
        self.len_ex_factor = len_ex_factor
        """The length extension factor of the output node"""
        self.template = template
        """The template for decoding the output frames from the chip(s)."""

        if self.is_online:
            self.tpl_ts = (
                self.template >> On_WF1F_1.TIMESLOT_OFFSET
            ) & On_WF1F_1.TIMESLOT_MASK
            self.tpl_ax = (self.template >> On_WF1F_1.AXON_OFFSET) & On_WF1F_1.AXON_MASK
        else:
            self.tpl_ts = (
                self.template >> Off_WF1F.TIMESLOT_OFFSET
            ) & Off_WF1F.TIMESLOT_MASK
            self.tpl_ax = (self.template >> Off_WF1F.AXON_OFFSET) & Off_WF1F.AXON_MASK

        self.tpl_pairs = list(zip(self.tpl_ts, self.tpl_ax))

    @property
    def size(self) -> int:
        return self.template.size

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.timestep, self.size // self.timestep)

    @property
    def src_type(self) -> CoreType:
        if not all(
            c.core_type == self.src_coords[0].core_type for c in self.src_coords
        ):
            raise ValueError(
                "all source cores of the output node must be of the same type"
            )

        return self.src_coords[0].core_type

    @property
    def is_online(self) -> bool:
        return self.src_type == CoreType.ONLINE


InputNodeRTCfgMap = dict[NodeName, InputNodeRTCfg]
OutputDestRTCfgMap = dict[NodeName, OutputDestRTCfg]


def get_n_timeslot_max(is_online: bool) -> int:
    return OnCoreCfg.N_TIMESLOT_MAX if is_online else OffCoreCfg.N_TIMESLOT_MAX
