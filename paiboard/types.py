from collections.abc import Sequence
from enum import Enum, auto, unique

import numpy as np
from paicorelib import PayloadDataType

from .runtime.runtime import VoltageType
from .runtime.types import NodeName

InputMappingType = dict[NodeName, np.ndarray]
InputMappingAnyType = np.ndarray | Sequence[np.ndarray] | InputMappingType
OutputMappingType = dict[NodeName, PayloadDataType]
NeuVoltageMappingType = VoltageType | dict[NodeName, VoltageType]


@unique
class InferenceMode(Enum):
    NO_BATCH_NO_CHUNK = auto()
    """Inference within a single batch, and 1 chunk/ts."""
    NO_BATCH_WITH_CHUNK = auto()
    """Inference within a single batch, and N chunks/ts."""
    WITH_BATCH_NO_CHUNK = auto()
    """Inference with multiple batches, and 1 chunk/ts in each batch."""
    WITH_BATCH_WITH_CHUNK = auto()
    """Inference with multiple batches, and N chunks/ts in each batch."""

    def is_batch_mode(self) -> bool:
        """Return true if the inference mode is no batch mode."""
        return self in (
            InferenceMode.WITH_BATCH_NO_CHUNK,
            InferenceMode.WITH_BATCH_WITH_CHUNK,
        )

    def is_chunk_mode(self) -> bool:
        """Return true if the inference mode is chunk mode."""
        return self in (
            InferenceMode.NO_BATCH_WITH_CHUNK,
            InferenceMode.WITH_BATCH_WITH_CHUNK,
        )
