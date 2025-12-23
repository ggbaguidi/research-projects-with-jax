from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

DatasetSplit = Literal["train", "valid", "test"]


@dataclass(frozen=True)
class Batch:
    """A single supervised batch.

    `x` is expected to be a NumPy array (adapters can yield NumPy; core converts to JAX).
    `y` is integer class labels of shape (batch,).
    """

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    accuracy: float | None = None
    extra: dict[str, Any] | None = None
