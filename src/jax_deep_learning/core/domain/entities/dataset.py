from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata required by the core training loop."""

    num_classes: int
    input_shape: tuple[int, ...]
    train_size: int | None = None
    valid_size: int | None = None
    test_size: int | None = None
    class_names: tuple[str, ...] | None = None
