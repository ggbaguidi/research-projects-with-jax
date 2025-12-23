from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from jax_deep_learning.core.domain.entities.base import Batch, DatasetSplit
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo


class DatasetProviderPort(Protocol):
    """Port for providing supervised mini-batches to the core."""

    @property
    def info(self) -> DatasetInfo: ...

    def iter_batches(
        self,
        *,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterable[Batch]: ...
