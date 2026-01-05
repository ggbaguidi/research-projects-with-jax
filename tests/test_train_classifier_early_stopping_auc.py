from __future__ import annotations

import numpy as np

from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.base import Batch
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.use_cases.train_classifier import TrainClassifierUseCase


class _TinyBinaryDataset(DatasetProviderPort):
    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._x_train = rng.normal(size=(64, 6)).astype(np.float32)
        self._y_train = rng.integers(0, 2, size=(64,), dtype=np.int32)
        self._x_valid = rng.normal(size=(64, 6)).astype(np.float32)
        self._y_valid = rng.integers(0, 2, size=(64,), dtype=np.int32)
        self._info = DatasetInfo(num_classes=2, input_shape=(6,))

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def iter_batches(self, *, split: str, batch_size: int, shuffle: bool, seed: int):
        x, y = (
            (self._x_train, self._y_train)
            if split == "train"
            else (self._x_valid, self._y_valid)
        )
        idx = np.arange(len(x))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for start in range(0, len(x), batch_size):
            sel = idx[start : start + batch_size]
            yield Batch(x=x[sel], y=y[sel])


def test_early_stopping_triggers_when_no_learning() -> None:
    # With lr=0, params won't change, so AUC should not improve across epochs.
    # With patience=1, training should stop after the second epoch.
    use_case = TrainClassifierUseCase(dataset_provider=_TinyBinaryDataset())
    cmd = TrainCommand(
        epochs=10, batch_size=16, learning_rate=0.0, early_stopping_patience=1
    )
    result = use_case.run(cmd)

    assert len(result.history) == 2
    assert result.history[-1]["epoch"] == 2
    assert "test/auc" in result.history[-1]
