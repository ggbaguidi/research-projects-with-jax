from __future__ import annotations

import numpy as np

from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.base import Batch
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.use_cases.train_classifier import TrainClassifierUseCase


class _TinyMulticlassDataset(DatasetProviderPort):
    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        # Small deterministic dataset; labels are imbalanced-ish but contain all classes.
        self._x_train = rng.normal(size=(90, 6)).astype(np.float32)
        self._y_train = np.asarray(([0] * 60) + ([1] * 25) + ([2] * 5), dtype=np.int32)
        self._x_valid = rng.normal(size=(90, 6)).astype(np.float32)
        self._y_valid = np.asarray(([0] * 60) + ([1] * 25) + ([2] * 5), dtype=np.int32)
        self._info = DatasetInfo(num_classes=3, input_shape=(6,))

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


def test_early_stopping_multiclass_macro_f1_triggers_when_no_learning() -> None:
    # With lr=0, params won't change, so macro-F1 shouldn't improve across epochs.
    # With patience=2, training should stop after epoch 3.
    use_case = TrainClassifierUseCase(dataset_provider=_TinyMulticlassDataset())
    cmd = TrainCommand(
        epochs=20,
        batch_size=16,
        learning_rate=0.0,
        early_stopping_patience=2,
    )
    result = use_case.run(cmd)

    assert len(result.history) == 3
    assert result.history[-1]["epoch"] == 3
    assert "test/macro_f1" in result.history[-1]
