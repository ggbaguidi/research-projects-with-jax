from __future__ import annotations

import os

# Ensure tests run on CPU-only machines even if JAX is installed with CUDA extras.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.base import Batch
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.use_cases.train_classifier import TrainClassifierUseCase


class _TinyDataset(DatasetProviderPort):
    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._x_train = rng.normal(size=(128, 8)).astype(np.float32)
        self._y_train = rng.integers(0, 3, size=(128,), dtype=np.int32)
        self._x_test = rng.normal(size=(64, 8)).astype(np.float32)
        self._y_test = rng.integers(0, 3, size=(64,), dtype=np.int32)
        self._info = DatasetInfo(num_classes=3, input_shape=(8,))

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def iter_batches(self, *, split: str, batch_size: int, shuffle: bool, seed: int):
        x, y = (self._x_train, self._y_train) if split == "train" else (self._x_test, self._y_test)
        idx = np.arange(len(x))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for start in range(0, len(x), batch_size):
            sel = idx[start : start + batch_size]
            yield Batch(x=x[sel], y=y[sel])


def test_train_classifier_runs_one_epoch() -> None:
    use_case = TrainClassifierUseCase(dataset_provider=_TinyDataset())
    result = use_case.run(
        TrainCommand(
            epochs=1,
            batch_size=16,
            learning_rate=1e-2,
            weight_decay=1e-3,
            adamw_b1=0.85,
            adamw_b2=0.98,
            adamw_eps=1e-7,
            adamw_eps_root=0.0,
            adamw_nesterov=True,
        )
    )
    assert result.history
    assert result.history[-1]["epoch"] == 1
