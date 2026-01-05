from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.base import Batch
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.domain.entities.model import TabularEmbedMlpClassifierFns
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.use_cases.train_classifier import TrainClassifierUseCase


class _TinyTabularEmbedDataset(DatasetProviderPort):
    def __init__(self) -> None:
        rng = np.random.default_rng(0)

        n_numeric = 3
        n_cat = 2

        # Numeric features
        x_num_tr = rng.normal(size=(128, n_numeric)).astype(np.float32)
        x_num_te = rng.normal(size=(64, n_numeric)).astype(np.float32)

        # Categorical indices stored as float32 in the batch (as done by the adapter)
        c0_tr = rng.integers(0, 4, size=(128, 1), dtype=np.int32).astype(np.float32)
        c1_tr = rng.integers(0, 7, size=(128, 1), dtype=np.int32).astype(np.float32)
        c0_te = rng.integers(0, 4, size=(64, 1), dtype=np.int32).astype(np.float32)
        c1_te = rng.integers(0, 7, size=(64, 1), dtype=np.int32).astype(np.float32)

        self._x_train = np.concatenate([x_num_tr, c0_tr, c1_tr], axis=1)
        self._x_test = np.concatenate([x_num_te, c0_te, c1_te], axis=1)

        # Binary labels
        self._y_train = rng.integers(0, 2, size=(128,), dtype=np.int32)
        self._y_test = rng.integers(0, 2, size=(64,), dtype=np.int32)

        self._info = DatasetInfo(num_classes=2, input_shape=(n_numeric + n_cat,))

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


def test_train_classifier_runs_with_tabular_embeddings() -> None:
    ds = _TinyTabularEmbedDataset()
    model = TabularEmbedMlpClassifierFns(n_numeric=3, categorical_cardinalities=(4, 7), embed_dim=4, hidden_sizes=(16,))
    use_case = TrainClassifierUseCase(dataset_provider=ds, model_fns=model)

    result = use_case.run(TrainCommand(epochs=1, batch_size=32, learning_rate=1e-2))
    assert result.history
    assert result.history[-1]["epoch"] == 1
