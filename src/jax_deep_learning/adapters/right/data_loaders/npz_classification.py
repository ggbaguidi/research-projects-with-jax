from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from jax_deep_learning.core.domain.entities.base import Batch, DatasetSplit
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort


class NpzClassificationDatasetProvider(DatasetProviderPort):
    """Loads supervised arrays from a .npz file.

    Expected keys:
      - x_train, y_train
      - x_test, y_test  (or x_valid, y_valid)

    This is a pragmatic adapter for Kaggle/Zindi: convert raw competition data to
    NPZ once, then iterate fast in JAX.
    """

    def __init__(self, *, path: str) -> None:
        data = np.load(path)
        self._x_train = data["x_train"]
        self._y_train = data["y_train"]

        if "x_valid" in data and "y_valid" in data:
            self._x_test = data["x_valid"]
            self._y_test = data["y_valid"]
        else:
            self._x_test = data["x_test"]
            self._y_test = data["y_test"]

        num_classes = int(np.max(self._y_train)) + 1
        self._info = DatasetInfo(
            num_classes=num_classes, input_shape=tuple(self._x_train.shape[1:])
        )

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def iter_batches(
        self,
        *,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterable[Batch]:
        if split == "train":
            x, y = self._x_train, self._y_train
        else:
            x, y = self._x_test, self._y_test

        n = len(x)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            yield Batch(x=x[sel], y=y[sel])
