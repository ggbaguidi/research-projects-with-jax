from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from jax_deep_learning.core.domain.entities.base import Batch, DatasetSplit
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort


class TfdsClassificationDatasetProvider(DatasetProviderPort):
    """TFDS-backed supervised dataset provider.

    Uses TFDS + tf.data for preprocessing/batching, then yields NumPy batches.
    """

    def __init__(
        self,
        *,
        name: str,
        data_dir: str = "/tmp/tfds",
        image_normalize_0_1: bool = True,
    ) -> None:
        data, info = tfds.load(name=name, data_dir=data_dir, as_supervised=True, with_info=True)
        self._tf_train = data.get("train")
        self._tf_test = data.get("test") or data.get("validation")
        if self._tf_train is None or self._tf_test is None:
            raise ValueError(f"TFDS dataset '{name}' must have train and test/validation splits")

        self._num_classes = int(info.features["label"].num_classes)

        # input shape from features; for images, it's (H, W, C)
        shape = tuple(int(d) for d in info.features["image"].shape)
        self._info = DatasetInfo(num_classes=self._num_classes, input_shape=shape)

        def preprocess(image, label):
            if image_normalize_0_1:
                image = tf.cast(image, tf.float32) / 255.0
            return image, label

        self._preprocess = preprocess

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def _pipeline(self, ds: tf.data.Dataset, *, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
        ds = ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def iter_batches(
        self,
        *,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterable[Batch]:
        ds = self._tf_train if split == "train" else self._tf_test
        for x, y in tfds.as_numpy(self._pipeline(ds, batch_size=batch_size, shuffle=shuffle, seed=seed)):
            yield Batch(x=np.asarray(x), y=np.asarray(y))
