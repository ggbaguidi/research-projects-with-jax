from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

import numpy as np

from jax_deep_learning.core.domain.entities.base import Batch, DatasetSplit
from jax_deep_learning.core.domain.entities.dataset import DatasetInfo
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort


@dataclass(frozen=True)
class TabularCsvConfig:
    id_column: str = "id"
    target_column: str = "diagnosed_diabetes"


def _can_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


class TabularCsvBinaryClassificationDatasetProvider(DatasetProviderPort):
    """Binary classification dataset from train.csv/test.csv (Kaggle style).

    - Handles mixed numeric + categorical columns.
    - One-hot encodes categoricals using vocab built from (train + test) for stability.
    - Standardizes numeric columns using train statistics.

    Splits:
      - "train": training subset
      - "test": validation subset (labels present)

    Inference:
      - call `get_inference()` to get (ids, X_test) for Kaggle submission.
    """

    def __init__(
        self,
        *,
        train_csv_path: str,
        test_csv_path: str,
        valid_fraction: float = 0.2,
        seed: int = 0,
        max_train_rows: int | None = None,
        max_test_rows: int | None = None,
        config: TabularCsvConfig | None = None,
    ) -> None:
        self._cfg = config or TabularCsvConfig()

        train_rows = _read_csv_rows(train_csv_path)
        test_rows = _read_csv_rows(test_csv_path)

        if max_train_rows is not None:
            train_rows = train_rows[: max_train_rows]
        if max_test_rows is not None:
            test_rows = test_rows[: max_test_rows]

        if not train_rows:
            raise ValueError(f"No rows found in {train_csv_path}")
        if not test_rows:
            raise ValueError(f"No rows found in {test_csv_path}")

        # Determine feature columns
        all_cols = list(train_rows[0].keys())
        if self._cfg.id_column not in all_cols:
            raise ValueError(f"Missing id column '{self._cfg.id_column}'")
        if self._cfg.target_column not in all_cols:
            raise ValueError(f"Missing target column '{self._cfg.target_column}'")

        feature_cols = [c for c in all_cols if c not in (self._cfg.id_column, self._cfg.target_column)]

        # Infer numeric vs categorical based on train rows
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for c in feature_cols:
            # if all train values can be cast to float, treat as numeric
            if all(_can_float(r[c]) for r in train_rows):
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)

        # Build categorical vocab from train+test so unseen test categories don't break
        cat_vocab: dict[str, dict[str, int]] = {}
        for c in categorical_cols:
            vals = [r[c] for r in train_rows] + [r[c] for r in test_rows]
            uniq = sorted(set(vals))
            cat_vocab[c] = {v: i for i, v in enumerate(uniq)}

        # Compute numeric stats from train
        num_means: dict[str, float] = {}
        num_stds: dict[str, float] = {}
        for c in numeric_cols:
            arr = np.asarray([float(r[c]) for r in train_rows], dtype=np.float32)
            mu = float(arr.mean())
            std = float(arr.std())
            if std == 0.0:
                std = 1.0
            num_means[c] = mu
            num_stds[c] = std

        self._schema: dict[str, Any] = {
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "cat_vocab": cat_vocab,
            "num_means": num_means,
            "num_stds": num_stds,
        }

        def encode_features(rows: list[dict[str, str]]) -> np.ndarray:
            feats: list[np.ndarray] = []

            if numeric_cols:
                num = np.stack(
                    [
                        (np.asarray([float(r[c]) for r in rows], dtype=np.float32) - num_means[c]) / num_stds[c]
                        for c in numeric_cols
                    ],
                    axis=1,
                )
                feats.append(num)

            for c in categorical_cols:
                vocab = cat_vocab[c]
                idx = np.asarray([vocab[r[c]] for r in rows], dtype=np.int32)
                one_hot = np.zeros((len(rows), len(vocab)), dtype=np.float32)
                one_hot[np.arange(len(rows)), idx] = 1.0
                feats.append(one_hot)

            if not feats:
                raise ValueError("No features detected")
            return np.concatenate(feats, axis=1).astype(np.float32)

        # Encode train (with labels)
        ids_train = np.asarray([int(r[self._cfg.id_column]) for r in train_rows], dtype=np.int64)
        y_all = np.asarray([int(float(r[self._cfg.target_column])) for r in train_rows], dtype=np.int32)
        x_all = encode_features(train_rows)

        # Split train/valid
        rng = np.random.default_rng(seed)
        idx = np.arange(len(x_all))
        rng.shuffle(idx)
        n_valid = int(round(len(idx) * valid_fraction))
        n_valid = max(1, min(n_valid, len(idx) - 1))
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]

        self._x_train = x_all[train_idx]
        self._y_train = y_all[train_idx]
        self._x_valid = x_all[valid_idx]
        self._y_valid = y_all[valid_idx]
        self._ids_valid = ids_train[valid_idx]

        # Encode Kaggle test (no labels)
        self._ids_test = np.asarray([int(r[self._cfg.id_column]) for r in test_rows], dtype=np.int64)
        self._x_test = encode_features(test_rows)

        self._info = DatasetInfo(
            num_classes=2,
            input_shape=(int(self._x_train.shape[1]),),
            train_size=int(self._x_train.shape[0]),
            valid_size=int(self._x_valid.shape[0]),
            test_size=int(self._x_test.shape[0]),
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
    ):
        if split == "train":
            x, y = self._x_train, self._y_train
        else:
            # Map "test" split to validation (labels available)
            x, y = self._x_valid, self._y_valid

        n = len(x)
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)

        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            yield Batch(x=x[sel], y=y[sel])

    def get_validation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ids_valid, x_valid, y_valid)."""

        return self._ids_valid, self._x_valid, self._y_valid

    def get_inference(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids_test, x_test) for Kaggle submission."""

        return self._ids_test, self._x_test
