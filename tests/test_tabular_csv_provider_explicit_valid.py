from __future__ import annotations

import csv

import numpy as np

from jax_deep_learning.adapters.right.data_loaders import (
    TabularCsvConfig,
    TabularCsvMulticlassClassificationDatasetProvider,
)


def _write_csv(path, fieldnames, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_multiclass_provider_accepts_explicit_valid_csv(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"

    # Keep it tiny but multiclass.
    fieldnames = ["ID", "num", "cat", "Target"]
    train_rows = [
        {"ID": "1", "num": "1.0", "cat": "A", "Target": "Low"},
        {"ID": "2", "num": "2.0", "cat": "B", "Target": "Medium"},
        {"ID": "3", "num": "3.0", "cat": "A", "Target": "High"},
    ]
    valid_rows = [
        {"ID": "4", "num": "4.0", "cat": "B", "Target": "Low"},
        {"ID": "5", "num": "5.0", "cat": "A", "Target": "Medium"},
    ]
    test_rows = [
        {"ID": "10", "num": "10.0", "cat": "A"},
        {"ID": "11", "num": "11.0", "cat": "B"},
    ]

    _write_csv(train_path, fieldnames, train_rows)
    _write_csv(valid_path, fieldnames, valid_rows)
    _write_csv(test_path, ["ID", "num", "cat"], test_rows)

    cfg = TabularCsvConfig(
        id_column="ID",
        target_column="Target",
        enable_feature_engineering=False,
        categorical_encoding="onehot",
    )

    ds = TabularCsvMulticlassClassificationDatasetProvider(
        train_csv_path=str(train_path),
        valid_csv_path=str(valid_path),
        test_csv_path=str(test_path),
        config=cfg,
        class_names=("Low", "Medium", "High"),
    )

    ids_v, x_v, y_v = ds.get_validation()
    assert ids_v.shape[0] == len(valid_rows)
    assert x_v.shape[0] == len(valid_rows)
    assert y_v.shape[0] == len(valid_rows)

    # Expect a stable, dense float32 feature matrix.
    assert x_v.dtype == np.float32
    assert x_v.ndim == 2
    assert x_v.shape[1] == ds.info.input_shape[0]

    # Ensure y is in-range.
    assert set(np.unique(y_v)).issubset(set(range(ds.info.num_classes)))
