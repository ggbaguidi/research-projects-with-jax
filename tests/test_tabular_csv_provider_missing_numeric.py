from __future__ import annotations

import csv

import numpy as np

from jax_deep_learning.adapters.right.data_loaders import (
    TabularCsvBinaryClassificationDatasetProvider,
)


def _write_csv(path, fieldnames, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_missing_numeric_does_not_turn_into_categorical(tmp_path) -> None:
    # Column `age` is mostly numeric but includes missing values; it should still be treated as numeric.
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    fieldnames = ["id", "age", "cat", "diagnosed_diabetes"]
    train_rows = [
        {"id": "1", "age": "40", "cat": "A", "diagnosed_diabetes": "0"},
        {"id": "2", "age": "", "cat": "B", "diagnosed_diabetes": "1"},
        {"id": "3", "age": "50", "cat": "A", "diagnosed_diabetes": "0"},
        {"id": "4", "age": "60", "cat": "B", "diagnosed_diabetes": "1"},
        {"id": "5", "age": "55", "cat": "A", "diagnosed_diabetes": "0"},
    ]
    test_rows = [
        {"id": "10", "age": "45", "cat": "A"},
        {"id": "11", "age": "", "cat": "B"},
    ]

    _write_csv(train_path, fieldnames, train_rows)
    _write_csv(test_path, ["id", "age", "cat"], test_rows)

    ds = TabularCsvBinaryClassificationDatasetProvider(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        valid_fraction=0.4,
        seed=0,
    )

    desc = ds.describe()
    # Feature engineering adds Age_40+ and Age_55+ when `age` exists.
    assert desc["n_numeric"] == 3
    assert desc["n_categorical"] == 1

    # Feature dim: 3 numeric (age, Age_40+, Age_55+) + one-hot(cat) with 2 categories (A,B)
    assert ds.info.input_shape == (5,)

    ids_v, x_v, y_v = ds.get_validation()
    assert x_v.shape[1] == 5
    assert set(np.unique(y_v)).issubset({0, 1})
    assert ids_v.shape[0] == x_v.shape[0]
