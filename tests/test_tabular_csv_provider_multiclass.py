from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from jax_deep_learning.adapters.right.data_loaders.tabular_csv_kaggle import (
    TabularCsvConfig,
    TabularCsvMulticlassClassificationDatasetProvider,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_multiclass_provider_builds_label_mapping_and_ids(tmp_path: Path) -> None:
    train_path = tmp_path / "Train.csv"
    test_path = tmp_path / "Test.csv"

    fieldnames = ["ID", "n0", "cat", "Target"]

    # 3 classes, 3 examples each -> stratified holdout should keep all classes.
    train_rows: list[dict[str, str]] = []
    for i in range(3):
        train_rows.append({"ID": f"ID_L{i}", "n0": str(1 + i), "cat": "A", "Target": "Low"})
        train_rows.append({"ID": f"ID_M{i}", "n0": str(10 + i), "cat": "B", "Target": "Medium"})
        train_rows.append({"ID": f"ID_H{i}", "n0": str(100 + i), "cat": "C", "Target": "High"})

    _write_csv(train_path, fieldnames, train_rows)

    test_fieldnames = ["ID", "n0", "cat"]
    test_rows = [
        {"ID": "ID_T0", "n0": "7", "cat": "A"},
        {"ID": "ID_T1", "n0": "8", "cat": "B"},
    ]
    _write_csv(test_path, test_fieldnames, test_rows)

    cfg = TabularCsvConfig(id_column="ID", target_column="Target", enable_feature_engineering=False)

    ds = TabularCsvMulticlassClassificationDatasetProvider(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        valid_fraction=0.33,
        seed=0,
        config=cfg,
    )

    assert ds.info.num_classes == 3
    # Known competition ordering (if exactly {Low,Medium,High})
    assert ds.class_names == ("Low", "Medium", "High")

    ids_v, x_v, y_v = ds.get_validation()
    assert ids_v.dtype == object
    assert x_v.dtype == np.float32
    assert y_v.dtype == np.int32
    assert set(np.unique(y_v).tolist()).issubset({0, 1, 2})

    # Validate that all classes exist in the overall dataset (train+valid).
    y_all = np.concatenate([ds._y_train, ds._y_valid], axis=0)
    assert set(np.unique(y_all).tolist()) == {0, 1, 2}

    ids_t, x_t = ds.get_inference()
    assert ids_t.tolist() == ["ID_T0", "ID_T1"]
    assert x_t.shape[1] == ds.info.input_shape[0]
