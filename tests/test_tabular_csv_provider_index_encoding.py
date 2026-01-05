from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from jax_deep_learning.adapters.right.data_loaders.tabular_csv_kaggle import (
    TabularCsvBinaryClassificationDatasetProvider,
    TabularCsvConfig,
)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join([",".join(header), *[",".join(r) for r in rows]]) + "\n",
        encoding="utf-8",
    )


def test_index_encoding_appends_cat_indices(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    header = ["id", "diagnosed_diabetes", "age", "gender", "smoking_status"]
    train_rows = [
        ["1", "0", "40", "male", "never"],
        ["2", "1", "55", "female", "former"],
        ["3", "0", "60", "male", "current"],
        ["4", "1", "45", "female", "never"],
    ]
    test_header = ["id", "age", "gender", "smoking_status"]
    test_rows = [
        ["10", "50", "male", "never"],
        ["11", "35", "female", "former"],
    ]

    _write_csv(train_path, header, train_rows)
    _write_csv(test_path, test_header, test_rows)

    cfg = TabularCsvConfig(
        enable_feature_engineering=False, categorical_encoding="index"
    )
    ds = TabularCsvBinaryClassificationDatasetProvider(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        valid_fraction=0.25,
        seed=0,
        config=cfg,
    )

    # age is numeric; gender+smoking_status are categorical -> numeric_dim=1, n_cat=2
    assert ds.numeric_dim == 1
    assert ds.categorical_cols == ("gender", "smoking_status")
    assert len(ds.categorical_cardinalities) == 2

    ids_v, x_v, y_v = ds.get_validation()
    assert x_v.shape[1] == ds.numeric_dim + len(ds.categorical_cols)
    assert x_v.dtype == np.float32

    # categorical indices should be near-integers and within range
    cat_part = x_v[:, ds.numeric_dim :]
    assert np.all(np.isfinite(cat_part))
    assert np.allclose(cat_part, np.round(cat_part))
    for j, card in enumerate(ds.categorical_cardinalities):
        idx = cat_part[:, j]
        assert idx.min() >= 0
        assert idx.max() <= (card - 1)

    assert ids_v.shape[0] == y_v.shape[0]
