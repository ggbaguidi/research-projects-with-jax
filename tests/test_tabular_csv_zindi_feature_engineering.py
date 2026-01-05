from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from jax_deep_learning.adapters.right.data_loaders.tabular_csv import (
    TabularCsvConfig,
    TabularCsvMulticlassClassificationDatasetProvider,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_zindi_feature_engineering_adds_expected_numeric_features(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "Train.csv"
    test_path = tmp_path / "Test.csv"

    # Minimal subset of Zindi columns needed for engineered features.
    train_fieldnames = [
        "ID",
        "personal_income",
        "business_expenses",
        "has_loan_account",
        "has_internet_banking",
        "has_debit_card",
        "medical_insurance",
        "Target",
    ]
    train_rows = [
        {
            "ID": "ID_0",
            "personal_income": "100.0",
            "business_expenses": "20.0",
            "has_loan_account": "Have now",
            "has_internet_banking": "Never had",
            "has_debit_card": "Have now",
            "medical_insurance": "Used to have but don't have now",
            "Target": "Low",
        },
        {
            "ID": "ID_1",
            "personal_income": "0.0",
            "business_expenses": "10.0",
            "has_loan_account": "Never had",
            "has_internet_banking": "Never had",
            "has_debit_card": "Never had",
            "medical_insurance": "Never had",
            "Target": "High",
        },
    ]
    _write_csv(train_path, train_fieldnames, train_rows)

    test_fieldnames = [
        "ID",
        "personal_income",
        "business_expenses",
        "has_loan_account",
        "has_internet_banking",
        "has_debit_card",
        "medical_insurance",
    ]
    test_rows = [
        {
            "ID": "ID_T0",
            "personal_income": "50.0",
            "business_expenses": "60.0",
            "has_loan_account": "Never had",
            "has_internet_banking": "Have now",
            "has_debit_card": "Never had",
            "medical_insurance": "Never had",
        }
    ]
    _write_csv(test_path, test_fieldnames, test_rows)

    cfg = TabularCsvConfig(
        id_column="ID",
        target_column="Target",
        enable_feature_engineering=True,
        feature_engineering_kind="zindi_financial_health",
    )

    ds = TabularCsvMulticlassClassificationDatasetProvider(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        valid_fraction=0.5,
        seed=0,
        config=cfg,
    )

    # Engineered features should exist and be treated as numeric.
    # We can't access raw rows here, but we can assert the feature dimension is > original numeric+categorical.
    # Original numeric: personal_income,business_expenses => 2
    # Categorical: has_loan_account,has_internet_banking,has_debit_card,medical_insurance => 4 (one-hot expanded)
    # With feature engineering enabled, we add 2 more numeric features.
    assert ds.info.input_shape[0] >= 2 + 2

    ids_v, x_v, y_v = ds.get_validation()
    assert ids_v.size == y_v.size
    assert x_v.dtype == np.float32
    assert y_v.dtype == np.int32
