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
    numeric_min_fraction: float = 0.95
    missing_tokens: tuple[str, ...] = ("", "na", "nan", "none", "null")
    categorical_missing_token: str = "__MISSING__"
    # How to encode categoricals:
    # - "onehot": append one-hot vectors per categorical column (current default)
    # - "index": append one integer index per categorical column (for embedding models)
    categorical_encoding: str = "onehot"
    enable_feature_engineering: bool = True
    # Feature engineering profile (used only when enable_feature_engineering=True):
    # - "diabetes": engineered features for the Kaggle diabetes playground dataset
    # - "zindi_financial_health": engineered features for the data.org Financial Health challenge
    feature_engineering_kind: str = "diabetes"


def _safe_lower(s: str) -> str:
    return (s or "").strip().lower()


def _can_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _is_missing(s: str, *, cfg: TabularCsvConfig) -> bool:
    return s is None or s.strip().lower() in cfg.missing_tokens


def _to_float_or_nan(s: str, *, cfg: TabularCsvConfig) -> float:
    if _is_missing(s, cfg=cfg):
        return float("nan")

    try:
        return float(s)
    except Exception:
        return float("nan")


def _format_float_for_csv(x: float) -> str:
    if x != x:  # NaN
        return ""
    # Use repr-like formatting to preserve enough precision while staying compact.
    return f"{float(x):.10g}"


def _format_int_for_csv(x: int) -> str:
    return "1" if int(x) != 0 else "0"


def _apply_feature_engineering_diabetes(
    rows: list[dict[str, str]], *, cfg: TabularCsvConfig
) -> None:
    """In-place feature engineering (pandas-free).

    Mirrors the user's provided feature engineering function, but operates on
    CSV row dicts.

    Rules:
    - If required source columns are missing/unparseable, we skip adding that
      engineered feature (to avoid silently injecting constant/empty features).
    - Engineered numeric values are written as strings; missing is written as ''.
    """

    def has_cols(*cols: str) -> bool:
        for c in cols:
            if not any((c in r) for r in rows):
                return False
        return True

    # Age-derived
    add_age = has_cols("age")
    add_central_obesity = has_cols("gender", "waist_to_hip_ratio")
    add_bmi_whr = has_cols("bmi", "waist_to_hip_ratio")
    add_sleep_screen_stress = has_cols(
        "sleep_hours_per_day", "screen_time_hours_per_day"
    )
    add_alcohol = has_cols("alcohol_consumption_per_week")
    add_low_activity = has_cols("physical_activity_minutes_per_week")
    add_hypertension = has_cols("systolic_bp", "diastolic_bp")
    add_bp_load = has_cols("systolic_bp", "diastolic_bp")
    add_tg_hdl = has_cols("triglycerides", "hdl_cholesterol")
    add_lipid_risk = has_cols("triglycerides", "hdl_cholesterol")
    add_high_hr = has_cols("heart_rate")
    add_low_ses = has_cols("education_level", "income_level")
    add_genetic_lifestyle = has_cols("family_history_diabetes", "bmi")
    add_comorbidity = has_cols(
        "hypertension_history", "cardiovascular_history", "family_history_diabetes"
    )
    add_silent_diabetic = has_cols(
        "bmi", "physical_activity_minutes_per_week", "triglycerides"
    )

    for r in rows:
        # Age_40+, Age_55+
        if add_age:
            age = _to_float_or_nan(r.get("age", ""), cfg=cfg)
            if age == age:
                r["Age_40+"] = _format_int_for_csv(int(age >= 40))
                r["Age_55+"] = _format_int_for_csv(int(age >= 55))

        # Central_Obesity
        if add_central_obesity:
            gender = (r.get("gender", "") or "").strip()
            whr = _to_float_or_nan(r.get("waist_to_hip_ratio", ""), cfg=cfg)
            if gender and whr == whr:
                central = (gender.lower() == "male" and whr > 0.9) or (
                    gender.lower() == "female" and whr > 0.85
                )
                r["Central_Obesity"] = _format_int_for_csv(int(central))

        # BMI_WHR
        if add_bmi_whr:
            bmi = _to_float_or_nan(r.get("bmi", ""), cfg=cfg)
            whr = _to_float_or_nan(r.get("waist_to_hip_ratio", ""), cfg=cfg)
            if bmi == bmi and whr == whr:
                r["BMI_WHR"] = _format_float_for_csv(bmi * whr)

        # Sleep_Screen_Stress
        if add_sleep_screen_stress:
            sleep_h = _to_float_or_nan(r.get("sleep_hours_per_day", ""), cfg=cfg)
            screen_h = _to_float_or_nan(r.get("screen_time_hours_per_day", ""), cfg=cfg)
            if sleep_h == sleep_h and screen_h == screen_h:
                r["Sleep_Screen_Stress"] = _format_int_for_csv(
                    int((sleep_h < 6) and (screen_h > 6))
                )

        # Alcohol_High, Alcohol_Zero
        if add_alcohol:
            alc = _to_float_or_nan(r.get("alcohol_consumption_per_week", ""), cfg=cfg)
            if alc == alc:
                r["Alcohol_High"] = _format_int_for_csv(int(alc > 14))
                r["Alcohol_Zero"] = _format_int_for_csv(int(alc == 0))

        # Low_Activity
        if add_low_activity:
            act = _to_float_or_nan(
                r.get("physical_activity_minutes_per_week", ""), cfg=cfg
            )
            if act == act:
                r["Low_Activity"] = _format_int_for_csv(int(act < 150))

        # Hypertension
        if add_hypertension:
            sys_bp = _to_float_or_nan(r.get("systolic_bp", ""), cfg=cfg)
            dia_bp = _to_float_or_nan(r.get("diastolic_bp", ""), cfg=cfg)
            if sys_bp == sys_bp and dia_bp == dia_bp:
                r["Hypertension"] = _format_int_for_csv(
                    int((sys_bp >= 130) or (dia_bp >= 80))
                )

        # BP_Load
        if add_bp_load:
            sys_bp = _to_float_or_nan(r.get("systolic_bp", ""), cfg=cfg)
            dia_bp = _to_float_or_nan(r.get("diastolic_bp", ""), cfg=cfg)
            if sys_bp == sys_bp and dia_bp == dia_bp:
                r["BP_Load"] = _format_float_for_csv(sys_bp * dia_bp)

        # TG_HDL_Ratio
        if add_tg_hdl:
            tg = _to_float_or_nan(r.get("triglycerides", ""), cfg=cfg)
            hdl = _to_float_or_nan(r.get("hdl_cholesterol", ""), cfg=cfg)
            if tg == tg and hdl == hdl:
                r["TG_HDL_Ratio"] = _format_float_for_csv(tg / (hdl + 1.0))

        # Lipid_Risk
        if add_lipid_risk:
            tg = _to_float_or_nan(r.get("triglycerides", ""), cfg=cfg)
            hdl = _to_float_or_nan(r.get("hdl_cholesterol", ""), cfg=cfg)
            if tg == tg and hdl == hdl:
                r["Lipid_Risk"] = _format_int_for_csv(int((tg > 150) and (hdl < 40)))

        # High_Heart_Rate
        if add_high_hr:
            hr = _to_float_or_nan(r.get("heart_rate", ""), cfg=cfg)
            if hr == hr:
                r["High_Heart_Rate"] = _format_int_for_csv(int(hr > 85))

        # Low_SES
        if add_low_ses:
            edu = (r.get("education_level", "") or "").strip().lower()
            inc = (r.get("income_level", "") or "").strip().lower()
            if edu or inc:
                low = (edu == "low") or (inc == "low")
                r["Low_SES"] = _format_int_for_csv(int(low))

        # Genetic_Lifestyle_Risk
        if add_genetic_lifestyle:
            fam = _to_float_or_nan(r.get("family_history_diabetes", ""), cfg=cfg)
            bmi = _to_float_or_nan(r.get("bmi", ""), cfg=cfg)
            if fam == fam and bmi == bmi:
                r["Genetic_Lifestyle_Risk"] = _format_float_for_csv(
                    fam * float(bmi > 30)
                )

        # Comorbidity_Count
        if add_comorbidity:
            hyp = _to_float_or_nan(r.get("hypertension_history", ""), cfg=cfg)
            cvd = _to_float_or_nan(r.get("cardiovascular_history", ""), cfg=cfg)
            fam = _to_float_or_nan(r.get("family_history_diabetes", ""), cfg=cfg)
            if hyp == hyp and cvd == cvd and fam == fam:
                r["Comorbidity_Count"] = _format_float_for_csv(hyp + cvd + fam)

        # Silent_Diabetic
        if add_silent_diabetic:
            bmi = _to_float_or_nan(r.get("bmi", ""), cfg=cfg)
            act = _to_float_or_nan(
                r.get("physical_activity_minutes_per_week", ""), cfg=cfg
            )
            tg = _to_float_or_nan(r.get("triglycerides", ""), cfg=cfg)
            if bmi == bmi and act == act and tg == tg:
                silent = (bmi < 25) and (act > 150) and (tg > 150)
                r["Silent_Diabetic"] = _format_int_for_csv(int(silent))


def _apply_feature_engineering_zindi_financial_health(
    rows: list[dict[str, str]], *, cfg: TabularCsvConfig
) -> None:
    """Feature engineering for the data.org Financial Health challenge.

    Adds (when source columns exist):
      - profit_margin = (personal_income - business_expenses) / personal_income, capped to [-1, 1]
      - financial_access_score in [0, 1] based on access to formal financial services

    Notes:
      - Numeric engineered features are stored as strings using _format_float_for_csv.
      - Missing stays missing (empty string).
    """

    def has_cols(*cols: str) -> bool:
        for c in cols:
            if not any((c in r) for r in rows):
                return False
        return True

    add_profit_margin = has_cols("personal_income", "business_expenses")

    # Use only columns that actually exist; normalize by those present.
    access_cols = [
        "has_loan_account",
        "has_internet_banking",
        "has_debit_card",
        "has_credit_card",
        "has_mobile_money",
        "has_insurance",
        "medical_insurance",
        "funeral_insurance",
        "motor_vehicle_insurance",
    ]
    available_access_cols = [c for c in access_cols if any((c in r) for r in rows)]

    def access_value_score(v: str) -> float | None:
        if _is_missing(v, cfg=cfg):
            return None
        s = _safe_lower(v)
        # Past access gets partial credit.
        if "used to have" in s:
            return 0.5
        # Current access signals.
        if "have now" in s:
            return 1.0
        if s in {"yes", "y", "true", "1"}:
            return 1.0
        if s.startswith("yes"):
            # e.g. "Yes, always", "Yes, sometimes"
            return 1.0
        # Explicit negatives or other categories.
        if s in {"no", "n", "false", "0"}:
            return 0.0
        if "never had" in s:
            return 0.0
        if "don't have now" in s:
            # Should have been caught by "used to have" but keep as safety.
            return 0.0
        if "don't know" in s or "n/a" in s:
            return None
        # Default: treat as unknown -> missing.
        return None

    for r in rows:
        # profit_margin
        if add_profit_margin:
            income = _to_float_or_nan(r.get("personal_income", ""), cfg=cfg)
            expenses = _to_float_or_nan(r.get("business_expenses", ""), cfg=cfg)
            if income == income and expenses == expenses and income != 0.0:
                margin = (income - expenses) / income
                # Cap to [-1, 1] to reduce extreme effects.
                if margin > 1.0:
                    margin = 1.0
                elif margin < -1.0:
                    margin = -1.0
                r["profit_margin"] = _format_float_for_csv(float(margin))

        # financial_access_score
        if available_access_cols:
            scores: list[float] = []
            for c in available_access_cols:
                sc = access_value_score(r.get(c, ""))
                if sc is not None:
                    scores.append(float(sc))
            if scores:
                r["financial_access_score"] = _format_float_for_csv(
                    float(sum(scores) / len(scores))
                )


def _apply_feature_engineering(
    rows: list[dict[str, str]], *, cfg: TabularCsvConfig
) -> None:
    kind = _safe_lower(getattr(cfg, "feature_engineering_kind", "diabetes"))
    if kind in {"diabetes", "kaggle_diabetes", "kaggle"}:
        _apply_feature_engineering_diabetes(rows, cfg=cfg)
    elif kind in {"zindi_financial_health", "zindi-financial-health", "zindi"}:
        _apply_feature_engineering_zindi_financial_health(rows, cfg=cfg)
    else:
        # Unknown kind: do nothing (safer than crashing in the CLI).
        return


def _stratified_split_binary(
    *,
    y: np.ndarray,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, valid_idx) for binary labels with stratification."""

    y = np.asarray(y).astype(np.int32)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n = len(y)
    n_valid_total = int(round(n * valid_fraction))
    n_valid_total = max(1, min(n_valid_total, n - 1))

    # Allocate validation counts proportionally, but ensure each class (if present)
    # contributes at least 1 example.
    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        # Degenerate case; fall back to random split.
        idx = np.arange(n)
        rng.shuffle(idx)
        valid_idx = idx[:n_valid_total]
        train_idx = idx[n_valid_total:]
        return train_idx, valid_idx

    frac1 = n1 / n
    n_valid_1 = int(round(n_valid_total * frac1))
    n_valid_1 = max(1, min(n_valid_1, n1 - 1))
    n_valid_0 = n_valid_total - n_valid_1
    n_valid_0 = max(1, min(n_valid_0, n0 - 1))

    # Re-adjust total if rounding pushed us over.
    while (n_valid_0 + n_valid_1) > n_valid_total:
        if n_valid_0 > 1:
            n_valid_0 -= 1
        elif n_valid_1 > 1:
            n_valid_1 -= 1
        else:
            break
    while (n_valid_0 + n_valid_1) < n_valid_total:
        # Add to the larger class if possible.
        if n0 - n_valid_0 > n1 - n_valid_1 and n_valid_0 < (n0 - 1):
            n_valid_0 += 1
        elif n_valid_1 < (n1 - 1):
            n_valid_1 += 1
        else:
            break

    valid_idx = np.concatenate([idx0[:n_valid_0], idx1[:n_valid_1]], axis=0)
    train_idx = np.concatenate([idx0[n_valid_0:], idx1[n_valid_1:]], axis=0)
    rng.shuffle(valid_idx)
    rng.shuffle(train_idx)
    return train_idx, valid_idx


def _stratified_split_multiclass(
    *,
    y: np.ndarray,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, valid_idx) for integer labels with stratification.

    Ensures that, when possible, each class keeps at least 1 example in the
    training set. Falls back to a random split for degenerate inputs.
    """

    y = np.asarray(y).astype(np.int32)
    n = int(y.shape[0])

    rng = np.random.default_rng(seed)

    n_valid_total = int(round(n * valid_fraction))
    n_valid_total = max(1, min(n_valid_total, n - 1))

    classes, counts = np.unique(y, return_counts=True)
    if classes.size <= 1:
        idx = np.arange(n)
        rng.shuffle(idx)
        valid_idx = idx[:n_valid_total]
        train_idx = idx[n_valid_total:]
        return train_idx, valid_idx

    # Desired per-class allocation (rounded) with constraints.
    per_class_valid: dict[int, int] = {}
    for c, cnt in zip(classes.tolist(), counts.tolist()):
        if cnt <= 1:
            per_class_valid[int(c)] = 0
            continue
        target = int(round(n_valid_total * (cnt / n)))
        # Keep at least 1 in valid (if possible) but also at least 1 in train.
        target = max(1, min(target, cnt - 1))
        per_class_valid[int(c)] = int(target)

    # Adjust totals to match n_valid_total.
    def total_valid() -> int:
        return int(sum(per_class_valid.values()))

    # Reduce if we overshot.
    while total_valid() > n_valid_total:
        # Reduce from the class with the largest current allocation (but not below 0).
        c_best = None
        best_v = 0
        for c, v in per_class_valid.items():
            if v > best_v and v > 0:
                c_best = c
                best_v = v
        if c_best is None:
            break
        per_class_valid[c_best] -= 1

    # Increase if we undershot.
    if total_valid() < n_valid_total:
        count_by_class = {
            int(c): int(cnt) for c, cnt in zip(classes.tolist(), counts.tolist())
        }
        while total_valid() < n_valid_total:
            # Add to the class with most remaining capacity.
            c_best = None
            best_cap = -1
            for c, cnt in count_by_class.items():
                cap = (cnt - 1) - per_class_valid.get(c, 0)
                if cap > best_cap and cap > 0:
                    c_best = c
                    best_cap = cap
            if c_best is None:
                break
            per_class_valid[c_best] += 1

    # Materialize indices.
    valid_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for c in classes.tolist():
        c = int(c)
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_valid_c = int(per_class_valid.get(c, 0))
        valid_parts.append(idx[:n_valid_c])
        train_parts.append(idx[n_valid_c:])

    valid_idx = np.concatenate(valid_parts, axis=0)
    train_idx = np.concatenate(train_parts, axis=0)
    rng.shuffle(valid_idx)
    rng.shuffle(train_idx)
    return train_idx, valid_idx


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
        add_noise: bool = False,
        noise_std: float = 0.1,
    ) -> None:
        self._cfg = config or TabularCsvConfig()
        self._add_noise = bool(add_noise)
        self._noise_std = float(noise_std)
        self._numeric_dim = 0

        train_rows = _read_csv_rows(train_csv_path)
        test_rows = _read_csv_rows(test_csv_path)

        if self._cfg.enable_feature_engineering:
            _apply_feature_engineering(train_rows, cfg=self._cfg)
            _apply_feature_engineering(test_rows, cfg=self._cfg)

        if max_train_rows is not None:
            train_rows = train_rows[:max_train_rows]
        if max_test_rows is not None:
            test_rows = test_rows[:max_test_rows]

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

        feature_cols = [
            c
            for c in all_cols
            if c not in (self._cfg.id_column, self._cfg.target_column)
        ]

        # Infer numeric vs categorical based on train rows
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for c in feature_cols:
            values = [r.get(c, "") for r in train_rows]
            non_missing = [v for v in values if not _is_missing(v, cfg=self._cfg)]
            if not non_missing:
                # All missing: treat as categorical so it can at least learn a constant token.
                categorical_cols.append(c)
                continue

            n_ok = sum(1 for v in non_missing if _can_float(v))
            frac_ok = n_ok / max(1, len(non_missing))
            if frac_ok >= self._cfg.numeric_min_fraction:
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)

        # Build categorical vocab from train+test so unseen test categories don't break
        cat_vocab: dict[str, dict[str, int]] = {}
        for c in categorical_cols:
            vals = []
            for r in train_rows:
                v = r.get(c, "")
                vals.append(
                    self._cfg.categorical_missing_token
                    if _is_missing(v, cfg=self._cfg)
                    else v
                )
            for r in test_rows:
                v = r.get(c, "")
                vals.append(
                    self._cfg.categorical_missing_token
                    if _is_missing(v, cfg=self._cfg)
                    else v
                )
            uniq = sorted(set(vals))
            cat_vocab[c] = {v: i for i, v in enumerate(uniq)}

        # Compute numeric stats from train
        num_means: dict[str, float] = {}
        num_stds: dict[str, float] = {}
        for c in numeric_cols:
            arr = np.asarray(
                [_to_float_or_nan(r.get(c, ""), cfg=self._cfg) for r in train_rows],
                dtype=np.float32,
            )
            mu = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            if np.isnan(mu):
                mu = 0.0
            if np.isnan(std) or std == 0.0:
                std = 1.0
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

        # Feature layout is: [standardized numeric cols..., one-hot categoricals...]
        self._numeric_dim = int(len(numeric_cols))

        def encode_features(rows: list[dict[str, str]]) -> np.ndarray:
            feats: list[np.ndarray] = []

            if numeric_cols:
                num = np.stack(
                    [
                        (
                            np.nan_to_num(
                                np.asarray(
                                    [
                                        _to_float_or_nan(r.get(c, ""), cfg=self._cfg)
                                        for r in rows
                                    ],
                                    dtype=np.float32,
                                ),
                                nan=num_means[c],
                            )
                            - num_means[c]
                        )
                        / num_stds[c]
                        for c in numeric_cols
                    ],
                    axis=1,
                )
                feats.append(num)

            for c in categorical_cols:
                vocab = cat_vocab[c]
                idx = np.asarray(
                    [
                        vocab[
                            (
                                self._cfg.categorical_missing_token
                                if _is_missing(r.get(c, ""), cfg=self._cfg)
                                else r.get(c, "")
                            )
                        ]
                        for r in rows
                    ],
                    dtype=np.int32,
                )
                enc = (self._cfg.categorical_encoding or "onehot").strip().lower()
                if enc == "index":
                    # One integer id per categorical column (stored as float32 for JAX-friendly batching).
                    feats.append(idx.reshape(-1, 1).astype(np.float32))
                else:
                    one_hot = np.zeros((len(rows), len(vocab)), dtype=np.float32)
                    one_hot[np.arange(len(rows)), idx] = 1.0
                    feats.append(one_hot)

            if not feats:
                raise ValueError("No features detected")
            return np.concatenate(feats, axis=1).astype(np.float32)

        # Encode train (with labels)
        ids_train = np.asarray(
            [int(r[self._cfg.id_column]) for r in train_rows], dtype=np.int64
        )
        y_all = np.asarray(
            [int(float(r[self._cfg.target_column])) for r in train_rows], dtype=np.int32
        )
        x_all = encode_features(train_rows)

        # Split train/valid (stratified)
        train_idx, valid_idx = _stratified_split_binary(
            y=y_all, valid_fraction=valid_fraction, seed=seed
        )

        self._x_train = x_all[train_idx]
        self._y_train = y_all[train_idx]
        self._x_valid = x_all[valid_idx]
        self._y_valid = y_all[valid_idx]
        self._ids_valid = ids_train[valid_idx]

        # Encode Kaggle test (no labels)
        self._ids_test = np.asarray(
            [int(r[self._cfg.id_column]) for r in test_rows], dtype=np.int64
        )
        self._x_test = encode_features(test_rows)

        self._info = DatasetInfo(
            num_classes=2,
            input_shape=(int(self._x_train.shape[1]),),
            train_size=int(self._x_train.shape[0]),
            valid_size=int(self._x_valid.shape[0]),
            test_size=int(self._x_test.shape[0]),
        )

    @property
    def numeric_dim(self) -> int:
        return int(self._numeric_dim)

    @property
    def categorical_cols(self) -> tuple[str, ...]:
        return tuple(self._schema.get("categorical_cols", []))

    @property
    def categorical_cardinalities(self) -> tuple[int, ...]:
        cols = list(self._schema.get("categorical_cols", []))
        vocab = self._schema.get("cat_vocab", {})
        return tuple(int(len(vocab[c])) for c in cols)

    @property
    def categorical_encoding(self) -> str:
        return str(self._cfg.categorical_encoding)

    def describe(self) -> dict[str, Any]:
        """Human/debug friendly summary (useful for CLI logging)."""

        ytr = np.asarray(self._y_train)
        yva = np.asarray(self._y_valid)
        pos_rate_train = float(ytr.mean()) if ytr.size else float("nan")
        pos_rate_valid = float(yva.mean()) if yva.size else float("nan")

        cat_card = {
            c: int(len(self._schema["cat_vocab"][c]))
            for c in self._schema["categorical_cols"]
        }

        return {
            "train_size": int(self._info.train_size),
            "valid_size": int(self._info.valid_size),
            "test_size": int(self._info.test_size),
            "n_features": int(self._info.input_shape[0]),
            "n_numeric": int(len(self._schema["numeric_cols"])),
            "n_categorical": int(len(self._schema["categorical_cols"])),
            "categorical_cols": list(self._schema["categorical_cols"]),
            "categorical_cardinalities": cat_card,
            "categorical_encoding": str(self._cfg.categorical_encoding),
            "pos_rate_train": pos_rate_train,
            "pos_rate_valid": pos_rate_valid,
        }

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
            x_batch = x[sel]

            # Optional augmentation: add Gaussian noise to *numeric* features only, on the training split.
            # (Categoricals are one-hot and should not be noised; validation should remain clean.)
            if (
                self._add_noise
                and split == "train"
                and self._numeric_dim > 0
                and self._noise_std > 0.0
            ):
                rng = np.random.default_rng(seed + start)  # Deterministic per batch
                noise = rng.normal(
                    0.0, self._noise_std, (x_batch.shape[0], self._numeric_dim)
                ).astype(x_batch.dtype)
                x_batch = x_batch.copy()
                x_batch[:, : self._numeric_dim] = (
                    x_batch[:, : self._numeric_dim] + noise
                )
            yield Batch(x=x_batch, y=y[sel])

    def get_validation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ids_valid, x_valid, y_valid)."""

        return self._ids_valid, self._x_valid, self._y_valid

    def get_inference(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids_test, x_test) for Kaggle submission."""

        return self._ids_test, self._x_test


class TabularCsvMulticlassClassificationDatasetProvider(DatasetProviderPort):
    """Multi-class classification dataset from train.csv/test.csv (Kaggle/Zindi style).

    - Handles mixed numeric + categorical columns.
    - One-hot encodes categoricals using vocab built from (train + test) for stability.
    - Standardizes numeric columns using train statistics.

    Supports string targets by mapping to integer class ids.

    Splits:
      - "train": training subset
      - "test": validation subset (labels present)

    Inference:
      - call `get_inference()` to get (ids_test, X_test) for submission.
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
        add_noise: bool = False,
        noise_std: float = 0.1,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        self._cfg = config or TabularCsvConfig(enable_feature_engineering=False)
        self._add_noise = bool(add_noise)
        self._noise_std = float(noise_std)
        self._numeric_dim = 0

        train_rows = _read_csv_rows(train_csv_path)
        test_rows = _read_csv_rows(test_csv_path)

        if self._cfg.enable_feature_engineering:
            _apply_feature_engineering(train_rows, cfg=self._cfg)
            _apply_feature_engineering(test_rows, cfg=self._cfg)

        if max_train_rows is not None:
            train_rows = train_rows[:max_train_rows]
        if max_test_rows is not None:
            test_rows = test_rows[:max_test_rows]

        if not train_rows:
            raise ValueError(f"No rows found in {train_csv_path}")
        if not test_rows:
            raise ValueError(f"No rows found in {test_csv_path}")

        all_cols = list(train_rows[0].keys())
        if self._cfg.id_column not in all_cols:
            raise ValueError(f"Missing id column '{self._cfg.id_column}'")
        if self._cfg.target_column not in all_cols:
            raise ValueError(f"Missing target column '{self._cfg.target_column}'")

        feature_cols = [
            c
            for c in all_cols
            if c not in (self._cfg.id_column, self._cfg.target_column)
        ]

        # Infer numeric vs categorical based on train rows
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for c in feature_cols:
            values = [r.get(c, "") for r in train_rows]
            non_missing = [v for v in values if not _is_missing(v, cfg=self._cfg)]
            if not non_missing:
                categorical_cols.append(c)
                continue

            n_ok = sum(1 for v in non_missing if _can_float(v))
            frac_ok = n_ok / max(1, len(non_missing))
            if frac_ok >= self._cfg.numeric_min_fraction:
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)

        # Build categorical vocab from train+test so unseen test categories don't break
        cat_vocab: dict[str, dict[str, int]] = {}
        for c in categorical_cols:
            vals: list[str] = []
            for r in train_rows:
                v = r.get(c, "")
                vals.append(
                    self._cfg.categorical_missing_token
                    if _is_missing(v, cfg=self._cfg)
                    else v
                )
            for r in test_rows:
                v = r.get(c, "")
                vals.append(
                    self._cfg.categorical_missing_token
                    if _is_missing(v, cfg=self._cfg)
                    else v
                )
            uniq = sorted(set(vals))
            cat_vocab[c] = {v: i for i, v in enumerate(uniq)}

        # Compute numeric stats from train
        num_means: dict[str, float] = {}
        num_stds: dict[str, float] = {}
        for c in numeric_cols:
            arr = np.asarray(
                [_to_float_or_nan(r.get(c, ""), cfg=self._cfg) for r in train_rows],
                dtype=np.float32,
            )
            mu = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            if np.isnan(mu):
                mu = 0.0
            if np.isnan(std) or std == 0.0:
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

        self._numeric_dim = int(len(numeric_cols))

        def encode_features(rows: list[dict[str, str]]) -> np.ndarray:
            feats: list[np.ndarray] = []

            if numeric_cols:
                num = np.stack(
                    [
                        (
                            np.nan_to_num(
                                np.asarray(
                                    [
                                        _to_float_or_nan(r.get(c, ""), cfg=self._cfg)
                                        for r in rows
                                    ],
                                    dtype=np.float32,
                                ),
                                nan=num_means[c],
                            )
                            - num_means[c]
                        )
                        / num_stds[c]
                        for c in numeric_cols
                    ],
                    axis=1,
                )
                feats.append(num)

            for c in categorical_cols:
                vocab = cat_vocab[c]
                idx = np.asarray(
                    [
                        vocab[
                            (
                                self._cfg.categorical_missing_token
                                if _is_missing(r.get(c, ""), cfg=self._cfg)
                                else r.get(c, "")
                            )
                        ]
                        for r in rows
                    ],
                    dtype=np.int32,
                )
                enc = (self._cfg.categorical_encoding or "onehot").strip().lower()
                if enc == "index":
                    feats.append(idx.reshape(-1, 1).astype(np.float32))
                else:
                    one_hot = np.zeros((len(rows), len(vocab)), dtype=np.float32)
                    one_hot[np.arange(len(rows)), idx] = 1.0
                    feats.append(one_hot)

            if not feats:
                raise ValueError("No features detected")
            return np.concatenate(feats, axis=1).astype(np.float32)

        # IDs (kept as strings to support non-numeric IDs like 'ID_XXXX')
        ids_train = np.asarray(
            [str(r[self._cfg.id_column]) for r in train_rows], dtype=object
        )

        # Targets: build label mapping
        y_raw = [str(r.get(self._cfg.target_column, "")).strip() for r in train_rows]
        uniq_labels = sorted(set(y_raw))
        if class_names is not None:
            ordered = [c for c in class_names if c in uniq_labels]
            missing = [c for c in uniq_labels if c not in ordered]
            if missing:
                ordered.extend(missing)
            uniq_labels = ordered
        else:
            # Common Zindi convention for this competition
            tri = {"Low", "Medium", "High"}
            if set(uniq_labels) == tri:
                uniq_labels = ["Low", "Medium", "High"]

        label_to_int = {lab: i for i, lab in enumerate(uniq_labels)}
        self._class_names = tuple(uniq_labels)
        y_all = np.asarray([label_to_int[lab] for lab in y_raw], dtype=np.int32)

        x_all = encode_features(train_rows)

        train_idx, valid_idx = _stratified_split_multiclass(
            y=y_all, valid_fraction=valid_fraction, seed=seed
        )

        self._x_train = x_all[train_idx]
        self._y_train = y_all[train_idx]
        self._x_valid = x_all[valid_idx]
        self._y_valid = y_all[valid_idx]
        self._ids_valid = ids_train[valid_idx]

        self._ids_test = np.asarray(
            [str(r[self._cfg.id_column]) for r in test_rows], dtype=object
        )
        self._x_test = encode_features(test_rows)

        self._info = DatasetInfo(
            num_classes=int(len(self._class_names)),
            input_shape=(int(self._x_train.shape[1]),),
            train_size=int(self._x_train.shape[0]),
            valid_size=int(self._x_valid.shape[0]),
            test_size=int(self._x_test.shape[0]),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    @property
    def numeric_dim(self) -> int:
        return int(self._numeric_dim)

    @property
    def categorical_cols(self) -> tuple[str, ...]:
        return tuple(self._schema.get("categorical_cols", []))

    @property
    def categorical_cardinalities(self) -> tuple[int, ...]:
        cols = list(self._schema.get("categorical_cols", []))
        vocab = self._schema.get("cat_vocab", {})
        return tuple(int(len(vocab[c])) for c in cols)

    def describe(self) -> dict[str, Any]:
        ytr = np.asarray(self._y_train)
        yva = np.asarray(self._y_valid)
        cat_card = {
            c: int(len(self._schema["cat_vocab"][c]))
            for c in self._schema["categorical_cols"]
        }

        def dist(y: np.ndarray) -> dict[str, int]:
            out: dict[str, int] = {name: 0 for name in self._class_names}
            if y.size:
                vals, cnts = np.unique(y, return_counts=True)
                for v, c in zip(vals.tolist(), cnts.tolist()):
                    out[self._class_names[int(v)]] = int(c)
            return out

        return {
            "train_size": int(self._info.train_size),
            "valid_size": int(self._info.valid_size),
            "test_size": int(self._info.test_size),
            "n_features": int(self._info.input_shape[0]),
            "n_numeric": int(len(self._schema["numeric_cols"])),
            "n_categorical": int(len(self._schema["categorical_cols"])),
            "categorical_cols": list(self._schema["categorical_cols"]),
            "categorical_cardinalities": cat_card,
            "categorical_encoding": str(self._cfg.categorical_encoding),
            "class_names": list(self._class_names),
            "label_dist_train": dist(ytr),
            "label_dist_valid": dist(yva),
        }

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
            x, y = self._x_valid, self._y_valid

        n = len(x)
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)

        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            x_batch = x[sel]

            if (
                self._add_noise
                and split == "train"
                and self._numeric_dim > 0
                and self._noise_std > 0.0
            ):
                rng = np.random.default_rng(seed + start)
                noise = rng.normal(
                    0.0, self._noise_std, (x_batch.shape[0], self._numeric_dim)
                ).astype(x_batch.dtype)
                x_batch = x_batch.copy()
                x_batch[:, : self._numeric_dim] = (
                    x_batch[:, : self._numeric_dim] + noise
                )

            yield Batch(x=x_batch, y=y[sel])

    def get_validation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._ids_valid, self._x_valid, self._y_valid

    def get_inference(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ids_test, self._x_test
