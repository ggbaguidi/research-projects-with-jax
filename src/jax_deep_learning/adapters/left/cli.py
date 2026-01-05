from __future__ import annotations

import csv
import os
import warnings
import zipfile
from dataclasses import asdict
from pathlib import Path

import inject
import jax
import jax.numpy as jnp
import numpy as np
import typer

# Default to CPU unless explicitly overridden by the user.
# This avoids noisy CUDA plugin initialization errors on machines without CUDA libraries.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Reduce known-noisy warning coming from TF/Keras in some environments.
warnings.filterwarnings(
    "ignore",
    message=r"In the future `np\.object` will be defined as the corresponding NumPy scalar\.",
    category=FutureWarning,
)

from jax_deep_learning.adapters.left.inject_config import configure_injections
from jax_deep_learning.adapters.right.checkpoints_filesystem import \
    FilesystemCheckpointStore
from jax_deep_learning.adapters.right.data_loaders import (
    NpzClassificationDatasetProvider,
    TabularCsvBinaryClassificationDatasetProvider, TabularCsvConfig,
    TabularCsvMulticlassClassificationDatasetProvider,
    TfdsClassificationDatasetProvider)
from jax_deep_learning.adapters.right.metrics_jsonl import (
    CompositeMetricsSink, JsonlFileMetricsSink)
from jax_deep_learning.adapters.right.metrics_plotting import \
    plot_metrics_from_logs
from jax_deep_learning.adapters.right.metrics_stdout import StdoutMetricsSink
from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.model import (
    DerfMlpClassifierFns, MlpClassifierFns, TabularEmbedMlpClassifierFns)
from jax_deep_learning.core.domain.utils.metrics import roc_auc_score_binary
from jax_deep_learning.core.ports.checkpoint_store import CheckpointStorePort
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.ports.metrics_sink import MetricsSinkPort
from jax_deep_learning.core.use_cases.train_classifier import \
    TrainClassifierUseCase

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command(name="plot-metrics")
def plot_metrics(
    log_paths: list[Path] = typer.Argument(
        ...,
        help="One or more JSONL log files written by --log-path (e.g. logs/run.jsonl)",
    ),
    out_path: str = typer.Option(
        "metrics.png",
        help="Where to save the plot image (PNG). Ignored if --show is used without saving.",
    ),
    show: bool = typer.Option(
        False,
        "--show/--no-show",
        help="Show an interactive window (requires a GUI backend).",
    ),
    x_axis: str = typer.Option(
        "step",
        help="X axis to use: step | global_step | epoch",
    ),
    metrics: list[str] = typer.Option(
        [],
        "--metric",
        help="Repeatable metric keys to plot (e.g. --metric train/loss --metric test/loss). Defaults to common metrics.",
    ),
    group_by: str = typer.Option(
        "suffix",
        help="How to group plots: suffix (loss/acc/auc) | none (one subplot per metric)",
    ),
    title: str = typer.Option(
        "",
        help="Optional figure title",
    ),
) -> None:
    """Plot training metrics from JSONL logs."""

    # If user only wants to show, allow out_path to be empty.
    out: str | None = out_path.strip() if out_path.strip() else None

    saved = plot_metrics_from_logs(
        log_paths=log_paths,
        out_path=out,
        show=bool(show),
        x_axis=x_axis,
        metrics=metrics if metrics else None,
        group_by=group_by,
        title=title.strip() or None,
    )
    if saved is not None:
        typer.echo(f"Saved plot to: {saved}")


def _confusion_matrix_counts(
    *, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """Return integer confusion matrix of shape (C, C) for labels in [0, C).

    Rows are true labels, columns are predicted labels.
    """

    yt = np.asarray(y_true, dtype=np.int32).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have same length, got {yt.shape[0]} vs {yp.shape[0]}"
        )
    if n_classes <= 0:
        raise ValueError(f"n_classes must be > 0, got {n_classes}")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # Only count valid labels (guards against accidental -1 or out-of-range values).
    mask = (yt >= 0) & (yt < n_classes) & (yp >= 0) & (yp < n_classes)
    if np.any(mask):
        np.add.at(cm, (yt[mask], yp[mask]), 1)
    return cm


def _print_confusion_matrix(*, cm: np.ndarray, class_names: tuple[str, ...]) -> None:
    """Pretty-print confusion matrix to stdout.

    Shows both counts and row-normalized percentages.
    """

    cm = np.asarray(cm)
    n = int(cm.shape[0])
    if cm.shape != (n, n):
        raise ValueError(f"cm must be square, got shape={cm.shape}")
    if len(class_names) != n:
        raise ValueError(
            f"class_names length must match cm size, got {len(class_names)} vs {n}"
        )

    name_w = max(4, max(len(str(x)) for x in class_names))
    cell_w = max(7, max(len(str(int(x))) for x in cm.reshape(-1).tolist()) + 1)

    typer.echo("Confusion matrix (rows=true, cols=pred):")
    header = " " * (name_w + 2) + " ".join(f"{name:>{cell_w}}" for name in class_names)
    typer.echo(header)
    for i, name in enumerate(class_names):
        row = cm[i, :]
        row_str = " ".join(f"{int(v):>{cell_w}d}" for v in row.tolist())
        typer.echo(f"{name:>{name_w}} | {row_str}")

    typer.echo("Row-normalized (%):")
    typer.echo(header)
    for i, name in enumerate(class_names):
        row = cm[i, :].astype(np.float64)
        denom = float(row.sum())
        if denom <= 0:
            pct = np.zeros_like(row)
        else:
            pct = 100.0 * (row / denom)
        row_str = " ".join(f"{p:>{cell_w}.1f}" for p in pct.tolist())
        typer.echo(f"{name:>{name_w}} | {row_str}")


@app.command()
def train(
    dataset_kind: str = typer.Option("tfds", help="Dataset adapter to use: tfds | npz"),
    tfds_name: str = typer.Option(
        "mnist", help="TFDS dataset name (when dataset_kind=tfds)"
    ),
    tfds_data_dir: str = typer.Option(
        "/tmp/tfds", help="TFDS cache directory (when dataset_kind=tfds)"
    ),
    npz_path: str = typer.Option("", help="Path to .npz (when dataset_kind=npz)"),
    epochs: int = typer.Option(10, min=1),
    batch_size: int = typer.Option(32, min=1),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(0.0),
    adamw_b1: float = typer.Option(0.9, min=0.0, max=1.0, help="AdamW beta1"),
    adamw_b2: float = typer.Option(0.999, min=0.0, max=1.0, help="AdamW beta2"),
    adamw_eps: float = typer.Option(1e-8, min=0.0, help="AdamW eps"),
    adamw_eps_root: float = typer.Option(0.0, min=0.0, help="AdamW eps_root"),
    adamw_nesterov: bool = typer.Option(
        False,
        "--adamw-nesterov/--no-adamw-nesterov",
        help="Use Nesterov momentum in AdamW",
    ),
    hidden: list[int] = typer.Option(
        [512, 256], help="Repeatable hidden sizes: --hidden 512 --hidden 256"
    ),
    model_kind: str = typer.Option(
        "mlp",
        help="Model to use: mlp | derf-mlp",
    ),
    remat: bool = typer.Option(
        False,
        "--remat/--no-remat",
        help="Use jax.checkpoint per layer (helps deep nets fit in memory; slower)",
    ),
    seed: int = typer.Option(0),
    ckpt_dir: str = typer.Option("", help="If set, save checkpoints to this folder"),
    log_path: str = typer.Option(
        "",
        help="If set, append metrics/events as JSONL to this path (e.g. logs/train.jsonl)",
    ),
    cpu: bool = typer.Option(
        True,
        "--cpu/--no-cpu",
        help="Force CPU (recommended on machines without CUDA libs)",
    ),
    early_stopping_patience: int = typer.Option(
        0,
        min=0,
        help="If >0 and binary classification, stop when valid AUC hasn't improved for N epochs",
    ),
) -> None:
    """Train a simple classifier using the core training use case."""

    if cpu and os.environ.get("JAX_PLATFORMS") != "cpu":
        typer.echo("Note: set JAX_PLATFORMS=cpu before running to force CPU.")

    dataset_kind = dataset_kind.lower().strip()

    if dataset_kind == "tfds":
        dataset = TfdsClassificationDatasetProvider(
            name=tfds_name, data_dir=tfds_data_dir
        )
    elif dataset_kind == "npz":
        if not npz_path:
            raise typer.BadParameter("--npz-path is required when dataset_kind=npz")
        dataset = NpzClassificationDatasetProvider(path=npz_path)
    else:
        raise typer.BadParameter("dataset_kind must be one of: tfds, npz")

    ckpt = FilesystemCheckpointStore(dir_path=ckpt_dir) if ckpt_dir else None

    stdout_metrics = StdoutMetricsSink()
    metrics = (
        CompositeMetricsSink(stdout_metrics, JsonlFileMetricsSink(path=log_path))
        if log_path
        else stdout_metrics
    )

    model_kind = model_kind.lower().strip()
    if model_kind == "mlp":
        model = MlpClassifierFns(hidden_sizes=tuple(hidden), remat=bool(remat))
    elif model_kind in {"derf", "derf-mlp"}:
        model = DerfMlpClassifierFns(hidden_sizes=tuple(hidden), remat=bool(remat))
    else:
        raise typer.BadParameter("model_kind must be one of: mlp, derf-mlp")

    configure_injections(
        dataset_provider=dataset,
        checkpoint_store=ckpt,
        metrics_sink=metrics,
        model_fns=model,
    )

    cmd = TrainCommand(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=lr,
        weight_decay=weight_decay,
        adamw_b1=adamw_b1,
        adamw_b2=adamw_b2,
        adamw_eps=adamw_eps,
        adamw_eps_root=adamw_eps_root,
        adamw_nesterov=adamw_nesterov,
        hidden_sizes=tuple(hidden),
        log_every_steps=100,
        early_stopping_patience=early_stopping_patience,
    )

    use_case = inject.instance(TrainClassifierUseCase)

    metrics.log(
        step=0,
        metrics={
            "event": "run_start",
            "command": "train",
            "dataset_kind": dataset_kind,
            "model_kind": model_kind,
            "remat": bool(remat),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "adamw/b1": adamw_b1,
            "adamw/b2": adamw_b2,
            "adamw/eps": adamw_eps,
            "adamw/eps_root": adamw_eps_root,
            "adamw/nesterov": bool(adamw_nesterov),
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
        },
    )

    result = use_case.run(cmd)
    typer.echo("Training complete")
    typer.echo(f"Final epoch summary: {result.history[-1] if result.history else {}}")
    typer.echo(f"Train command: {asdict(cmd)}")


@app.command(name="kaggle-diabetes")
def kaggle_diabetes(
    data_dir: str = typer.Option(
        "data/s5e12", help="Folder containing train.csv and test.csv"
    ),
    out_path: str = typer.Option(
        "submission.csv", help="Where to write the submission file"
    ),
    valid_fraction: float = typer.Option(
        0.2, min=0.01, max=0.5, help="Holdout fraction from train.csv"
    ),
    epochs: int = typer.Option(10, min=1),
    batch_size: int = typer.Option(256, min=1),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(0.0),
    adamw_b1: float = typer.Option(0.9, min=0.0, max=1.0, help="AdamW beta1"),
    adamw_b2: float = typer.Option(0.999, min=0.0, max=1.0, help="AdamW beta2"),
    adamw_eps: float = typer.Option(1e-8, min=0.0, help="AdamW eps"),
    adamw_eps_root: float = typer.Option(0.0, min=0.0, help="AdamW eps_root"),
    adamw_nesterov: bool = typer.Option(
        False,
        "--adamw-nesterov/--no-adamw-nesterov",
        help="Use Nesterov momentum in AdamW",
    ),
    hidden: list[int] = typer.Option(
        [2025, 128, 64], help="Repeatable hidden sizes: --hidden 256 --hidden 128"
    ),
    embed_dim: int = typer.Option(8, min=1, help="Embedding dim for tabular-embed-mlp"),
    model_kind: str = typer.Option(
        "derf-mlp",
        help="Model to use: derf-mlp (default) | mlp | tabular-embed-mlp",
    ),
    remat: bool = typer.Option(
        False,
        "--remat/--no-remat",
        help="Use jax.checkpoint per layer (helps deep nets fit in memory; slower)",
    ),
    seed: int = typer.Option(0),
    cpu: bool = typer.Option(
        True,
        "--cpu/--no-cpu",
        help="Force CPU (recommended on machines without CUDA libs)",
    ),
    max_train_rows: int = typer.Option(
        0, help="If >0, only use the first N train rows (quick test)"
    ),
    max_test_rows: int = typer.Option(
        0, help="If >0, only use the first N test rows (quick test)"
    ),
    log_path: str = typer.Option(
        "logs/kaggle_diabetes.jsonl",
        help="Append metrics/events as JSONL to this file (set to '' to disable)",
    ),
    early_stopping_patience: int = typer.Option(
        0,
        min=0,
        help="If >0, stop when valid AUC hasn't improved for N epochs (recommended)",
    ),
    add_noise: bool = typer.Option(
        False,
        "--add-noise/--no-add-noise",
        help="Add Gaussian noise to numeric features for augmentation",
    ),
    noise_std: float = typer.Option(
        0.1, min=0.0, help="Std dev of noise for augmentation (if --add-noise)"
    ),
    zip_output: bool = typer.Option(
        False, "--zip/--no-zip", help="If set, zip the output submission file"
    ),
) -> None:
    """Train and generate a Kaggle submission for the diabetes playground dataset.

    Expects:
      - data/s5e12/train.csv with target column 'diagnosed_diabetes'
      - data/s5e12/test.csv without the target
    Writes:
      - submission.csv with columns: id,diagnosed_diabetes
    """

    if cpu and os.environ.get("JAX_PLATFORMS") != "cpu":
        typer.echo("Note: set JAX_PLATFORMS=cpu before running to force CPU.")

    model_kind = model_kind.lower().strip()

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    # For embedding models, encode categoricals as indices.
    tab_cfg = TabularCsvConfig(
        categorical_encoding=(
            "index" if model_kind == "tabular-embed-mlp" else "onehot"
        )
    )

    dataset = TabularCsvBinaryClassificationDatasetProvider(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        valid_fraction=valid_fraction,
        seed=seed,
        max_train_rows=(max_train_rows or None),
        max_test_rows=(max_test_rows or None),
        config=tab_cfg,
        add_noise=add_noise,
        noise_std=noise_std,
    )

    out_path_p = Path(out_path)

    stdout_metrics = StdoutMetricsSink()
    metrics = (
        CompositeMetricsSink(stdout_metrics, JsonlFileMetricsSink(path=log_path))
        if log_path
        else stdout_metrics
    )

    if model_kind == "mlp":
        model = MlpClassifierFns(hidden_sizes=tuple(hidden), remat=bool(remat))
    elif model_kind in {"derf", "derf-mlp"}:
        model = DerfMlpClassifierFns(hidden_sizes=tuple(hidden), remat=bool(remat))
    elif model_kind in {"tabular-embed-mlp", "embed-mlp"}:
        model = TabularEmbedMlpClassifierFns(
            n_numeric=dataset.numeric_dim,
            categorical_cardinalities=dataset.categorical_cardinalities,
            embed_dim=int(embed_dim),
            hidden_sizes=tuple(hidden),
        )
    else:
        raise typer.BadParameter(
            "model_kind must be one of: derf-mlp, mlp, tabular-embed-mlp"
        )

    cmd = TrainCommand(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=lr,
        weight_decay=weight_decay,
        adamw_b1=adamw_b1,
        adamw_b2=adamw_b2,
        adamw_eps=adamw_eps,
        adamw_eps_root=adamw_eps_root,
        adamw_nesterov=adamw_nesterov,
        hidden_sizes=tuple(hidden),
        log_every_steps=50,
        early_stopping_patience=early_stopping_patience,
    )

    configure_injections(
        dataset_provider=dataset,
        checkpoint_store=None,
        metrics_sink=metrics,
        model_fns=model,
    )
    use_case = inject.instance(TrainClassifierUseCase)

    metrics.log(
        step=0,
        metrics={
            "event": "run_start",
            "command": "kaggle-diabetes",
            "data_dir": data_dir,
            "dataset": dataset.describe(),
            "model_kind": model_kind,
            "remat": bool(remat),
            "embed_dim": int(embed_dim),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "adamw/b1": adamw_b1,
            "adamw/b2": adamw_b2,
            "adamw/eps": adamw_eps,
            "adamw/eps_root": adamw_eps_root,
            "adamw/nesterov": bool(adamw_nesterov),
            "seed": seed,
            "valid_fraction": valid_fraction,
            "max_train_rows": max_train_rows,
            "max_test_rows": max_test_rows,
            "out_path": str(out_path_p),
            "early_stopping_patience": early_stopping_patience,
            "add_noise": bool(add_noise),
            "noise_std": noise_std,
        },
    )

    result = use_case.run(cmd)

    # Evaluate AUC on validation split
    ids_valid, x_valid, y_valid = dataset.get_validation()

    @jax.jit
    def predict_proba(params, x: jax.Array) -> jax.Array:
        logits = model.apply(params, x, is_training=False)
        probs = jax.nn.softmax(logits, axis=-1)
        return probs[:, 1]

    p_valid = np.asarray(predict_proba(result.params, jnp.asarray(x_valid)))
    auc = roc_auc_score_binary(y_valid, p_valid)
    typer.echo(f"Validation ROC AUC: {auc:.5f} (valid n={len(y_valid)})")

    global_step = int(result.history[-1].get("global_step", 0)) if result.history else 0
    metrics.log(
        step=global_step, metrics={"valid/auc": float(auc), "event": "valid_auc"}
    )

    # Predict Kaggle test and write submission
    ids_test, x_test = dataset.get_inference()
    p_test = np.asarray(predict_proba(result.params, jnp.asarray(x_test)))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "diagnosed_diabetes"])
        for i, p in zip(ids_test.tolist(), p_test.tolist()):
            w.writerow([int(i), float(p)])

    typer.echo(f"Wrote submission to: {out_path_p}")
    metrics.log(
        step=global_step,
        metrics={
            "event": "submission_written",
            "out_path": str(out_path_p),
            "n_test": int(len(ids_test)),
        },
    )
    typer.echo(f"Final epoch summary: {result.history[-1] if result.history else {}}")

    if zip_output:
        zip_path = out_path_p.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(out_path_p, arcname=out_path_p.name)
        typer.echo(f"Wrote zipped submission to: {zip_path}")
        metrics.log(
            step=global_step,
            metrics={"event": "submission_zipped", "zip_path": str(zip_path)},
        )


@app.command(name="zindi-financial-health")
def zindi_financial_health(
    data_dir: str = typer.Option(
        "data/data.org", help="Folder containing Train.csv and Test.csv"
    ),
    out_path: str = typer.Option(
        "submission_zindi_financial_health.csv",
        help="Where to write the submission file",
    ),
    valid_fraction: float = typer.Option(
        0.2, min=0.01, max=0.5, help="Holdout fraction from Train.csv"
    ),
    epochs: int = typer.Option(15, min=1),
    batch_size: int = typer.Option(256, min=1),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(0.0),
    adamw_b1: float = typer.Option(0.9, min=0.0, max=1.0, help="AdamW beta1"),
    adamw_b2: float = typer.Option(0.999, min=0.0, max=1.0, help="AdamW beta2"),
    adamw_eps: float = typer.Option(1e-8, min=0.0, help="AdamW eps"),
    adamw_eps_root: float = typer.Option(0.0, min=0.0, help="AdamW eps_root"),
    adamw_nesterov: bool = typer.Option(
        False,
        "--adamw-nesterov/--no-adamw-nesterov",
        help="Use Nesterov momentum in AdamW",
    ),
    hidden: list[int] = typer.Option(
        [256, 128], help="Repeatable hidden sizes: --hidden 256 --hidden 128"
    ),
    embed_dim: int = typer.Option(8, min=1, help="Embedding dim for tabular-embed-mlp"),
    model_kind: str = typer.Option(
        "derf-mlp",
        help="Model to use: derf-mlp (default) | mlp | tabular-embed-mlp",
    ),
    remat: bool = typer.Option(
        False,
        "--remat/--no-remat",
        help="Use jax.checkpoint per layer (helps deep nets fit in memory; slower)",
    ),
    seed: int = typer.Option(0),
    cpu: bool = typer.Option(
        True,
        "--cpu/--no-cpu",
        help="Force CPU (recommended on machines without CUDA libs)",
    ),
    max_train_rows: int = typer.Option(
        0, help="If >0, only use the first N train rows (quick test)"
    ),
    max_test_rows: int = typer.Option(
        0, help="If >0, only use the first N test rows (quick test)"
    ),
    log_path: str = typer.Option(
        "logs/zindi_financial_health.jsonl",
        help="Append metrics/events as JSONL to this file (set to '' to disable)",
    ),
    add_noise: bool = typer.Option(
        False,
        "--add-noise/--no-add-noise",
        help="Add Gaussian noise to numeric features for augmentation",
    ),
    noise_std: float = typer.Option(
        0.1, min=0.0, help="Std dev of noise for augmentation (if --add-noise)"
    ),
    zip_output: bool = typer.Option(
        False, "--zip/--no-zip", help="If set, zip the output submission file"
    ),
    confusion_matrix: bool = typer.Option(
        True,
        "--confusion-matrix/--no-confusion-matrix",
        help="Print a confusion matrix for the validation split",
    ),
    feature_engineering: bool = typer.Option(
        True,
        "--feature-engineering/--no-feature-engineering",
        help="Enable simple domain feature engineering (profit_margin, financial_access_score)",
    ),
) -> None:
    """Train and generate a Zindi submission for the data.org Financial Health challenge.

    Expects (default paths):
      - data/data.org/Train.csv with target column 'Target'
      - data/data.org/Test.csv without the target
    Writes:
      - submission CSV with columns: ID,Target (Target in {Low,Medium,High})
    """

    if cpu and os.environ.get("JAX_PLATFORMS") != "cpu":
        typer.echo("Note: set JAX_PLATFORMS=cpu before running to force CPU.")

    model_kind = model_kind.lower().strip()

    train_csv = os.path.join(data_dir, "Train.csv")
    test_csv = os.path.join(data_dir, "Test.csv")

    tab_cfg = TabularCsvConfig(
        id_column="ID",
        target_column="Target",
        enable_feature_engineering=bool(feature_engineering),
        feature_engineering_kind="zindi_financial_health",
        categorical_encoding=(
            "index" if model_kind == "tabular-embed-mlp" else "onehot"
        ),
    )

    dataset = TabularCsvMulticlassClassificationDatasetProvider(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        valid_fraction=valid_fraction,
        seed=seed,
        max_train_rows=(max_train_rows or None),
        max_test_rows=(max_test_rows or None),
        config=tab_cfg,
        add_noise=add_noise,
        noise_std=noise_std,
        class_names=("Low", "Medium", "High"),
    )

    out_path_p = Path(out_path)

    stdout_metrics = StdoutMetricsSink()
    metrics = (
        CompositeMetricsSink(stdout_metrics, JsonlFileMetricsSink(path=log_path))
        if log_path
        else stdout_metrics
    )

    if model_kind == "mlp":
        model = MlpClassifierFns(hidden_sizes=tuple(hidden), remat=bool(remat))
    elif model_kind in {"derf", "derf-mlp"}:
        model = DerfMlpClassifierFns(hidden_sizes=tuple(hidden), remat=bool(remat))
    elif model_kind in {"tabular-embed-mlp", "embed-mlp"}:
        model = TabularEmbedMlpClassifierFns(
            n_numeric=dataset.numeric_dim,
            categorical_cardinalities=dataset.categorical_cardinalities,
            embed_dim=int(embed_dim),
            hidden_sizes=tuple(hidden),
        )
    else:
        raise typer.BadParameter(
            "model_kind must be one of: derf-mlp, mlp, tabular-embed-mlp"
        )

    cmd = TrainCommand(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=lr,
        weight_decay=weight_decay,
        adamw_b1=adamw_b1,
        adamw_b2=adamw_b2,
        adamw_eps=adamw_eps,
        adamw_eps_root=adamw_eps_root,
        adamw_nesterov=adamw_nesterov,
        hidden_sizes=tuple(hidden),
        log_every_steps=50,
        early_stopping_patience=0,  # accuracy-only task
    )

    configure_injections(
        dataset_provider=dataset,
        checkpoint_store=None,
        metrics_sink=metrics,
        model_fns=model,
    )
    use_case = inject.instance(TrainClassifierUseCase)

    metrics.log(
        step=0,
        metrics={
            "event": "run_start",
            "command": "zindi-financial-health",
            "data_dir": data_dir,
            "dataset": dataset.describe(),
            "model_kind": model_kind,
            "remat": bool(remat),
            "embed_dim": int(embed_dim),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "adamw/b1": adamw_b1,
            "adamw/b2": adamw_b2,
            "adamw/eps": adamw_eps,
            "adamw/eps_root": adamw_eps_root,
            "adamw/nesterov": bool(adamw_nesterov),
            "seed": seed,
            "valid_fraction": valid_fraction,
            "max_train_rows": max_train_rows,
            "max_test_rows": max_test_rows,
            "out_path": str(out_path_p),
            "add_noise": bool(add_noise),
            "noise_std": noise_std,
        },
    )

    result = use_case.run(cmd)

    ids_valid, x_valid, y_valid = dataset.get_validation()

    @jax.jit
    def predict_class(params, x: jax.Array) -> jax.Array:
        logits = model.apply(params, x, is_training=False)
        return jnp.argmax(logits, axis=-1)

    y_hat_valid = np.asarray(predict_class(result.params, jnp.asarray(x_valid))).astype(
        np.int32
    )
    acc = float(np.mean(y_hat_valid == np.asarray(y_valid, dtype=np.int32)))
    typer.echo(f"Validation accuracy: {acc:.5f} (valid n={len(y_valid)})")

    if confusion_matrix:
        cm = _confusion_matrix_counts(
            y_true=np.asarray(y_valid, dtype=np.int32),
            y_pred=np.asarray(y_hat_valid, dtype=np.int32),
            n_classes=dataset.info.num_classes,
        )
        _print_confusion_matrix(cm=cm, class_names=dataset.class_names)

    global_step = int(result.history[-1].get("global_step", 0)) if result.history else 0
    metrics.log(
        step=global_step, metrics={"valid/acc": float(acc), "event": "valid_acc"}
    )

    ids_test, x_test = dataset.get_inference()
    y_hat_test = np.asarray(predict_class(result.params, jnp.asarray(x_test))).astype(
        np.int32
    )
    labels = dataset.class_names
    pred_labels = [labels[int(i)] for i in y_hat_test.tolist()]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        for _id, lab in zip(ids_test.tolist(), pred_labels):
            w.writerow([str(_id), str(lab)])

    typer.echo(f"Wrote submission to: {out_path_p}")
    metrics.log(
        step=global_step,
        metrics={
            "event": "submission_written",
            "out_path": str(out_path_p),
            "n_test": int(len(ids_test)),
        },
    )
    typer.echo(f"Final epoch summary: {result.history[-1] if result.history else {}}")

    if zip_output:
        zip_path = out_path_p.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(out_path_p, arcname=out_path_p.name)
        typer.echo(f"Wrote zipped submission to: {zip_path}")
        metrics.log(
            step=global_step,
            metrics={"event": "submission_zipped", "zip_path": str(zip_path)},
        )
