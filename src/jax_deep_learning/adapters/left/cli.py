from __future__ import annotations

import csv
import os
import warnings
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

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


def _f1s_and_balanced_accuracy(
    *, cm: np.ndarray
) -> tuple[float, float, float]:
    """Return (macro_f1, weighted_f1, balanced_accuracy) from a confusion matrix.

    - macro_f1: mean F1 across classes
    - balanced_accuracy: mean recall across classes

    This is often more informative than plain accuracy for imbalanced multiclass
    datasets (like Low/Medium/High here).
    """

    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square, got shape={cm.shape}")

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    macro_f1 = float(np.mean(f1))
    support = np.sum(cm, axis=1)
    denom = float(np.sum(support))
    weighted_f1 = float(np.sum(f1 * support) / denom) if denom > 0 else 0.0
    bal_acc = float(np.mean(recall))
    return macro_f1, weighted_f1, bal_acc


def _read_csv_dicts(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = [dict(row) for row in r]
    if not fieldnames:
        raise ValueError(f"No header found in CSV: {path}")
    return fieldnames, rows


def _write_csv_dicts(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _stratified_kfold_indices(
    *, y: np.ndarray, n_splits: int, seed: int
) -> list[np.ndarray]:
    """Return list of validation indices, one array per fold.

    Simple stratified K-fold splitter without sklearn.
    """

    y = np.asarray(y, dtype=np.int32).reshape(-1)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    rng = np.random.default_rng(int(seed))
    folds: list[list[int]] = [[] for _ in range(int(n_splits))]

    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts)) if counts.size else 0
    if min_count < n_splits:
        raise ValueError(
            "Not enough samples in the smallest class for stratified K-fold: "
            f"min_class_count={min_count} < n_splits={n_splits}"
        )

    for cls in classes.tolist():
        idx = np.where(y == int(cls))[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx.tolist()):
            folds[i % n_splits].append(int(j))

    return [np.asarray(sorted(f), dtype=np.int32) for f in folds]


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
    loss_kind: str = typer.Option(
        "softmax",
        help="Loss to use: softmax | ordinal | ordinal-rank (ordinal is best for ordered labels)",
    ),
    ordinal_rank_lambda: float = typer.Option(
        0.05,
        min=0.0,
        help="Ranking regularizer weight (only used for loss_kind=ordinal-rank)",
    ),
    ordinal_rank_margin: float = typer.Option(
        0.25,
        min=0.0,
        help="Ranking margin (only used for loss_kind=ordinal-rank)",
    ),
    ordinal_rank_pairs_per_batch: int = typer.Option(
        256,
        min=0,
        help="Pairs to sample per batch for ranking (0 => all pairs; only used for loss_kind=ordinal-rank)",
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
        loss_kind=str(loss_kind),
        ordinal_rank_lambda=float(ordinal_rank_lambda),
        ordinal_rank_margin=float(ordinal_rank_margin),
        ordinal_rank_pairs_per_batch=int(ordinal_rank_pairs_per_batch),
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
            "loss_kind": str(loss_kind),
            "ordinal_rank_lambda": float(ordinal_rank_lambda),
            "ordinal_rank_margin": float(ordinal_rank_margin),
            "ordinal_rank_pairs_per_batch": int(ordinal_rank_pairs_per_batch),
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
    loss_kind: str = typer.Option(
        "ordinal-rank",
        help="Loss to use: softmax | ordinal | ordinal-rank (recommended for Low/Medium/High)",
    ),
    ordinal_rank_lambda: float = typer.Option(
        0.031,
        min=0.0,
        help="Ranking regularizer weight (only used for loss_kind=ordinal-rank)",
    ),
    ordinal_rank_margin: float = typer.Option(
        0.25,
        min=0.0,
        help="Ranking margin (only used for loss_kind=ordinal-rank)",
    ),
    ordinal_rank_pairs_per_batch: int = typer.Option(
        1024,
        min=0,
        help="Pairs to sample per batch for ranking (0 => all pairs; only used for loss_kind=ordinal-rank)",
    ),
    early_stopping_patience: int = typer.Option(
        0,
        min=0,
        help="If >0, stop when validation macro-F1 hasn't improved for N epochs",
    ),
    early_stopping_metric: str = typer.Option(
        "macro_f1",
        help="Multiclass early stopping metric: macro_f1 | weighted_f1",
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
        early_stopping_patience=int(early_stopping_patience),
        multiclass_early_stopping_metric=str(early_stopping_metric),
        loss_kind=str(loss_kind),
        ordinal_rank_lambda=float(ordinal_rank_lambda),
        ordinal_rank_margin=float(ordinal_rank_margin),
        ordinal_rank_pairs_per_batch=int(ordinal_rank_pairs_per_batch),
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
            "loss_kind": str(loss_kind),
            "ordinal_rank_lambda": float(ordinal_rank_lambda),
            "ordinal_rank_margin": float(ordinal_rank_margin),
            "ordinal_rank_pairs_per_batch": int(ordinal_rank_pairs_per_batch),
            "early_stopping_patience": int(early_stopping_patience),
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

    cm: np.ndarray | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None
    bal_acc: float | None = None

    if confusion_matrix:
        cm = _confusion_matrix_counts(
            y_true=np.asarray(y_valid, dtype=np.int32),
            y_pred=np.asarray(y_hat_valid, dtype=np.int32),
            n_classes=dataset.info.num_classes,
        )
        _print_confusion_matrix(cm=cm, class_names=dataset.class_names)

        macro_f1, weighted_f1, bal_acc = _f1s_and_balanced_accuracy(cm=cm)
        typer.echo(f"Validation macro-F1: {macro_f1:.5f}")
        typer.echo(f"Validation weighted-F1: {weighted_f1:.5f}")
        typer.echo(f"Validation balanced accuracy: {bal_acc:.5f}")

    global_step = int(result.history[-1].get("global_step", 0)) if result.history else 0
    metrics.log(
        step=global_step,
        metrics={
            "valid/acc": float(acc),
            "valid/macro_f1": float(macro_f1) if macro_f1 is not None else None,
            "valid/weighted_f1": float(weighted_f1)
            if weighted_f1 is not None
            else None,
            "valid/bal_acc": float(bal_acc) if bal_acc is not None else None,
            "event": "valid_metrics",
        },
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


@app.command(name="zindi-financial-health-cv")
def zindi_financial_health_cv(
    data_dir: str = typer.Option(
        "data/data.org", help="Folder containing Train.csv and Test.csv"
    ),
    out_path: str = typer.Option(
        "submission_zindi_financial_health_cv.csv",
        help="Where to write the ensemble submission file",
    ),
    n_folds: int = typer.Option(5, min=2, max=20, help="Number of CV folds"),
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
    log_dir: str = typer.Option(
        "logs/zindi_financial_health_cv",
        help="Folder to write fold JSONL logs and fold artifacts",
    ),
    zip_output: bool = typer.Option(
        False, "--zip/--no-zip", help="If set, zip the output submission file"
    ),
    feature_engineering: bool = typer.Option(
        True,
        "--feature-engineering/--no-feature-engineering",
        help="Enable simple domain feature engineering (profit_margin, financial_access_score)",
    ),
    loss_kind: str = typer.Option(
        "ordinal-rank",
        help="Loss to use: softmax | ordinal | ordinal-rank (recommended for Low/Medium/High)",
    ),
    ordinal_rank_lambda: float = typer.Option(
        0.05,
        min=0.0,
        help="Ranking regularizer weight (only used for loss_kind=ordinal-rank)",
    ),
    ordinal_rank_margin: float = typer.Option(
        0.25,
        min=0.0,
        help="Ranking margin (only used for loss_kind=ordinal-rank)",
    ),
    ordinal_rank_pairs_per_batch: int = typer.Option(
        256,
        min=0,
        help="Pairs to sample per batch for ranking (0 => all pairs; only used for loss_kind=ordinal-rank)",
    ),
    early_stopping_patience: int = typer.Option(
        3,
        min=0,
        help="If >0, stop when validation macro-F1 hasn't improved for N epochs",
    ),
    early_stopping_metric: str = typer.Option(
        "macro_f1",
        help="Multiclass early stopping metric: macro_f1 | weighted_f1",
    ),
    ensemble_method: str = typer.Option(
        "mean_logp",
        help="Ensemble aggregation: mean_prob | mean_logp (geometric mean via log-probs)",
    ),
) -> None:
    """Stratified K-fold CV + probability ensembling for the Zindi financial health task.

    Writes an ensemble submission using the mean predicted probability across folds.
    """

    if cpu and os.environ.get("JAX_PLATFORMS") != "cpu":
        typer.echo("Note: set JAX_PLATFORMS=cpu before running to force CPU.")

    model_kind = model_kind.lower().strip()
    train_csv = os.path.join(data_dir, "Train.csv")
    test_csv = os.path.join(data_dir, "Test.csv")

    # Read full Train.csv once to build folds.
    fieldnames, train_rows = _read_csv_dicts(train_csv)
    if "Target" not in fieldnames:
        raise typer.BadParameter("Train.csv must contain a 'Target' column")
    if "ID" not in fieldnames:
        raise typer.BadParameter("Train.csv must contain an 'ID' column")

    y_str = [str(r.get("Target", "")).strip() for r in train_rows]
    label_order = ("Low", "Medium", "High")
    label_to_int = {k: i for i, k in enumerate(label_order)}
    if not set(y_str).issubset(set(label_order)):
        raise typer.BadParameter(
            f"Unexpected Target labels in Train.csv: {sorted(set(y_str))}. Expected {list(label_order)}"
        )
    y_all = np.asarray([label_to_int[v] for v in y_str], dtype=np.int32)
    fold_valid_indices = _stratified_kfold_indices(
        y=y_all, n_splits=int(n_folds), seed=int(seed)
    )

    tab_cfg = TabularCsvConfig(
        id_column="ID",
        target_column="Target",
        enable_feature_engineering=bool(feature_engineering),
        feature_engineering_kind="zindi_financial_health",
        categorical_encoding=(
            "index" if model_kind == "tabular-embed-mlp" else "onehot"
        ),
    )

    log_dir_p = Path(log_dir)
    log_dir_p.mkdir(parents=True, exist_ok=True)
    out_path_p = Path(out_path)

    test_probs_sum: np.ndarray | None = None
    test_logp_sum: np.ndarray | None = None
    fold_f1s: list[float] = []
    fold_bal_accs: list[float] = []
    fold_weighted_f1s: list[float] = []

    with TemporaryDirectory(prefix="zindi_cv_") as tmp:
        tmp_dir = Path(tmp)

        for fold_idx, valid_idx in enumerate(fold_valid_indices):
            valid_set = set(valid_idx.tolist())
            train_fold_rows = [r for i, r in enumerate(train_rows) if i not in valid_set]
            valid_fold_rows = [r for i, r in enumerate(train_rows) if i in valid_set]

            fold_train_csv = tmp_dir / f"fold_{fold_idx}_train.csv"
            fold_valid_csv = tmp_dir / f"fold_{fold_idx}_valid.csv"
            _write_csv_dicts(str(fold_train_csv), fieldnames, train_fold_rows)
            _write_csv_dicts(str(fold_valid_csv), fieldnames, valid_fold_rows)

            dataset = TabularCsvMulticlassClassificationDatasetProvider(
                train_csv_path=str(fold_train_csv),
                valid_csv_path=str(fold_valid_csv),
                test_csv_path=test_csv,
                valid_fraction=0.0,
                seed=int(seed) + int(fold_idx),
                config=tab_cfg,
                class_names=label_order,
            )

            # Fold-local metrics.
            fold_log_path = log_dir_p / f"fold_{fold_idx}.jsonl"
            stdout_metrics = StdoutMetricsSink()
            metrics: MetricsSinkPort = CompositeMetricsSink(
                stdout_metrics, JsonlFileMetricsSink(path=str(fold_log_path))
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
                seed=int(seed) + int(fold_idx),
                learning_rate=lr,
                weight_decay=weight_decay,
                adamw_b1=adamw_b1,
                adamw_b2=adamw_b2,
                adamw_eps=adamw_eps,
                adamw_eps_root=adamw_eps_root,
                adamw_nesterov=adamw_nesterov,
                hidden_sizes=tuple(hidden),
                log_every_steps=50,
                early_stopping_patience=int(early_stopping_patience),
                multiclass_early_stopping_metric=str(early_stopping_metric),
                loss_kind=str(loss_kind),
                ordinal_rank_lambda=float(ordinal_rank_lambda),
                ordinal_rank_margin=float(ordinal_rank_margin),
                ordinal_rank_pairs_per_batch=int(ordinal_rank_pairs_per_batch),
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
                    "event": "fold_start",
                    "command": "zindi-financial-health-cv",
                    "fold": int(fold_idx),
                    "n_folds": int(n_folds),
                    "dataset": dataset.describe(),
                    "model_kind": model_kind,
                    "remat": bool(remat),
                    "embed_dim": int(embed_dim),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "seed": int(seed) + int(fold_idx),
                    "loss_kind": str(loss_kind),
                    "early_stopping_patience": int(early_stopping_patience),
                },
            )

            result = use_case.run(cmd)

            ids_valid, x_valid, y_valid = dataset.get_validation()
            ids_test, x_test = dataset.get_inference()

            @jax.jit
            def predict_proba(params, x: jax.Array) -> jax.Array:
                logits = model.apply(params, x, is_training=False)
                return jax.nn.softmax(logits, axis=-1)

            p_valid = np.asarray(predict_proba(result.params, jnp.asarray(x_valid)))
            y_hat_valid = np.asarray(np.argmax(p_valid, axis=-1)).astype(np.int32)
            cm = _confusion_matrix_counts(
                y_true=np.asarray(y_valid, dtype=np.int32),
                y_pred=np.asarray(y_hat_valid, dtype=np.int32),
                n_classes=int(len(label_order)),
            )
            fold_macro_f1, fold_weighted_f1, fold_bal_acc = _f1s_and_balanced_accuracy(
                cm=cm
            )
            fold_f1s.append(float(fold_macro_f1))
            fold_bal_accs.append(float(fold_bal_acc))
            fold_weighted_f1s.append(float(fold_weighted_f1))

            global_step = (
                int(result.history[-1].get("global_step", 0)) if result.history else 0
            )
            metrics.log(
                step=global_step,
                metrics={
                    "event": "fold_valid_metrics",
                    "fold": int(fold_idx),
                    "valid/macro_f1": float(fold_macro_f1),
                    "valid/weighted_f1": float(fold_weighted_f1),
                    "valid/bal_acc": float(fold_bal_acc),
                    "valid/n": int(len(y_valid)),
                },
            )
            typer.echo(
                f"Fold {fold_idx}/{n_folds} macro-F1: {fold_macro_f1:.5f} | weighted-F1: {fold_weighted_f1:.5f} | bal-acc: {fold_bal_acc:.5f}"
            )

            p_test = np.asarray(predict_proba(result.params, jnp.asarray(x_test)))
            eps = 1e-9
            logp_test = np.log(p_test + eps)
            if test_probs_sum is None:
                test_probs_sum = p_test.astype(np.float64)
                test_logp_sum = logp_test.astype(np.float64)
                ids_test_ref = ids_test
            else:
                # Sanity check: same test IDs order.
                if ids_test.shape[0] != ids_test_ref.shape[0] or not np.all(
                    ids_test == ids_test_ref
                ):
                    raise RuntimeError(
                        "Test IDs/order changed between folds; cannot safely ensemble"
                    )
                test_probs_sum += p_test.astype(np.float64)
                assert test_logp_sum is not None
                test_logp_sum += logp_test.astype(np.float64)

        if test_probs_sum is None:
            raise RuntimeError("CV produced no folds")

        method = (ensemble_method or "mean_logp").strip().lower()
        if method not in {"mean_prob", "mean_logp"}:
            raise typer.BadParameter("ensemble_method must be one of: mean_prob, mean_logp")

        if method == "mean_prob":
            test_probs_mean = (test_probs_sum / float(n_folds)).astype(np.float32)
            y_hat_test = np.argmax(test_probs_mean, axis=-1).astype(np.int32)
        else:
            assert test_logp_sum is not None
            test_logp_mean = (test_logp_sum / float(n_folds)).astype(np.float32)
            y_hat_test = np.argmax(test_logp_mean, axis=-1).astype(np.int32)

    # Write final submission.
    labels = label_order
    pred_labels = [labels[int(i)] for i in y_hat_test.tolist()]
    with open(out_path_p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        for i, lab in zip(ids_test_ref.tolist(), pred_labels):
            w.writerow([str(i), str(lab)])

    typer.echo(f"Wrote ensemble submission to: {out_path_p}")
    typer.echo(
        f"CV macro-F1: mean={float(np.mean(fold_f1s)):.5f} std={float(np.std(fold_f1s)):.5f}"
    )
    typer.echo(
        f"CV weighted-F1: mean={float(np.mean(fold_weighted_f1s)):.5f} std={float(np.std(fold_weighted_f1s)):.5f}"
    )
    typer.echo(
        f"CV bal-acc:  mean={float(np.mean(fold_bal_accs)):.5f} std={float(np.std(fold_bal_accs)):.5f}"
    )

    if zip_output:
        zip_path = out_path_p.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path_p, arcname=out_path_p.name)
        typer.echo(f"Wrote zipped submission to: {zip_path}")
