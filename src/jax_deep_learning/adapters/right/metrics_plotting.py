from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricSeries:
    name: str
    xs: list[float]
    ys: list[float]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def read_jsonl_metrics_records(path: str | Path) -> list[dict[str, Any]]:
    """Read our JSONL metrics sink format.

    Expected one JSON object per line:
      {"ts": "...", "step": 123, "metrics": {...}}

    Returns a list of dict records. Invalid JSON lines are skipped.
    """

    p = Path(path)
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if "metrics" not in rec or "step" not in rec:
                continue
            records.append(rec)
    return records


def infer_run_name(records: list[dict[str, Any]], *, fallback: str) -> str:
    """Best-effort label for the run."""

    for rec in records:
        metrics = rec.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metrics.get("event") == "run_start":
            cmd = metrics.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()
    return fallback


def extract_metric_series(
    records: list[dict[str, Any]],
    *,
    x_axis: str = "step",
    include_metrics: Iterable[str] | None = None,
) -> dict[str, MetricSeries]:
    """Extract numeric time series from records.

    x_axis:
      - "step": record["step"]
      - "global_step": metrics.get("global_step", record["step"])
      - "epoch": metrics.get("epoch") (records without epoch are skipped)

    include_metrics: if provided, only these metric keys are collected.
    """

    x_axis = x_axis.strip().lower()
    if x_axis not in {"step", "global_step", "epoch"}:
        raise ValueError(
            f"x_axis must be one of: step, global_step, epoch (got {x_axis!r})"
        )

    include: set[str] | None = None
    if include_metrics is not None:
        include = {m.strip() for m in include_metrics if m and m.strip()}

    buckets: dict[str, tuple[list[float], list[float]]] = {}

    for rec in records:
        step = rec.get("step")
        if not _is_number(step):
            continue

        metrics = rec.get("metrics")
        if not isinstance(metrics, dict):
            continue

        if x_axis == "step":
            x = float(step)
        elif x_axis == "global_step":
            gs = metrics.get("global_step")
            x = float(gs) if _is_number(gs) else float(step)
        else:  # epoch
            ep = metrics.get("epoch")
            if not _is_number(ep):
                continue
            x = float(ep)

        for k, v in metrics.items():
            if k in {"event", "command", "dataset", "epoch", "global_step"}:
                continue
            if include is not None and k not in include:
                continue
            if v is None or not _is_number(v):
                continue

            xs, ys = buckets.setdefault(k, ([], []))
            xs.append(x)
            ys.append(float(v))

    out: dict[str, MetricSeries] = {}
    for name, (xs, ys) in buckets.items():
        if len(xs) < 2:
            continue
        # Sort by x for nicer plots.
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out[name] = MetricSeries(
            name=name,
            xs=[xs[i] for i in order],
            ys=[ys[i] for i in order],
        )
    return out


def default_metric_selection(series_by_name: dict[str, MetricSeries]) -> list[str]:
    """Pick a reasonable default set of metrics for plotting."""

    preferred = [
        "train/loss",
        "test/loss",
        "train/acc",
        "test/acc",
        "valid/acc",
        "train/auc",
        "test/auc",
        "valid/auc",
    ]

    selected = [m for m in preferred if m in series_by_name]

    if selected:
        return selected

    # Fallback: plot all slash-names (e.g. train/loss) first, then others.
    slash = sorted([k for k in series_by_name.keys() if "/" in k])
    other = sorted([k for k in series_by_name.keys() if "/" not in k])
    return slash + other


def group_metrics(
    metric_names: list[str], *, group_by: str = "suffix"
) -> dict[str, list[str]]:
    group_by = group_by.strip().lower()
    if group_by not in {"suffix", "none"}:
        raise ValueError(f"group_by must be one of: suffix, none (got {group_by!r})")

    groups: dict[str, list[str]] = {}
    for name in metric_names:
        if group_by == "none":
            g = name
        else:
            g = name.split("/")[-1]
        groups.setdefault(g, []).append(name)
    return groups


def plot_metrics_from_logs(
    *,
    log_paths: list[str | Path],
    out_path: str | Path | None,
    show: bool,
    x_axis: str = "step",
    metrics: list[str] | None = None,
    group_by: str = "suffix",
    title: str | None = None,
) -> Path | None:
    """Plot metrics from one or more JSONL log files.

    If show is False, out_path must be provided and the figure will be saved.
    Returns the saved path (or None if nothing saved).
    """

    if not show and not out_path:
        raise ValueError("out_path is required when show=False")

    # Choose backend before importing pyplot.
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt  # pylint: disable=import-error

    runs: list[tuple[str, dict[str, MetricSeries]]] = []
    for lp in log_paths:
        p = Path(lp)
        recs = read_jsonl_metrics_records(p)
        run_name = infer_run_name(recs, fallback=p.stem)
        series = extract_metric_series(recs, x_axis=x_axis, include_metrics=metrics)
        runs.append((run_name, series))

    if not runs:
        raise ValueError("No log files provided")

    # Decide which metric names to plot.
    if metrics and len(metrics) > 0:
        metric_names = [m for m in metrics if m]
    else:
        # Union defaults across runs.
        union_series: dict[str, MetricSeries] = {}
        for _, s in runs:
            union_series.update(s)
        metric_names = default_metric_selection(union_series)

    # Filter to metrics that exist in at least one run.
    existing = [
        m
        for m in metric_names
        if any(m in series_by_name for _, series_by_name in runs)
    ]
    if not existing:
        raise ValueError("No matching numeric metrics found to plot")

    groups = group_metrics(existing, group_by=group_by)
    group_names = list(groups.keys())

    fig_h = max(3.0, 3.0 * len(group_names))
    fig, axes = plt.subplots(
        nrows=len(group_names),
        ncols=1,
        figsize=(11.0, fig_h),
        sharex=True,
        constrained_layout=True,
    )

    if len(group_names) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, group_names, strict=True):
        for run_name, series_by_name in runs:
            for metric_name in groups[group_name]:
                s = series_by_name.get(metric_name)
                if s is None:
                    continue
                label = f"{run_name}: {metric_name}" if len(runs) > 1 else metric_name
                ax.plot(
                    s.xs, s.ys, marker="o", markersize=2.5, linewidth=1.5, label=label
                )

        ax.set_title(group_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel(x_axis)

    if title:
        fig.suptitle(title)

    saved: Path | None = None
    if out_path:
        saved = Path(out_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=140)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved
