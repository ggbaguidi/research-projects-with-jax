from __future__ import annotations

import json
from pathlib import Path

from jax_deep_learning.adapters.right.metrics_plotting import plot_metrics_from_logs


def test_plot_metrics_writes_png(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    out_path = tmp_path / "metrics.png"

    # Minimal JSONL log in the project format.
    records = [
        {
            "ts": "2026-01-01T00:00:00+00:00",
            "step": 0,
            "metrics": {"event": "run_start", "command": "train"},
        },
        {
            "ts": "2026-01-01T00:00:01+00:00",
            "step": 1,
            "metrics": {"train/loss": 1.0, "train/acc": 0.4},
        },
        {
            "ts": "2026-01-01T00:00:02+00:00",
            "step": 2,
            "metrics": {"train/loss": 0.8, "train/acc": 0.5},
        },
        {
            "ts": "2026-01-01T00:00:03+00:00",
            "step": 3,
            "metrics": {
                "epoch": 1,
                "global_step": 3,
                "test/loss": 0.9,
                "test/acc": 0.45,
            },
        },
        {
            "ts": "2026-01-01T00:00:04+00:00",
            "step": 4,
            "metrics": {
                "epoch": 2,
                "global_step": 4,
                "test/loss": 0.7,
                "test/acc": 0.55,
            },
        },
    ]

    with log_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")

    saved = plot_metrics_from_logs(
        log_paths=[log_path],
        out_path=out_path,
        show=False,
        x_axis="step",
        metrics=None,
        group_by="suffix",
        title="unit test",
    )

    assert saved is not None
    assert saved.exists()
    assert saved.stat().st_size > 0
