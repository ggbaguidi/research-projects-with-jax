from __future__ import annotations

import json

from jax_deep_learning.adapters.right.metrics_jsonl import JsonlFileMetricsSink


def test_jsonl_metrics_sink_writes_valid_lines(tmp_path) -> None:
    p = tmp_path / "metrics.jsonl"
    sink = JsonlFileMetricsSink(path=p)

    sink.log(step=0, metrics={"event": "start", "lr": 1e-3})
    sink.log(step=10, metrics={"train/loss": 0.5, "train/acc": 0.9})

    text = p.read_text(encoding="utf-8").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2

    rec0 = json.loads(lines[0])
    assert rec0["step"] == 0
    assert rec0["metrics"]["event"] == "start"

    rec1 = json.loads(lines[1])
    assert rec1["step"] == 10
    assert "train/loss" in rec1["metrics"]
