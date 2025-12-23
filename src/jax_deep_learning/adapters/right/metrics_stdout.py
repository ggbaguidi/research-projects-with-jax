from __future__ import annotations

from typing import Any

from jax_deep_learning.core.ports.metrics_sink import MetricsSinkPort


class StdoutMetricsSink(MetricsSinkPort):
    def log(self, *, step: int, metrics: dict[str, Any]) -> None:
        items = ", ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"[step={step}] {items}")
