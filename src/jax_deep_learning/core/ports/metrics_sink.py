from __future__ import annotations

from typing import Any, Protocol


class MetricsSinkPort(Protocol):
    """Port for logging metrics (stdout, TensorBoard, W&B, etc.)."""

    def log(self, *, step: int, metrics: dict[str, Any]) -> None:
        ...
