from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from jax_deep_learning.core.ports.metrics_sink import MetricsSinkPort


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of metrics values to JSON-serializable types."""

    # Fast-path for common scalar types.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # numpy scalars
    if isinstance(value, np.generic):
        return value.item()

    # arrays (numpy / jax) -> try scalar or list
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            arr = np.asarray(value)
            if arr.shape == ():
                return arr.item()
            return arr.tolist()
        except Exception:  # pylint: disable=broad-exception-caught
            return str(value)

    # mappings
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    # sequences
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    return str(value)


class JsonlFileMetricsSink(MetricsSinkPort):
    """Append-only JSONL metrics sink.

    Each call writes one JSON object on a single line:
      {"ts": "...", "step": 123, "metrics": {...}}
    """

    def __init__(self, *, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, *, step: int, metrics: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "step": int(step),
            "metrics": _to_jsonable(metrics),
        }
        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


class CompositeMetricsSink(MetricsSinkPort):
    """Tee metrics to multiple sinks."""

    def __init__(self, *sinks: MetricsSinkPort) -> None:
        self._sinks = [s for s in sinks if s is not None]

    def log(self, *, step: int, metrics: dict[str, Any]) -> None:
        for s in self._sinks:
            s.log(step=step, metrics=metrics)
