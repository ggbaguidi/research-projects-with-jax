from __future__ import annotations

from typing import Any, Protocol


class CheckpointStorePort(Protocol):
    """Port for saving/loading model state.

    Keep I/O out of core; adapters implement this (filesystem, S3, etc.).
    """

    def save(self, *, step: int, state: Any, metadata: dict[str, Any] | None = None) -> None: ...

    def load_latest(self) -> tuple[int, Any] | None: ...
