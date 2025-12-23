from __future__ import annotations

import json
import os
from typing import Any

from safetensors.numpy import save_file

from jax_deep_learning.core.ports.checkpoint_store import CheckpointStorePort


class FilesystemCheckpointStore(CheckpointStorePort):
    """Very small checkpoint adapter.

    Saves a state dict containing at least `params`.
    For now we only persist params (as safetensors) and minimal metadata.
    """

    def __init__(self, *, dir_path: str) -> None:
        self._dir = dir_path
        os.makedirs(self._dir, exist_ok=True)

    def save(self, *, step: int, state: Any, metadata: dict[str, Any] | None = None) -> None:
        params = state.get("params") if isinstance(state, dict) else None
        if params is None:
            raise ValueError("state must be a dict containing 'params'")

        # Flatten only top-level array leaves. For nested pytrees, a richer serializer is needed.
        # This is intentionally minimal and can be expanded later.
        flat = {}
        for i, layer in enumerate(params):
            flat[f"layer_{i}.w"] = layer["w"]
            flat[f"layer_{i}.b"] = layer["b"]

        ckpt_path = os.path.join(self._dir, f"params_step_{step}.safetensors")
        save_file(flat, ckpt_path)

        meta_path = os.path.join(self._dir, f"meta_step_{step}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata or {}, f)

    def load_latest(self) -> tuple[int, Any] | None:
        # Not implemented yet (requires reading safetensors back into the exact pytree structure).
        return None
