from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainCommand:
    """Intent to train a classifier."""

    epochs: int = 10
    batch_size: int = 32
    seed: int = 0

    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # Default model: MLP over flattened inputs
    hidden_sizes: tuple[int, ...] = (512, 256)

    # Logging
    log_every_steps: int = 100
