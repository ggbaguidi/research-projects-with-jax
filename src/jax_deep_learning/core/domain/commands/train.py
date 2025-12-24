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

    # Optimizer (Optax AdamW)
    # Matches optax.adamw signature defaults so existing behavior is unchanged.
    adamw_b1: float = 0.9
    adamw_b2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_eps_root: float = 0.0
    adamw_nesterov: bool = False

    # Default model: MLP over flattened inputs
    hidden_sizes: tuple[int, ...] = (512, 256)

    # Logging
    log_every_steps: int = 1000

    # Early stopping (optional)
    # If >0 and the task is binary classification, stop when validation AUC
    # hasn't improved for this many epochs. AUC is computed on the provider's
    # "test" split (which is used as validation in some adapters).
    early_stopping_patience: int = 0
