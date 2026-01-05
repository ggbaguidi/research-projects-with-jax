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
    # If >0, stop when the chosen validation metric hasn't improved for this
    # many epochs. Metric choice:
    # - binary classification: ROC AUC
    # - multiclass classification: macro-F1 (default) or weighted-F1
    # Metrics are computed on the provider's "test" split (which is used as
    # validation in some adapters).
    early_stopping_patience: int = 0

    # For multiclass classification only: which metric to use for early stopping.
    # Allowed values: "macro_f1", "weighted_f1".
    multiclass_early_stopping_metric: str = "macro_f1"

    # Loss configuration (optional)
    # - "softmax": classic multi-class softmax cross-entropy (default)
    # - "ordinal": ordinal decomposition loss derived from softmax probabilities
    # - "ordinal-rank": ordinal loss + pairwise ranking regularizer on expected class score
    loss_kind: str = "softmax"

    # Pairwise ranking regularizer (only used when loss_kind="ordinal-rank")
    # The score is s(x)=E[class|x] = sum_k k * softmax(logits)_k.
    ordinal_rank_lambda: float = 0.05
    ordinal_rank_margin: float = 0.25
    # How many ordered pairs to sample per batch (0 => use all ordered pairs).
    # Sampling avoids O(B^2) work for large batches.
    ordinal_rank_pairs_per_batch: int = 256
