from __future__ import annotations

import jax


def fold_in_step(key: jax.Array, step: int) -> jax.Array:
    """Derive a deterministic per-step key."""

    return jax.random.fold_in(key, step)
