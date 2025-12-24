from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from jax_deep_learning.core.domain.entities.model import TabularEmbedMlpClassifierFns


def test_tabular_embed_mlp_outputs_logits() -> None:
    model = TabularEmbedMlpClassifierFns(
        n_numeric=3,
        categorical_cardinalities=(4, 7),
        embed_dim=5,
        hidden_sizes=(16,),
    )

    # Layout: [3 numeric floats, 2 categorical indices]
    x = jnp.asarray(
        [
            [0.1, -0.2, 1.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 3.0, 2.0],
            [1.0, 2.0, 3.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    key = jax.random.PRNGKey(0)
    params = model.init(key=key, input_dim=x.shape[1], num_classes=2)
    logits = model.apply(params, x, is_training=True)

    assert logits.shape == (x.shape[0], 2)
    assert jnp.isfinite(logits).all()
