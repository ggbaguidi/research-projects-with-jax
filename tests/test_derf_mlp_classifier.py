from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_deep_learning.core.domain.entities.model import DerfMlpClassifierFns


def test_derf_mlp_shapes_and_jit() -> None:
    model = DerfMlpClassifierFns(hidden_sizes=(16, 8))

    key = jax.random.PRNGKey(0)
    params = model.init(key=key, input_dim=10, num_classes=2)

    x = jax.random.normal(jax.random.PRNGKey(1), (4, 10))

    y = model.apply(params, x, is_training=True)
    assert y.shape == (4, 2)

    y2 = jax.jit(lambda p, xx: model.apply(p, xx, is_training=False))(params, x)
    assert y2.shape == (4, 2)

    # Ensure outputs are finite
    assert jnp.isfinite(y2).all()
