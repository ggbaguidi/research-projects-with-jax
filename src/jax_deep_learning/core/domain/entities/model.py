from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp

Params = Any  # JAX pytree


class ClassifierFns(Protocol):
    """Pure model functions the core training loop can use.

    Implementations must be JAX-compatible (jit/vmap friendly).
    """

    def init(self, *, key: jax.Array, input_dim: int, num_classes: int) -> Params: ...

    def apply(self, params: Params, x: jax.Array, *, is_training: bool) -> jax.Array: ...


@dataclass(frozen=True)
class MlpClassifierFns:
    hidden_sizes: tuple[int, ...] = (512, 256)
    param_scale: float = 1e-2
    activation: Callable[[jax.Array], jax.Array] = jax.nn.swish

    def init(self, *, key: jax.Array, input_dim: int, num_classes: int) -> Params:
        sizes = (input_dim, *self.hidden_sizes, num_classes)

        def init_layer(m: int, n: int, k: jax.Array):
            w_key, b_key = jax.random.split(k)
            w = self.param_scale * jax.random.normal(w_key, (m, n))
            b = self.param_scale * jax.random.normal(b_key, (n,))
            return {"w": w, "b": b}

        keys = jax.random.split(key, len(sizes) - 1)
        return [init_layer(m, n, k) for (m, n), k in zip(zip(sizes[:-1], sizes[1:]), keys)]

    def apply(self, params: Params, x: jax.Array, *, is_training: bool) -> jax.Array:
        # x: (batch, input_dim)
        h = x
        for layer in params[:-1]:
            h = self.activation(jnp.dot(h, layer["w"]) + layer["b"])
        last = params[-1]
        return jnp.dot(h, last["w"]) + last["b"]
