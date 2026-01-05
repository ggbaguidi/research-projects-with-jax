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

    def init(self, *, key: jax.Array, input_dim: int, num_classes: int) -> Params:
        ...

    def apply(self, params: Params, x: jax.Array, *, is_training: bool) -> jax.Array:
        ...


def _derf(
    x: jax.Array,
    *,
    alpha: jax.Array,
    s: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Dynamic erf (Derf) point-wise function.

    Based on arXiv:2512.10938v1, Eq. (10):
        Derf(x) = gamma * erf(alpha * x + s) + beta

    Paper (HTML): https://arxiv.org/html/2512.10938v1
    Paper (PDF):  https://arxiv.org/pdf/2512.10938v1

    This is statistics-free and can be used as a normalization replacement or a
    stable saturating activation.
    """

    return gamma * jax.lax.erf(alpha * x + s) + beta


@dataclass(frozen=True)
class MlpClassifierFns:
    hidden_sizes: tuple[int, ...] = (512, 256)
    param_scale: float = 1e-2
    activation: Callable[[jax.Array], jax.Array] = jax.nn.swish
    # If True, wraps each hidden layer with `jax.checkpoint` (a.k.a. remat).
    # This trades compute for lower activation memory, which can help for deeper MLPs.
    remat: bool = False

    def init(self, *, key: jax.Array, input_dim: int, num_classes: int) -> Params:
        sizes = (input_dim, *self.hidden_sizes, num_classes)

        def init_layer(m: int, n: int, k: jax.Array):
            w_key, b_key = jax.random.split(k)
            w = self.param_scale * jax.random.normal(w_key, (m, n))
            b = self.param_scale * jax.random.normal(b_key, (n,))
            return {"w": w, "b": b}

        keys = jax.random.split(key, len(sizes) - 1)
        return [
            init_layer(m, n, k) for (m, n), k in zip(zip(sizes[:-1], sizes[1:]), keys)
        ]

    def apply(self, params: Params, x: jax.Array, *, is_training: bool) -> jax.Array:
        # x: (batch, input_dim)
        h = x

        def _hidden_forward(hh: jax.Array, layer: dict[str, jax.Array]) -> jax.Array:
            return self.activation(jnp.dot(hh, layer["w"]) + layer["b"])

        hidden_forward = (
            jax.checkpoint(_hidden_forward) if self.remat else _hidden_forward
        )

        for layer in params[:-1]:
            h = hidden_forward(h, layer)
        last = params[-1]
        return jnp.dot(h, last["w"]) + last["b"]


@dataclass(frozen=True)
class DerfMlpClassifierFns:
    """MLP classifier using Derf as a normalization-free activation.

    Reference: "Stronger Normalization-Free Transformers" (Derf)
      - HTML: https://arxiv.org/html/2512.10938v1
      - PDF:  https://arxiv.org/pdf/2512.10938v1

    This keeps the existing 'pure functions' interface while incorporating the
    paper's learnable, point-wise Derf mapping. It is especially handy in tabular
    settings where LayerNorm is often omitted but training stability and
    generalization can still benefit from bounded, zero-centered transforms.
    """

    hidden_sizes: tuple[int, ...] = (256, 128)
    param_scale: float = 1e-2
    derf_alpha_init: float = 0.5
    derf_s_init: float = 0.0
    # If True, wraps each hidden layer with `jax.checkpoint` (a.k.a. remat).
    # This trades compute for lower activation memory, which can help for deeper MLPs.
    remat: bool = False

    def init(self, *, key: jax.Array, input_dim: int, num_classes: int) -> Params:
        sizes = (input_dim, *self.hidden_sizes, num_classes)

        def init_linear(m: int, n: int, k: jax.Array):
            w_key, b_key = jax.random.split(k)
            w = self.param_scale * jax.random.normal(w_key, (m, n))
            b = self.param_scale * jax.random.normal(b_key, (n,))
            return {"w": w, "b": b}

        def init_derf(n: int):
            # alpha and s are scalars in the paper; gamma/beta are per-channel.
            alpha = jnp.asarray(self.derf_alpha_init, dtype=jnp.float32)
            s = jnp.asarray(self.derf_s_init, dtype=jnp.float32)
            gamma = jnp.ones((n,), dtype=jnp.float32)
            beta = jnp.zeros((n,), dtype=jnp.float32)
            return {"alpha": alpha, "s": s, "gamma": gamma, "beta": beta}

        # One dict per layer. Hidden layers include both linear params and Derf params.
        keys = jax.random.split(key, len(sizes) - 1)
        layers: list[dict[str, Any]] = []
        for (m, n), k in zip(zip(sizes[:-1], sizes[1:]), keys):
            layers.append({"linear": init_linear(m, n, k)})

        for i, n in enumerate(sizes[1:-1]):
            layers[i]["derf"] = init_derf(n)

        return layers

    def apply(self, params: Params, x: jax.Array, *, is_training: bool) -> jax.Array:
        h = x

        def _hidden_forward(hh: jax.Array, layer: dict[str, Any]) -> jax.Array:
            lin = layer["linear"]
            z = jnp.dot(hh, lin["w"]) + lin["b"]
            d = layer["derf"]
            return _derf(
                z, alpha=d["alpha"], s=d["s"], gamma=d["gamma"], beta=d["beta"]
            )

        hidden_forward = (
            jax.checkpoint(_hidden_forward) if self.remat else _hidden_forward
        )

        # All but last are hidden layers
        for layer in params[:-1]:
            h = hidden_forward(h, layer)

        last = (
            params[-1]["linear"]
            if isinstance(params[-1], dict) and "linear" in params[-1]
            else params[-1]
        )
        return jnp.dot(h, last["w"]) + last["b"]


@dataclass(frozen=True)
class TabularEmbedMlpClassifierFns:
    """Tabular classifier: numeric features + categorical embeddings + MLP.

    Expects input layout:
      x[:, :n_numeric]               -> standardized numeric features (float)
      x[:, n_numeric:n_numeric+n_cat] -> categorical indices (stored as float but cast to int)

    This keeps the existing `ClassifierFns` interface so it can be used by the
    core training loop unchanged.
    """

    n_numeric: int
    categorical_cardinalities: tuple[int, ...]
    embed_dim: int = 8
    hidden_sizes: tuple[int, ...] = (512, 256)
    param_scale: float = 1e-2
    activation: Callable[[jax.Array], jax.Array] = jax.nn.swish

    def init(self, *, key: jax.Array, input_dim: int, num_classes: int) -> Params:
        n_cat = len(self.categorical_cardinalities)
        # Expected packed input_dim = n_numeric + n_cat
        # We don't hard-fail to keep the interface flexible, but the model
        # will behave incorrectly if the layout doesn't match.

        def init_embed(card: int, k: jax.Array) -> jax.Array:
            # Small random embedding table.
            return self.param_scale * jax.random.normal(
                k, (int(card), int(self.embed_dim))
            )

        def init_layer(m: int, n: int, k: jax.Array):
            w_key, b_key = jax.random.split(k)
            w = self.param_scale * jax.random.normal(w_key, (m, n))
            b = self.param_scale * jax.random.normal(b_key, (n,))
            return {"w": w, "b": b}

        # Embeddings
        k_emb, k_mlp = jax.random.split(key)
        if n_cat > 0:
            emb_keys = jax.random.split(k_emb, n_cat)
            embeddings = [
                init_embed(card, kk)
                for card, kk in zip(self.categorical_cardinalities, emb_keys)
            ]
        else:
            embeddings = []

        mlp_in = int(self.n_numeric + n_cat * self.embed_dim)
        sizes = (mlp_in, *self.hidden_sizes, int(num_classes))
        layer_keys = jax.random.split(k_mlp, len(sizes) - 1)
        mlp_layers = [
            init_layer(m, n, kk)
            for (m, n), kk in zip(zip(sizes[:-1], sizes[1:]), layer_keys)
        ]

        # NOTE: Keep params as real-valued JAX arrays only.
        # Putting Python/NumPy ints into the params pytree can make `jax.grad` fail
        # (it tries to take gradients w.r.t. integer leaves).
        return {"embeddings": embeddings, "mlp": mlp_layers}

    def apply(self, params: Params, x: jax.Array, *, is_training: bool) -> jax.Array:
        n_numeric = int(self.n_numeric)
        n_cat = len(params.get("embeddings", []))

        x_num = x[:, :n_numeric]
        x_cat = x[:, n_numeric : n_numeric + n_cat]
        x_cat = x_cat.astype(jnp.int32)

        embs = []
        for i, table in enumerate(params.get("embeddings", [])):
            idx = x_cat[:, i]
            # Clip for safety in case of corrupted inputs.
            idx = jnp.clip(idx, 0, table.shape[0] - 1)
            embs.append(table[idx])

        if embs:
            h = jnp.concatenate([x_num, *embs], axis=-1)
        else:
            h = x_num

        mlp_layers = params["mlp"]
        for layer in mlp_layers[:-1]:
            h = self.activation(jnp.dot(h, layer["w"]) + layer["b"])
        last = mlp_layers[-1]
        return jnp.dot(h, last["w"]) + last["b"]
