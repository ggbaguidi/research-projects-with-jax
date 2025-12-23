from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.base import Batch, StepMetrics
from jax_deep_learning.core.domain.entities.model import ClassifierFns, MlpClassifierFns, Params
from jax_deep_learning.core.domain.errors.training import TrainingError
from jax_deep_learning.core.ports.checkpoint_store import CheckpointStorePort
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.ports.metrics_sink import MetricsSinkPort


@dataclass(frozen=True)
class TrainResult:
    params: Params
    history: list[dict[str, Any]]


class TrainClassifierUseCase:
    def __init__(
        self,
        *,
        dataset_provider: DatasetProviderPort,
        checkpoint_store: CheckpointStorePort | None = None,
        metrics_sink: MetricsSinkPort | None = None,
        model_fns: ClassifierFns | None = None,
    ) -> None:
        self._dataset = dataset_provider
        self._ckpt = checkpoint_store
        self._metrics = metrics_sink
        self._model = model_fns or MlpClassifierFns()

    def run(self, command: TrainCommand) -> TrainResult:
        info = self._dataset.info
        if info.num_classes <= 1:
            raise TrainingError(f"num_classes must be >= 2, got {info.num_classes}")
        if len(info.input_shape) < 1:
            raise TrainingError(f"input_shape must be known, got {info.input_shape}")

        input_dim = 1
        for d in info.input_shape:
            input_dim *= d

        key = jax.random.PRNGKey(command.seed)
        params = self._model.init(key=key, input_dim=input_dim, num_classes=info.num_classes)

        optimizer = optax.adamw(learning_rate=command.learning_rate, weight_decay=command.weight_decay)
        opt_state = optimizer.init(params)

        def _prepare_x(x: jax.Array) -> jax.Array:
            # Flatten any (B, ...) into (B, D)
            return jnp.reshape(x, (x.shape[0], -1))

        def loss_and_metrics(p: Params, batch: Batch) -> StepMetrics:
            x = _prepare_x(jnp.asarray(batch.x))
            y = jnp.asarray(batch.y).astype(jnp.int32)
            logits = self._model.apply(p, x, is_training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return StepMetrics(loss=float(loss), accuracy=float(acc))

        @jax.jit
        def train_step(p: Params, s: optax.OptState, x: jax.Array, y: jax.Array):
            def _loss_fn(pp: Params):
                logits = self._model.apply(pp, x, is_training=True)
                return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

            loss, grads = jax.value_and_grad(_loss_fn)(p)
            updates, s2 = optimizer.update(grads, s, p)
            p2 = optax.apply_updates(p, updates)
            return p2, s2, loss

        @jax.jit
        def eval_step(p: Params, x: jax.Array, y: jax.Array):
            logits = self._model.apply(p, x, is_training=False)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, acc

        history: list[dict[str, Any]] = []
        global_step = 0

        for epoch in range(1, command.epochs + 1):
            # Train
            for batch in self._dataset.iter_batches(
                split="train",
                batch_size=command.batch_size,
                shuffle=True,
                seed=command.seed + epoch,
            ):
                x = _prepare_x(jnp.asarray(batch.x))
                y = jnp.asarray(batch.y).astype(jnp.int32)
                params, opt_state, loss = train_step(params, opt_state, x, y)
                global_step += 1

                if self._metrics and (global_step % command.log_every_steps == 0):
                    m = loss_and_metrics(params, batch)
                    self._metrics.log(step=global_step, metrics={"train/loss": m.loss, "train/acc": m.accuracy})

            # Eval (optional)
            eval_losses = []
            eval_accs = []
            for batch in self._dataset.iter_batches(
                split="test",
                batch_size=command.batch_size,
                shuffle=False,
                seed=command.seed,
            ):
                x = _prepare_x(jnp.asarray(batch.x))
                y = jnp.asarray(batch.y).astype(jnp.int32)
                l, a = eval_step(params, x, y)
                eval_losses.append(float(l))
                eval_accs.append(float(a))

            epoch_summary = {
                "epoch": epoch,
                "test/loss": float(sum(eval_losses) / max(1, len(eval_losses))) if eval_losses else None,
                "test/acc": float(sum(eval_accs) / max(1, len(eval_accs))) if eval_accs else None,
                "global_step": global_step,
            }
            history.append(epoch_summary)
            if self._metrics:
                self._metrics.log(step=global_step, metrics=epoch_summary)

            if self._ckpt:
                self._ckpt.save(step=global_step, state={"params": params, "opt_state": opt_state}, metadata=epoch_summary)

        return TrainResult(params=params, history=history)
