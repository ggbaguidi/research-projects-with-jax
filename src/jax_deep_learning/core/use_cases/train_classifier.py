from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_deep_learning.core.domain.commands.train import TrainCommand
from jax_deep_learning.core.domain.entities.base import Batch, StepMetrics
from jax_deep_learning.core.domain.entities.model import (ClassifierFns,
                                                          MlpClassifierFns,
                                                          Params)
from jax_deep_learning.core.domain.errors.training import TrainingError
from jax_deep_learning.core.domain.utils.metrics import roc_auc_score_binary
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
        params = self._model.init(
            key=key, input_dim=input_dim, num_classes=info.num_classes
        )

        optimizer = optax.adamw(
            learning_rate=command.learning_rate,
            b1=command.adamw_b1,
            b2=command.adamw_b2,
            eps=command.adamw_eps,
            eps_root=command.adamw_eps_root,
            weight_decay=command.weight_decay,
            nesterov=command.adamw_nesterov,
        )
        opt_state = optimizer.init(params)

        loss_kind = (command.loss_kind or "softmax").lower().strip()

        def _prepare_x(x: jax.Array) -> jax.Array:
            # Flatten any (B, ...) into (B, D)
            return jnp.reshape(x, (x.shape[0], -1))

        def _loss_softmax(*, logits: jax.Array, y: jax.Array) -> jax.Array:
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        def _loss_ordinal_from_softmax(*, logits: jax.Array, y: jax.Array) -> jax.Array:
            """Ordinal loss without class weights.

            For K classes, define binary events E_k := [y > k] for k=0..K-2.
            With p = softmax(logits), model P(E_k=1|x) = sum_{j>k} p_j.
            Loss is sum of binary log-losses over k.

            This leverages the label ordering (Low < Medium < High) and tends to
            reduce egregious errors like High->Low compared to plain softmax CE.
            """

            probs = jax.nn.softmax(logits, axis=-1)
            num_classes = int(logits.shape[-1])
            # cumulative probability of being strictly greater than k
            # p_gt[k] = sum_{j=k+1..K-1} probs[..., j]
            p_gt = 1.0 - jnp.cumsum(probs, axis=-1)[..., :-1]
            y = y.astype(jnp.int32)
            thresholds = jnp.arange(num_classes - 1, dtype=jnp.int32)
            # targets shape (B, K-1)
            t = (y[..., None] > thresholds[None, :]).astype(jnp.float32)
            eps = 1e-7
            p_gt = jnp.clip(p_gt, eps, 1.0 - eps)
            bce = -(t * jnp.log(p_gt) + (1.0 - t) * jnp.log(1.0 - p_gt))
            return jnp.mean(jnp.sum(bce, axis=-1))

        def _ranking_loss(
            *,
            logits: jax.Array,
            y: jax.Array,
            key: jax.Array,
            margin: float,
            pairs_per_batch: int,
        ) -> jax.Array:
            """Pairwise ranking loss on expected class score.

            Uses s(x)=E[class|x] = sum_k k * p_k with p=softmax(logits).
            For pairs (i,j) where y_i > y_j, penalize if s_i < s_j + margin.

            If pairs_per_batch>0, samples that many ordered pairs to avoid O(B^2).
            If pairs_per_batch==0, uses all ordered pairs (O(B^2)).
            """

            probs = jax.nn.softmax(logits, axis=-1)
            k_idx = jnp.arange(logits.shape[-1], dtype=jnp.float32)
            s = jnp.sum(probs * k_idx[None, :], axis=-1)  # (B,)
            y = y.astype(jnp.int32)
            b = int(s.shape[0])
            if b <= 1:
                return jnp.asarray(0.0, dtype=jnp.float32)

            if pairs_per_batch and pairs_per_batch > 0:
                # Sample i,j uniformly and keep only those with y_i > y_j.
                # This amplifies rare-class signal via combinatorics without weights.
                key_i, key_j = jax.random.split(key)
                i = jax.random.randint(key_i, shape=(pairs_per_batch,), minval=0, maxval=b)
                j = jax.random.randint(key_j, shape=(pairs_per_batch,), minval=0, maxval=b)
                yi = y[i]
                yj = y[j]
                si = s[i]
                sj = s[j]
                mask = (yi > yj) & (i != j)
                # If no valid pairs, return 0.
                denom = jnp.maximum(1.0, jnp.sum(mask.astype(jnp.float32)))
                losses = jnp.maximum(0.0, margin - (si - sj)) * mask.astype(jnp.float32)
                return jnp.sum(losses) / denom

            # All ordered pairs (i,j)
            si = s[:, None]
            sj = s[None, :]
            yi = y[:, None]
            yj = y[None, :]
            mask = (yi > yj).astype(jnp.float32)
            losses = jnp.maximum(0.0, margin - (si - sj)) * mask
            denom = jnp.maximum(1.0, jnp.sum(mask))
            return jnp.sum(losses) / denom

        def _loss_main(*, logits: jax.Array, y: jax.Array) -> jax.Array:
            if loss_kind == "softmax":
                return _loss_softmax(logits=logits, y=y)
            if loss_kind in {"ordinal", "ordinal-rank", "ordinal_rank"}:
                return _loss_ordinal_from_softmax(logits=logits, y=y)
            raise TrainingError(
                "loss_kind must be one of: softmax, ordinal, ordinal-rank"
            )

        def loss_and_metrics(p: Params, batch: Batch) -> StepMetrics:
            x = _prepare_x(jnp.asarray(batch.x))
            y = jnp.asarray(batch.y).astype(jnp.int32)
            logits = self._model.apply(p, x, is_training=True)
            loss = _loss_main(logits=logits, y=y)
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return StepMetrics(loss=float(loss), accuracy=float(acc))

        @jax.jit
        def train_step(
            p: Params,
            s: optax.OptState,
            key: jax.Array,
            x: jax.Array,
            y: jax.Array,
        ):
            key_next, key_rank = jax.random.split(key)

            def _loss_fn(pp: Params):
                logits = self._model.apply(pp, x, is_training=True)
                main = _loss_main(logits=logits, y=y)
                rank = jnp.asarray(0.0, dtype=jnp.float32)
                if loss_kind in {"ordinal-rank", "ordinal_rank"}:
                    rank = _ranking_loss(
                        logits=logits,
                        y=y,
                        key=key_rank,
                        margin=float(command.ordinal_rank_margin),
                        pairs_per_batch=int(command.ordinal_rank_pairs_per_batch),
                    )
                total = main + (float(command.ordinal_rank_lambda) * rank)
                return total, (main, rank)

            (loss, (loss_main, loss_rank)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(p)
            updates, s2 = optimizer.update(grads, s, p)
            p2 = optax.apply_updates(p, updates)

            # Stability diagnostics (cheap, useful for deeper networks):
            # Inspired by the stability analysis mindset in mHC:
            #   "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
            #   HTML: https://arxiv.org/html/2512.24880v1
            grad_norm = optax.global_norm(grads)
            param_norm = optax.global_norm(p)
            update_norm = optax.global_norm(updates)
            update_ratio = update_norm / (param_norm + 1e-12)
            return (
                p2,
                s2,
                key_next,
                loss,
                loss_main,
                loss_rank,
                grad_norm,
                param_norm,
                update_norm,
                update_ratio,
            )

        is_binary = info.num_classes == 2

        def _macro_f1_from_confusion(cm: np.ndarray) -> float:
            """Compute macro-F1 from a (C,C) confusion matrix."""

            cm = np.asarray(cm, dtype=np.float64)
            tp = np.diag(cm)
            fp = np.sum(cm, axis=0) - tp
            fn = np.sum(cm, axis=1) - tp
            precision = np.divide(
                tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
            )
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(tp),
                where=(precision + recall) > 0,
            )
            return float(np.mean(f1))

        def _weighted_f1_from_confusion(cm: np.ndarray) -> float:
            """Compute weighted-F1 from a (C,C) confusion matrix.

            Weighted-F1 averages per-class F1 weighted by class support.
            This often matches "F1 score" used in imbalanced multiclass leaderboards.
            """

            cm = np.asarray(cm, dtype=np.float64)
            tp = np.diag(cm)
            fp = np.sum(cm, axis=0) - tp
            fn = np.sum(cm, axis=1) - tp
            support = np.sum(cm, axis=1)
            precision = np.divide(
                tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
            )
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(tp),
                where=(precision + recall) > 0,
            )
            denom = float(np.sum(support))
            if denom <= 0:
                return 0.0
            return float(np.sum(f1 * support) / denom)

        @jax.jit
        def eval_step(p: Params, x: jax.Array, y: jax.Array):
            logits = self._model.apply(p, x, is_training=False)
            loss = _loss_main(logits=logits, y=y)
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return logits, loss, acc

        history: list[dict[str, Any]] = []
        global_step = 0

        step_key = jax.random.PRNGKey(command.seed + 12345)

        best_auc: float | None = None
        best_f1: float | None = None
        best_weighted_f1: float | None = None
        best_epoch: int | None = None
        best_params: Params | None = None
        epochs_since_improvement = 0
        auc_improvement_epsilon = 1e-6
        f1_improvement_epsilon = 1e-6

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
                (
                    params,
                    opt_state,
                    step_key,
                    loss,
                    loss_main,
                    loss_rank,
                    grad_norm,
                    param_norm,
                    update_norm,
                    update_ratio,
                ) = train_step(params, opt_state, step_key, x, y)
                global_step += 1

                if self._metrics and (global_step % command.log_every_steps == 0):
                    m = loss_and_metrics(params, batch)
                    self._metrics.log(
                        step=global_step,
                        metrics={
                            "train/loss": m.loss,
                            "train/loss_main": float(loss_main),
                            "train/loss_rank": float(loss_rank),
                            "train/acc": m.accuracy,
                            "train/grad_norm": float(grad_norm),
                            "train/param_norm": float(param_norm),
                            "train/update_norm": float(update_norm),
                            "train/update_ratio": float(update_ratio),
                            "train/loss_kind": loss_kind,
                        },
                    )

            # Eval (optional)
            eval_losses = []
            eval_accs = []
            y_true_pred_all: list[Any] = []
            y_true_all: list[Any] = []
            y_score_all: list[Any] = []
            for batch in self._dataset.iter_batches(
                split="test",
                batch_size=command.batch_size,
                shuffle=False,
                seed=command.seed,
            ):
                x = _prepare_x(jnp.asarray(batch.x))
                y = jnp.asarray(batch.y).astype(jnp.int32)
                logits, l, a = eval_step(params, x, y)
                eval_losses.append(float(l))
                eval_accs.append(float(a))

                # For F1 on multiclass (and usable for binary too): collect predicted labels.
                y_hat = jnp.argmax(logits, axis=-1)
                y_true_pred_all.append((jnp.asarray(y), jnp.asarray(y_hat)))

                if is_binary:
                    probs = jax.nn.softmax(logits, axis=-1)[:, 1]
                    y_true_all.append(jnp.asarray(y))
                    y_score_all.append(jnp.asarray(probs))

            eval_auc: float | None = None
            if is_binary and y_true_all:
                y_true_np = jnp.concatenate(y_true_all, axis=0)
                y_score_np = jnp.concatenate(y_score_all, axis=0)
                eval_auc = roc_auc_score_binary(
                    y_true=np.asarray(y_true_np, dtype=np.int32),
                    y_score=np.asarray(y_score_np, dtype=np.float64),
                )

                if not (eval_auc != eval_auc):  # not NaN
                    if best_auc is None or eval_auc > (
                        best_auc + auc_improvement_epsilon
                    ):
                        best_auc = float(eval_auc)
                        best_epoch = int(epoch)
                        best_params = params
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

            eval_macro_f1: float | None = None
            eval_weighted_f1: float | None = None
            if y_true_pred_all:
                yt = np.asarray(
                    jnp.concatenate([p[0] for p in y_true_pred_all], axis=0),
                    dtype=np.int32,
                )
                yp = np.asarray(
                    jnp.concatenate([p[1] for p in y_true_pred_all], axis=0),
                    dtype=np.int32,
                )
                c = int(info.num_classes)
                cm = np.zeros((c, c), dtype=np.int64)
                mask = (yt >= 0) & (yt < c) & (yp >= 0) & (yp < c)
                if np.any(mask):
                    np.add.at(cm, (yt[mask], yp[mask]), 1)
                eval_macro_f1 = _macro_f1_from_confusion(cm)
                eval_weighted_f1 = _weighted_f1_from_confusion(cm)

                chosen_metric = (command.multiclass_early_stopping_metric or "macro_f1").strip().lower()
                if chosen_metric not in {"macro_f1", "weighted_f1"}:
                    raise TrainingError(
                        "multiclass_early_stopping_metric must be one of: macro_f1, weighted_f1"
                    )
                metric_value = (
                    float(eval_weighted_f1)
                    if chosen_metric == "weighted_f1"
                    else float(eval_macro_f1)
                )

                # For multiclass tasks, early stopping uses macro-F1 (matches many leaderboards).
                if (not is_binary) and command.early_stopping_patience:
                    if chosen_metric == "weighted_f1":
                        if best_weighted_f1 is None or metric_value > (
                            best_weighted_f1 + f1_improvement_epsilon
                        ):
                            best_weighted_f1 = float(metric_value)
                            # Keep macro as well for logging/visibility.
                            best_f1 = float(eval_macro_f1)
                            best_epoch = int(epoch)
                            best_params = params
                            epochs_since_improvement = 0
                        else:
                            epochs_since_improvement += 1
                    else:
                        if best_f1 is None or metric_value > (best_f1 + f1_improvement_epsilon):
                            best_f1 = float(metric_value)
                            best_epoch = int(epoch)
                            best_params = params
                            epochs_since_improvement = 0
                        else:
                            epochs_since_improvement += 1

            epoch_summary = {
                "epoch": epoch,
                "test/loss": (
                    float(sum(eval_losses) / max(1, len(eval_losses)))
                    if eval_losses
                    else None
                ),
                "test/acc": (
                    float(sum(eval_accs) / max(1, len(eval_accs)))
                    if eval_accs
                    else None
                ),
                "test/auc": float(eval_auc) if eval_auc is not None else None,
                "test/macro_f1": float(eval_macro_f1) if eval_macro_f1 is not None else None,
                "test/weighted_f1": float(eval_weighted_f1)
                if eval_weighted_f1 is not None
                else None,
                "best/auc": best_auc,
                "best/f1": best_f1,
                "best/weighted_f1": best_weighted_f1,
                "best/epoch": best_epoch,
                "global_step": global_step,
            }
            history.append(epoch_summary)
            if self._metrics:
                self._metrics.log(step=global_step, metrics=epoch_summary)

            if self._ckpt:
                self._ckpt.save(
                    step=global_step,
                    state={"params": params, "opt_state": opt_state},
                    metadata=epoch_summary,
                )

            # Early stopping: only meaningful for binary tasks where AUC is computed.
            if is_binary and command.early_stopping_patience and best_auc is not None:
                if epochs_since_improvement >= command.early_stopping_patience:
                    if self._metrics:
                        self._metrics.log(
                            step=global_step,
                            metrics={
                                "event": "early_stop",
                                "epoch": epoch,
                                "best/auc": best_auc,
                                "best/epoch": best_epoch,
                                "patience": int(command.early_stopping_patience),
                            },
                        )
                    break

            # Early stopping for multiclass tasks using macro-F1.
            if (not is_binary) and command.early_stopping_patience and best_f1 is not None:
                if epochs_since_improvement >= command.early_stopping_patience:
                    if self._metrics:
                        self._metrics.log(
                            step=global_step,
                            metrics={
                                "event": "early_stop",
                                "epoch": epoch,
                                "best/f1": best_f1,
                                "best/weighted_f1": best_weighted_f1,
                                "best/epoch": best_epoch,
                                "patience": int(command.early_stopping_patience),
                                "metric": (command.multiclass_early_stopping_metric or "macro_f1").strip().lower(),
                            },
                        )
                    break

        # Prefer the best params by validation AUC if available.
        if best_params is not None:
            final_params = best_params
        else:
            final_params = params
        return TrainResult(params=final_params, history=history)
