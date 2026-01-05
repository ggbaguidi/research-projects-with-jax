from __future__ import annotations

from typing import Optional

import inject

from jax_deep_learning.core.domain.entities.model import ClassifierFns
from jax_deep_learning.core.ports.checkpoint_store import CheckpointStorePort
from jax_deep_learning.core.ports.dataset_provider import DatasetProviderPort
from jax_deep_learning.core.ports.metrics_sink import MetricsSinkPort
from jax_deep_learning.core.use_cases.train_classifier import \
    TrainClassifierUseCase


# pylint: disable=invalid-name
def get_dependencies_injection_config(
    *,
    dataset_provider: DatasetProviderPort,
    metrics_sink: MetricsSinkPort,
    checkpoint_store: Optional[CheckpointStorePort] = None,
    model_fns: Optional[ClassifierFns] = None,
):
    """Return an inject binder function (same pattern as your example).

    No imports occur inside the returned function.
    """

    def configure_dependencies_injection(binder: inject.Binder) -> None:
        binder.bind(DatasetProviderPort, dataset_provider)
        binder.bind(MetricsSinkPort, metrics_sink)
        if checkpoint_store is not None:
            binder.bind(CheckpointStorePort, checkpoint_store)

        # Bind the use case as a fully-wired object.
        binder.bind(
            TrainClassifierUseCase,
            TrainClassifierUseCase(
                dataset_provider=dataset_provider,
                checkpoint_store=checkpoint_store,
                metrics_sink=metrics_sink,
                model_fns=model_fns,
            ),
        )

    return configure_dependencies_injection


def configure_injections(
    *,
    dataset_provider: DatasetProviderPort,
    metrics_sink: MetricsSinkPort,
    checkpoint_store: Optional[CheckpointStorePort] = None,
    model_fns: Optional[ClassifierFns] = None,
) -> None:
    """Configure inject with this app's runtime bindings.

    Safe to call multiple times (clears previous bindings).
    """

    config = get_dependencies_injection_config(
        dataset_provider=dataset_provider,
        metrics_sink=metrics_sink,
        checkpoint_store=checkpoint_store,
        model_fns=model_fns,
    )

    if inject.is_configured():
        inject.clear_and_configure(config)
    else:
        inject.configure(config)
