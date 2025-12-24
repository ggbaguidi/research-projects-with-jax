
# JAX Deep Learning (Hexagonal Architecture)

A small-but-real JAX deep learning codebase structured using **hexagonal (ports & adapters)** architecture.

This repo’s goal is to keep the **core training logic** independent from I/O concerns (datasets, metrics logging, checkpoints), while still being easy to run from a CLI for experiments and Kaggle-style workflows.

## Features

- **Core-first design**: domain + ports + use-cases live in `src/jax_deep_learning/core/`.
- **Dataset-agnostic training** via `DatasetProviderPort`.
- **Typer CLI** entrypoint: `jaxdl`.
- **Dependency Injection** using `inject` (bindings configured once at CLI startup).
- **CPU-first JAX setup** to avoid CUDA plugin noise on non-CUDA machines.
- **Kaggle diabetes pipeline**: train + validate AUC + generate `submission.csv`.
- **Derf-based classifier** (Dynamic erf) inspired by the paper:
	- *Stronger Normalization-Free Transformers* (arXiv:2512.10938v1)

## Project layout

```
src/jax_deep_learning/
	core/
		domain/        # entities, commands, errors, pure utilities
		ports/         # Protocols: dataset provider, metrics sink, checkpoint store
		use_cases/     # orchestration: TrainClassifierUseCase
	adapters/
		left/          # inbound drivers (CLI)
		right/         # outbound adapters (datasets, logging, checkpoints)
```

### Key modules

- Training use-case: `src/jax_deep_learning/core/use_cases/train_classifier.py`
- Model functions: `src/jax_deep_learning/core/domain/entities/model.py`
	- `MlpClassifierFns` (baseline)
	- `DerfMlpClassifierFns` (Derf activation)
- Kaggle tabular dataset adapter: `src/jax_deep_learning/adapters/right/data_loaders/tabular_csv_kaggle.py`
- Metrics sinks:
	- `StdoutMetricsSink`: `src/jax_deep_learning/adapters/right/metrics_stdout.py`
	- `JsonlFileMetricsSink`: `src/jax_deep_learning/adapters/right/metrics_jsonl.py`

## Setup

This is a **Poetry** project.

1) Install dependencies

```bash
poetry install
```

2) Run tests

```bash
poetry run pytest
```

## CLI usage

The CLI entrypoint is configured in `pyproject.toml`:

```toml
[tool.poetry.scripts]
jaxdl = "jax_deep_learning.adapters.left.cli:app"
```

### Train on TFDS (example: MNIST)

```bash
poetry run jaxdl train --dataset-kind tfds --tfds-name mnist --epochs 3 --batch-size 64
```

### Optimizer tuning (AdamW)

Both `train` and `kaggle-diabetes` expose common `optax.adamw(...)` hyperparameters:

- `--lr` (learning rate)
- `--weight-decay`
- `--adamw-b1`, `--adamw-b2`
- `--adamw-eps`, `--adamw-eps-root`
- `--adamw-nesterov/--no-adamw-nesterov`

Example:

```bash
poetry run jaxdl kaggle-diabetes --lr 3e-4 --weight-decay 1e-3 --adamw-b1 0.9 --adamw-b2 0.995
```

### Train on an .npz dataset

```bash
poetry run jaxdl train --dataset-kind npz --npz-path /path/to/data.npz --epochs 5
```

### Model selection

Both `train` and `kaggle-diabetes` support `--model-kind`:

- `mlp`
- `derf-mlp`

For `kaggle-diabetes`, there is also a tabular-specific option:

- `tabular-embed-mlp` (categorical embeddings instead of one-hot)

Example:

```bash
poetry run jaxdl train --dataset-kind tfds --tfds-name mnist --model-kind derf-mlp
```

For Kaggle diabetes (categorical embeddings):

```bash
poetry run jaxdl kaggle-diabetes --model-kind tabular-embed-mlp --embed-dim 8
```

## Kaggle: diabetes playground workflow

This command expects:

- `data/train.csv` (must contain target column `diagnosed_diabetes`)
- `data/test.csv`

It trains a model, prints validation ROC AUC, and writes a submission CSV with:

- header: `id,diagnosed_diabetes`
- probabilities in `[0, 1]`

Run:

```bash
poetry run jaxdl kaggle-diabetes --data-dir data --epochs 10 --batch-size 512 --out-path submission.csv
```

Useful flags:

- `--max-train-rows N` / `--max-test-rows N`: fast local smoke runs.
- `--log-path logs/kaggle_diabetes.jsonl`: JSONL logs (default is enabled).
- `--zip/--no-zip`: optionally zip the submission file.
- `--add-noise/--no-add-noise`: add Gaussian noise to numerics for data augmentation.
- `--noise-std`: std dev of the noise (default 0.1).

## Logging (JSONL)

If `--log-path` is set (or left as default for `kaggle-diabetes`), the CLI will append structured records like:

```json
{"ts":"...","step":7,"metrics":{"epoch":1,"test/loss":0.68,"test/acc":0.57,"global_step":7}}
```

This makes it easy to parse runs later (for plots, comparisons, dashboards, etc.).

## Notes & troubleshooting

### CPU vs CUDA

This repo is configured **CPU-first**. If you see errors like:

- `cuInit(0) failed`
- `xla_cuda12.initialize()`

…it typically means CUDA plugin wheels were installed on a machine without CUDA libraries.
Use plain `jax` (no CUDA extras) and ensure `JAX_PLATFORMS=cpu`.

### “No imports inside functions” convention

The CLI is written to avoid imports inside command functions. Dependency injection is configured at startup via `inject` (see `src/jax_deep_learning/adapters/left/inject_config.py`).

## License

MIT

