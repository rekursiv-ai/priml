# priml

[![PyPI version](https://img.shields.io/pypi/v/priml.svg)](https://pypi.org/project/priml/)
[![CI](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

priml is a library of composable PyTorch building blocks — models, optimizers, losses, metrics,
and a step-based training loop — extracted from an internal training monorepo so individual
pieces (a model layer, a launcher, a checkpointing scheme) can be reused without pulling in the
whole stack. It exists to give training code a shared, tested vocabulary instead of every
experiment reinventing its own training loop and config plumbing.

> *If you want composable PyTorch building blocks — a layer, an optimizer, a training loop —
> that you can lift out one at a time instead of adopting a whole framework.*

## Install

```bash
pip install priml
```

```bash
uv add priml
```

Requires Python 3.12 or newer. HuggingFace import/export helpers (`priml.hub`, the Qwen3/Kimi-K2
loaders) are behind an optional extra:

```bash
pip install "priml[hub]"
```

## Quickstart

priml ships a generic launcher, `priml.launch`, that imports a config-returning function, builds
its config with [`configgle`](https://pypi.org/project/configgle/), and runs the resulting job. A
job is any object exposing `run(*args)`. This mirrors the worked example documented in
`priml/launch.py`:

```python
# my_experiment.py
from configgle import Fig, Makeable


class HelloJob:
    class Config(Fig["HelloJob"]):
        greeting: str = "hello"

    def __init__(self, config: Config) -> None:
        self.greeting = config.greeting

    def run(self, *args: str) -> None:
        print(self.greeting)


def experiment() -> Makeable[HelloJob]:
    return HelloJob.Config()
```

```bash
uv run python -m priml.launch my_experiment.experiment --override greeting=hi
```

`--override PATH=VALUE` sets a (possibly nested) config field, cast to its declared type, and is
repeatable. See `priml/launch.py` for multi-GPU launches under
`python -m torch.distributed.run` and resume support.

For an actual training job, `priml.train.train_loop.TrainLoop` is itself a valid launch target —
its `Config` composes a `priml.train.train_step.TrainStep` (model + optimizer + loss), a dataset,
and metrics:

```python
from configgle import Makeable

from priml.train.train_loop import TrainLoop


def experiment() -> Makeable[TrainLoop]:
    return TrainLoop.Config(max_steps=1_000)
```

Wire in a real model (`priml.model`), loss (`priml.loss`), and dataset (`priml.data`) in place of
the defaults — see the class docstrings in `priml/train/train_step.py` and
`priml/train/learnable.py` for fully worked `Config(...)` examples.

## What's inside

- **models** (`priml.model`) — composable model definitions and building blocks (attention,
  transformer, MoE, MLA, Mamba-style layers, and more).
- **optimizers** (`priml.optimizers`) — optimizer implementations for training (Muon, AdamATan2,
  SignSGD, Newton).
- **loss** (`priml.loss`) — loss functions (cross-entropy, diffusion, LPIPS, GAN, weighted losses).
- **metrics** (`priml.metrics`) — evaluation metrics.
- **math** (`priml.math`) — numerical and math utilities.
- **train** (`priml.train`) — `TrainLoop`/`TrainStep` orchestration: checkpointing, parallelism,
  EMA, profiling, schedules.
- **data** (`priml.data`) — dataset protocol (`DatasetProtocol`) plus a synthetic `DummyDataset`
  used as the training loop's default/test dataset. Real dataset pipelines are not part of the
  public package yet.
- **inference** (`priml.inference`) — inference helpers.

## Platform support

`pip install priml` and the pinned development environment (`uv sync`) both work on x86_64 and
aarch64 Linux. On Linux, `torch`/`torchvision` resolve from the `pytorch-cu128` index on both
architectures (full CUDA support, including Grace/GB10-class aarch64 hosts). `torchao` ships CUDA
kernels for x86_64 only, so aarch64 uses its pure-python `+cpu` build — functionally equivalent,
with torchao's optional custom kernels unavailable until upstream publishes aarch64 CUDA wheels.

## Development

```bash
uv sync --all-groups
uv run pytest
```

Before opening a pull request, also run:

```bash
uv run ruff check --no-fix --no-cache .
uv run ruff format --check --no-cache .
uv run codespell .
uv run ty check
uv run basedpyright priml
uv build
```

Tests are organized into tiers via pytest markers (`pyproject.toml`). The default `uv run pytest`
excludes the slower/gated ones (`-m 'not ci_smoke and not cuda and not integration and not
performance'`); opt back in with `-m` when you need them, e.g. `uv run pytest -m cuda` on a CUDA
box. See `CONTRIBUTING.md` for the full local-validation and public-contribution flow.

## See also

Sibling libraries in the [rekursiv-ai](https://github.com/rekursiv-ai) family:

- [configgle](https://github.com/rekursiv-ai/configgle) — Hierarchical experiment configuration in typed pure-Python dataclasses instead of YAML.
- [trackinizer](https://github.com/rekursiv-ai/trackinizer) — Centralized agent database for tracking inquiries, work, and the evidence behind conclusions.
- [sagent](https://github.com/rekursiv-ai/sagent) — The self-mutating multi-provider coding-agent CLI and typed Python library.
- [madcatter](https://github.com/rekursiv-ai/madcatter) — Rich-based Markdown renderer for the terminal; ships the `mdcat` CLI.
- [wesearch](https://github.com/rekursiv-ai/wesearch) — Web search, resilient page fetch, and scholarly-paper lookup without a browser stack.

## License

Apache License 2.0
