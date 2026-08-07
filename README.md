# priml✴️

[![PyPI version](https://img.shields.io/pypi/v/priml.svg)](https://pypi.org/project/priml/)
[![CI](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

ML building blocks for training experiments.

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uv add priml

# Alternatively: python -m pip install priml
```

## What's inside

- **model** -- composable model definitions and building blocks.
- **optimizers** -- optimizer implementations for training.
- **loss** -- loss functions.
- **metrics** -- evaluation metrics.
- **math** -- numerical and math utilities.
- **train** -- the training loop and experiment scaffolding.
- **data** -- the data pipeline and dataset utilities.
- **inference** -- inference helpers.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for local validation and the public
contribution flow.

## See also

Sibling projects in the [rekursiv-ai](https://github.com/rekursiv-ai) family:

- [sagent](https://github.com/rekursiv-ai/sagent) — The self-mutating multi-provider coding-agent CLI and typed Python library.
- [trackinizer](https://github.com/rekursiv-ai/trackinizer) — Centralized agent database for tracking inquiries, work, and the evidence behind conclusions.
- [wesearch](https://github.com/rekursiv-ai/wesearch) — Web search, resilient page fetch, and scholarly-paper lookup without a browser stack.
- [madcatter](https://github.com/rekursiv-ai/madcatter) — Rich-based Markdown renderer for the terminal; ships the `mdcat` CLI.
- [configgle](https://github.com/rekursiv-ai/configgle) — Hierarchical experiment configuration in typed pure-Python dataclasses instead of YAML.
- [copybarista](https://github.com/rekursiv-ai/copybarista) — Bidirectional source sync for publishing OSS-ready trees from a monorepo.
- [sudoku](https://github.com/rekursiv-ai/sudoku) — Sudoku-Extreme solved end to end with a 7M-parameter recursive transformer.

## Citing

If you find our work useful, please consider citing:

```bibtex
@misc{rekursivai2026priml,
      title={Priml - ML building blocks for training experiments.},
      author={Joshua V. Dillon and Dan Kondratyuk},
      year={2026},
      howpublished={Github},
      url={https://github.com/rekursiv-ai/priml},
}
```
