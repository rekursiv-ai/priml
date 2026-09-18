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

### Computational Cost

Priml has rich support for understanding where compute is (theoretically) being
spent.

For example,

```python
from priml import cost as theoretical
from priml.baselines.sudoku import experiments as sudoku

sudoku.exp000().finalize().step.model.cost(batch_size=64, dtype=torch.bfloat16)
#   Cost(params=6841858, params_active=6824450, bytes_state=8192)
#                        flops flops  bytes bytes
#                         bf16 int64   bf16 int64 intensity
#   primal  matmul      27.94M     - 102.7K     -       272
#   primal  elementwise 62.13K     -   175K     -    0.3551
#   primal  reduction   9.208K     -  18.7K     -    0.4923
#   primal  selection        -     - 2.144K 8.375         -
#   adjoint matmul      55.88M     - 205.4K     -       272
#   adjoint elementwise 75.94K     - 280.5K     -    0.2708
#   adjoint reduction   7.152K     -  14.5K     -    0.4934
#   adjoint selection      536     - 3.224K 8.375    0.1658
#   total               83.97M     - 802.1K 16.75     104.7

sudoku.exp001().finalize().step.model.cost(batch_size=64, dtype=torch.bfloat16)
#   Cost(params=4869122, params_active=4851714, bytes_state=0)
#                        flops flops  bytes bytes
#                         bf16 int64   bf16 int64 intensity
#   primal  matmul      22.03M     - 96.88K     -     227.4
#   primal  elementwise   116K     - 375.4K     -    0.3089
#   primal  reduction   16.65K     - 33.54K     -    0.4965
#   primal  selection        -     - 2.144K 8.375         -
#   adjoint matmul      44.06M     - 193.8K     -     227.4
#   adjoint elementwise 161.6K     -   750K     -    0.2154
#   adjoint reduction   17.16K     - 34.58K     -    0.4962
#   adjoint selection      536     - 3.224K 8.375    0.1658
#   total               66.41M     -  1.49M 16.75     44.58

theoretical.peak()["rtx5090", torch.bfloat16]
#   Cost(params=0, params_active=0, bytes_state=0)
#                flops  bytes intensity
#   matmul      209.5T 1.792T     116.9
#   elementwise 104.8T 1.792T     58.48
#   reduction   104.8T 1.792T     58.48
#   selection   104.8T 1.792T     58.48
#   sort        104.8T 1.792T     58.48

theoretical.peak()["h100", torch.bfloat16]
#   Cost(params=0, params_active=0, bytes_state=0)
#               flops bytes intensity
#   matmul       989T 3.35T     295.2
#   elementwise   67T 3.35T        20
#   reduction     67T 3.35T        20
#   selection     67T 3.35T        20
#   sort          67T 3.35T        20
```

Notice that `272/295.2-1=-.078591` i.e. `exp000` (Transformer) is within 8% of
optimal arithmetic intensity for an H100 versus `exp001` (MLP Mixer) which is
`227.4/295.2-1=-.229675` or 23% suboptimal. `utilization` computes exactly this
ratio per cell -- `1` is the roofline knee -- and, given the step's
`duration_sec`, the achieved fraction of the roofline ceiling (MFU for a
matmul cell):

```python
theoretical.utilization(
    sudoku.exp000().finalize().step.model.cost(batch_size=64, dtype=torch.bfloat16),
    device="h100",
    seq_len=81,
    batch_size=64,
)
#   Cost(params=0, params_active=0, bytes_state=0)
#                           bf16
#   primal  matmul        0.9214
#   primal  elementwise  0.01775
#   primal  reduction    0.02462
#   adjoint matmul        0.9214
#   adjoint elementwise  0.01354
#   adjoint reduction    0.02467
#   adjoint selection   0.008314
#   total                  1.932
```

The reason for the different intensity: the MLP Mixer has about 29% fewer
parameters and does about 21% fewer matmul flops (`22.03M` vs `27.94M`), but
its matmul bytes fall only 6% (`193.8K` vs `205.4K`): a smaller model that
places about the same load on the memory bus. Ie, the params/bytes_moved
shakes out as:
- transformer: `6841858 / 205.4K = 33.309922`
- mlpmixer:    `4869122 / 193.8K = 25.124468`

Meaning: in terms of memory transfer the transformer is getting ~33% better
deal on parameters per bus access. Outside the matmul silo the mixer moves
nearly twice the bytes (`1.49M` vs `802.1K` in total), so its whole-model
intensity is `44.58` against the transformer's `104.7`.


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

## Citing

If you find our work useful, please consider citing:

```bibtex
@misc{rekursivai2026priml,
      title={Priml - Strongly typed configurable building blocks for A/B testing ML research.}
      author={Joshua V. Dillon and Dan Kondratyuk},
      year={2026},
      howpublished={Github},
      url={https://github.com/rekursiv-ai/priml},
}
```
