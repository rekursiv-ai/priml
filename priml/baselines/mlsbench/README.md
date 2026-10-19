# MLS-Bench

[MLS-Bench](https://github.com/Imbernoulli/MLS-Bench) tasks, re-implemented as
native priml baselines. Each task asks for one edit to a training script: a
loss, a schedule, a pooling layer. Here that editable function is a **slot** of
a config, so a new idea is an `expNNN` fork that changes one value.

- `exp000` is the task's pristine recipe, end to end in one config, and frozen.
- Each published baseline is a fork that changes only the task's slot; a test
  enforces that.
- A capture script runs the pristine upstream script and the port with the
  same seed and compares them with exact equality. Its values are replayed in
  the default test tier, so CI checks parity without a clone.

MLS-Bench has no license, so no upstream source is copied. Every port pins
upstream commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

## Scope

A task is in scope when its whole recipe -- data, training and evaluation --
runs on one machine. Upstream runs each task in one sandbox of at most 8 H100s,
most of them 1 to 3, so every task passes except the two whose evaluation calls
a hosted model API: `agent-tool-reasoning` and `mas-topology`. The robotics
tasks run in simulation, so they are in scope.

## Coverage

MLS-Bench has 140 tasks in 12 areas. Each area is one subpackage, named by its
upstream area code; an area not yet started holds a TODO README that lists its
tasks.

| Area | Package | Tasks | In scope | Ported |
|---|---|---|---|---|
| DL | [`dl/`](dl/) | 11 | 11 | 11, every setting |
| LM | [`lm/`](lm/README.md) | 18 | 16 | 0 |
| CAL | [`cal/`](cal/README.md) | 14 | 14 | 0 |
| RL | [`rl/`](rl/README.md) | 13 | 13 | 0 |
| Opt | [`opt/`](opt/README.md) | 13 | 13 | 1 |
| Rob | [`rob/`](rob/README.md) | 12 | 12 | 0 |
| V&G | [`vg/`](vg/README.md) | 11 | 11 | 0 |
| Sys | [`mlsys/`](mlsys/README.md) | 10 | 10 | 0 |
| Sci | [`sci/`](sci/README.md) | 10 | 10 | 0 |
| TS | [`ts/`](ts/README.md) | 10 | 10 | 0 |
| SCR | [`scr/`](scr/README.md) | 10 | 10 | 0 |
| TL | [`tl/`](tl/README.md) | 8 | 8 | 4 |
| **Total** | | **140** | **138** | **16** |

`Sys` is packaged as `mlsys/` so it does not read as the standard library's
`sys`.

## DL packages

| Package | Task | Slot |
|---|---|---|
| [`dl/classification_loss/`](dl/classification_loss/README.md) | `cv-classification-loss` | `step.loss` |
| [`dl/lr_schedule/`](dl/lr_schedule/README.md) | `dl-lr-schedule` | `step.schedule` |
| [`dl/pooling/`](dl/pooling/README.md) | `cv-pooling-aggregation` | `step.model.pool` |
| [`dl/normalization/`](dl/normalization/README.md) | `dl-normalization` | `step.model.norm` |
| [`dl/initialization/`](dl/initialization/README.md) | `dl-weight-initialization` | `step.model.init_conv`, `init_head`, `init` |
| [`dl/activation/`](dl/activation/README.md) | `dl-activation-function` | `step.model.activation` |
| [`dl/regularization/`](dl/regularization/README.md) | `dl-regularization` | `step.regularization` |
| [`dl/multitask/`](dl/multitask/README.md) | `cv-multitask-loss` | `step.mtl_loss` |
| [`dl/weighting/`](dl/weighting/README.md) | `cv-sample-weighting` | `step.loss.class_weight` |
| [`dl/residual/`](dl/residual/README.md) | `dl-residual-connection` | `step.model.block` |
| [`dl/augmentation/`](dl/augmentation/README.md) | `cv-data-augmentation` | `dataset.augment` |

[`dl/vision/`](dl/vision/README.md) is no task: it holds the data, model, step
and `exp000` recipe every DL task reuses, and its `settings.py` switches a
recipe to each other upstream setting.

## Other ported packages

| Package | Task | Slot |
|---|---|---|
| [`opt/gradient_compression/`](opt/gradient_compression/README.md) | `optimization-gradient-compression` | `step.compressor` |
| [`tl/backdoor_defense/`](tl/backdoor_defense/README.md) | `security-backdoor-defense` | `defense` |
| [`tl/machine_unlearning/`](tl/machine_unlearning/README.md) | `security-machine-unlearning` | `method` |
| [`tl/membership_inference_defense/`](tl/membership_inference_defense/README.md) | `security-membership-inference-defense` | `step.loss` |
| [`tl/poison_robust_learning/`](tl/poison_robust_learning/README.md) | `security-poison-robust-learning` | `step.loss` |

## Proof against upstream

```bash
git clone https://github.com/Imbernoulli/MLS-Bench.git /opt/scratch/mlsbench
git -C /opt/scratch/mlsbench checkout 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c
```

Each package's `scripts/capture_reference.py --root /opt/scratch/mlsbench`
reports every mismatch and writes the goldens its `reference_test.py` replays.
