# ARC-AGI

Tiny recursive models on ARC-AGI-1 and ARC-AGI-2. Each task shows a few
input/output grid pairs and asks for the output of a held-out input. The
solver uses `priml/model/grid_solver.py` with ARC's values in its slots:
a 30x30 grid, twelve tokens, and a learned per-task table. Experiment
factories expose the model, optimizer, data, and evaluation in one
configuration tree.

For our [ARC-AGI blog post](https://rekursiv.ai/blog/pushing-limits-arc-agi/),
see [Reproduce](#reproduce) for the steps and
[Reproducing the blog post](#reproducing-the-blog-post) for the results.

## Table of contents

- [Reproduce](#reproduce)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
  - [Experiments](#experiments)
  - [Where files go](#where-files-go)
- [Reproducing the blog post](#reproducing-the-blog-post)
  - [ARC-AGI-1, 8xH100](#arc-agi-1-8xh100)
  - [ARC-AGI-2, 8xH100](#arc-agi-2-8xh100)
  - [References](#references)
- [CHARM validation](#charm-validation)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Files](#files)

## Reproduce

### Prerequisites

- One node with eight 80 GB GPUs for every numbered experiment. The record
  ran on H100s; `exp_smoke` runs on one GPU or on CPU.
- Disk: 6.8 GB for the ARC-AGI-1 tree, 7.4 GB for its spatial tree, and about
  9 GB for each ARC-AGI-2 tree an experiment builds.
- `git` and network access to GitHub. Every tree is built from
  [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
  at a pinned commit, which the builder verifies.
- Weights & Biases credentials are optional. Without them a run still writes
  its evaluation metrics to `metrics.json`.

### Data

Build each ARC-AGI-1 tree once, from a checkout with dependencies installed:

```bash
uv --quiet run --frozen python -m priml.baselines.arcagi.scripts.prepare_data --experiment exp000
uv --quiet run --frozen python -m priml.baselines.arcagi.scripts.prepare_data --experiment exp005
```

The first is the tree `exp000`-`exp004` and `exp_smoke` read; the second adds
the translated/rescaled views `exp005` and `exp006` read. The ARC-AGI-2
experiments build their own trees on first use.

CHARM's `exp009` and `exp010` read the augmented v2 Parquet files from the
[pinned data release](https://huggingface.co/datasets/water-vapor-vx/charm-arc-agi/tree/937316c8f766920724293600bdfd8ed3d45ebdc6).
Stage them beneath the configured dataset root:

```text
/opt/scratch/datasets/charm-release-937316c8/data/augmented/v2/
├── arc1_training_aug1000.parquet
├── arc1_eval_aug1000.parquet
├── rearc_aug100.parquet
├── arc2_training_aug1000.parquet
├── arc2_eval_aug1000.parquet
├── rearc2_aug100.parquet
└── concept_aug1000.parquet
```

### Experiments

Training experiments launch the same way on eight GPUs. No experiment reads
another's checkpoint, so any order works once its data exists.

```bash
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi.experiments.exp000
```

Replace `exp000` with the experiment's name. `exp_smoke` checks an
installation in seconds on one device, with no checkpoint and no score:

```bash
uv --quiet run --frozen python -m priml priml.baselines.arcagi.experiments.exp_smoke
```

| Experiment | Reads | Recorded pass@2 (step), 8xH100 |
|---|---|---|
| `exp000` | ARC-AGI-1 tree | 46.0 (370k) |
| `exp001` | ARC-AGI-1 tree | 58.125 (170k) |
| `exp002` | ARC-AGI-1 tree | 59.5 (370k) |
| `exp003` | ARC-AGI-1 tree | 67.625 (350k) |
| `exp004` | ARC-AGI-1 tree | 68.375 (340k) |
| `exp005` | ARC-AGI-1 spatial tree | 70.125 (380k) |
| `exp006` | ARC-AGI-1 spatial tree | 71.375 (280k) |
| `exp007` | ARC-AGI-2 tree, built on first use | 6.67 strict (170k) |
| `exp008` | ARC-AGI-2 spatial tree, built on first use | 17.50 strict (300k) |
| `exp009` | Released ARC-AGI-1 v2 Parquet mixture | TBD |
| `exp010` | Released ARC-AGI-2 v2 Parquet mixture | TBD |

`exp000`-`exp006` train 388,670 steps and `exp007`-`exp008` train 541,580;
a result is the best evaluation, every 10,000 steps. CHARM's `exp009` and
`exp010` train 1,600,000 and 2,380,000 steps. Launch them through the same
entry point:

```bash
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi.experiments.exp009
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.arcagi.experiments.exp010
```

Fresh full-training results for these two recipes remain unmeasured. The
[CHARM validation](#charm-validation) below records the completed port checks.

### Where files go

Everything resolves beneath one root, `base_dir`, which defaults to
`/opt/scratch`. Trees land in `<base_dir>/datasets/`; experiment `expNNN`
writes its checkpoints, `metrics.json`, and evaluation dumps to
`<base_dir>/runs/arcagi/expNNN/`. To use another root, pass it to both steps:

```bash
uv --quiet run --frozen python -m priml.baselines.arcagi.scripts.prepare_data --experiment exp000 --base-dir /data/scratch
uv --quiet run --frozen python -m priml priml.baselines.arcagi.experiments.exp_smoke --override base_dir=/data/scratch
```

## Reproducing the blog post

These experiments checkpoint the major improvements from the research campaign
described in [Pushing the Limits of Autonomous ML Science on
ARC-AGI](https://rekursiv.ai/blog/pushing-limits-arc-agi/). Each checkpoint
forks its named parent with one change; `experiments.py` states each change
and its result. `exp006` is the best single model.

Every number is pass@2 on the public evaluation set, from one run at seed 0
on one 8xH100 node, at the step where it peaked. Δ compares each rung with its
parent; `—` means no measured comparison.

### ARC-AGI-1, 8xH100

Rule B: the fraction of the 400 tasks solved. Training runs 388,670 steps.

| Experiment | Change | pass@2 (step) | Δ | Seeds |
|---|---|---:|---:|---|
| `exp000` | Reference TRM | 46.0 (370k) | — | 1 (0) |
| `exp001` | Muon body and gate-normalized SwiGLU | 58.125 (170k; crashed at 230k) | +12.125 | 1 (0) |
| `exp002` | Muon rate 0.01 | 59.5 (370k) | +1.375 | 1 (0) |
| `exp003` | One recurrent state, convolutional feed-forward, batch 96 | 67.625 (350k) | +8.125 | 1 (0) |
| `exp004` | Fused shift convolution and compiled evaluation | 68.375 (340k) | +0.75 | 1 (0) |
| `exp005` | Translation/scale augmentation and spatial test-time views | 70.125 (380k) | +1.75 | 1 (0) |
| `exp006` | QK-norm and corrupted prediction feedback | 71.375 (280k) | +1.25 | 1 (0) |

### ARC-AGI-2, 8xH100

Strict rule: a task counts only if every one of its test outputs is right,
over the 120 evaluation tasks. Both runs are mid-training snapshots of a
541,580-step horizon.

| Experiment | Change | strict@2 (step) | pass@2 | Δ vs `exp007` | Seeds |
|---|---|---:|---:|---:|---|
| `exp007` | `exp000` trained on ARC-AGI-2 | 6.67 (170k) | 7.36 | — | 1 (0) |
| `exp008` | `exp005` trained on ARC-AGI-2 | 17.50 (300k) | 20.556 | +10.83 | 1 (0) |

### References

- Jolicoeur-Martineau. [Less is More: Recursive Reasoning with Tiny
  Networks](https://arxiv.org/abs/2510.04871).
- Gao et al. [Universal Reasoning Model](https://arxiv.org/abs/2512.14693).
- Jordan et al. 2024. [Muon: an optimizer for hidden
  layers](https://kellerjordan.github.io/posts/muon/).
- Chollet. [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547).

## CHARM validation

During port validation, the original implementation and native priml
components evaluated the released
[CHARM](https://arxiv.org/html/2609.33270) checkpoints. The table below records
those completed runs. The shipped experiments reproduce training through
the standard launcher: `exp009` modifies URM
`exp003` for ARC-AGI-1 with a gated residual; `exp010` modifies `exp009` for
ARC-AGI-2 without the gate. Their training horizons are 1,600,000 and
2,380,000 updates. Use the data staging and standard training commands in
[Reproduce](#reproduce) to run these recipes.

| Benchmark | Checkpoint window | Source pass@2 | Native priml pass@2 |
|---|---|---:|---|
| ARC-AGI-1 | 1,510,000-1,590,000, nine checkpoints | 84.0% | 84.0% (6 H200, 5:10:00, 31.0 GPU-hours) |
| ARC-AGI-1 | 1,510,000-1,600,000, ten checkpoints | 83.25% | Not run; checkpoint 1,600,000 is absent from the release script |
| ARC-AGI-2 | 2,290,000-2,380,000, ten checkpoints | 46.6667% | 46.6667% (2 H200, 7:24:46, 14.8256 GPU-hours) |

The pinned ARC1 release script ends at checkpoint 1,590,000, so the
nine-checkpoint window is the primary comparison. The source ARC1 sweep used
six H200s for 4:56:54 (29.69 GPU-hours), including its extra tenth checkpoint;
the source ARC2 sweep used two H200s for 6:16:57 (12.565 GPU-hours). No matched
nine-checkpoint source runtime is reported. USD costs were not recorded.

Both native validation jobs completed with exit `0:0` before this module
reorganization. They evaluate released checkpoints; fresh full-size training
and full-size bit identity remain unmeasured. Evaluation uses 16 refinement
calls, global
batch 768, and votes across augmentations and checkpoints, with all votes
retained and no exponential decay. The native accumulated ARC1 pass@2 matches
the source at 84.0%, while its final-checkpoint score is 80.5%; ARC2's
accumulated score is 46.6667%, while its final-checkpoint score is 43.8889%.
Other pass@K aggregates and per-checkpoint scores differ from the source
results. Scores average held-out outputs within each puzzle, then the 400
ARC1 or 120 ARC2 evaluation puzzles. The comparable hill-climb scores above
are 71.375% on ARC1 and 20.556% on ARC2; the latter is an interim checkpoint.
Training and checkpoint voting budgets differ between these recipes. The tiny
exact-training replay is separate from these checkpoint evaluations.

The source revision is
[`728bf26903684ef5e8bcb476cb8e55e42cf35b9d`](https://github.com/water-vapor/charm/commit/728bf26903684ef5e8bcb476cb8e55e42cf35b9d),
and the data/checkpoint revision is
[`937316c8f766920724293600bdfd8ed3d45ebdc6`](https://huggingface.co/datasets/water-vapor-vx/charm-arc-agi/tree/937316c8f766920724293600bdfd8ed3d45ebdc6).

## Evaluation

Every test input is evaluated under up to 1,000 color and dihedral views, and,
from `exp005` on, two spatial views of each. Predictions are mapped back to the
canonical frame and voted; pass@K counts an input solved when its answer is
among the K most-voted grids. One wrong cell fails the grid.

`CanonicalPassK` reports three slices of one evaluation pass from `exp005` on:
the non-spatial views (the headline), every view at the non-spatial view
budget (`spatial_eq`), and every view (`spatial_big`). The ARC-AGI-2 rungs add
the strict and per-output rules.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/arcagi
uv --quiet run --frozen pytest priml/baselines/arcagi -m "compute_training or compute_distributed"
```

The first command runs the fast CPU tests. `train_step_test.py` replays a
bit-for-bit golden of two CPU training steps and one evaluation for `exp000`,
`exp001`, `exp003`, `exp005`, and `exp006`, shrunk by size only. The other
rungs share one of those steps, which `experiments_test.py` checks.
`experiments_test.py` also pins each rung's full configuration in
`testdata/<expNNN>.txt` and the one change each fork makes.

The CHARM replay shrinks the actual `exp009` configuration and runs three
updates against `testdata/exp009.pt`, minted from the pinned original implementation. It compares losses, model
parameters, optimizer state, EMA, carried state, and the next RNG draws
exactly. A sparse learning-rate perturbation must fail the same replay.

The second command runs the slower tier: `exp000`'s trajectory as minted from
the original implementation at port time (`testdata/train_step.pt`), and the
two-rank runs. `parity/` holds a second implementation of the ARC-AGI-2
reference step, kept to replay its single- and two-rank trajectories.

## Files

| File | Responsibility |
|---|---|
| `experiments.py` | Experiment factories and training loops |
| `model.py` | ARC rotary block, task table, and convolutional feed-forward |
| `../../model/grid_solver.py`, `../../model/recurrence.py` | Shared grid solver and recurrent schedules |
| `../../model/grid_embedding.py`, `../../model/prefix.py`, `../../model/conditioning.py` | Grid embedding, sparse prefixes, and task/augmentation composition |
| `train_step.py` | ACT training step, evaluation signals, data parallelism |
| `../../loss/act.py`, `../../loss/token.py` | ACT objective, token losses, and batch reductions |
| `data.py`, `puzzle_data.py` | Prepared arrays, v2 Parquet schema, and ordered pair/view indexing |
| `augmentation.py` | ARC views and CanvasEOS packing |
| `../../data/color_dihedral.py` | Shared D4 and color transforms |
| `metric.py` | Canonical and Parquet pass@K, strict and per-output rules |
| `signal_dump.py` | Per-view evaluation signals written beside a run |
| `scripts/prepare_data.py`, `scripts/build_dataset.py` | ARC-AGI-1 trees |
| `scripts/build_spatial_eval.py` | Spatial evaluation views |
| `scripts/build_arc2_dataset.py` | ARC-AGI-2 tree |
| `parity/` | Second reference implementation and its replay tests |
