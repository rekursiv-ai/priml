# Maze-Hard

Marking the shortest path through a 30x30 maze: read a grid of walls, open
cells, a start, and a goal, and predict the same grid with one shortest path
marked. A maze counts only if all 900 cells are right. Training sees 1,000
mazes; scoring uses the 1,000 test mazes of `sapientinc/maze-30x30-hard-1k`.

Most mazes have several shortest paths, and only the stored one scores. The
stored labels follow one fixed rule -- breadth-first search expanding down,
left, right, then up -- so the task includes learning that tie-break.
Rotating or reflecting a maze moves its label to a path the rule would not
pick in the new frame; `exp001` onward relabel each training view in its own
frame instead.

## Table of contents

- [Reproduce](#reproduce)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
  - [Experiments](#experiments)
  - [Where files go](#where-files-go)
- [Reproducing the results](#reproducing-the-results)
  - [H100, 1 hour](#h100-1-hour)
  - [H200, 1 hour](#h200-1-hour)
  - [H100, 4 hours](#h100-4-hours)
  - [References](#references)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Files](#files)

## Reproduce

### Prerequisites

- One GPU per experiment. The record ran on H100s; `exp_smoke` runs on any
  device.
- The `hub` extra (`pip install priml[hub]`), and network access to Hugging
  Face for the dataset download.
- Under 1 GB of disk for both prepared trees.

### Data

From a checkout with dependencies installed:

```bash
uv --quiet run --frozen python -m priml.baselines.maze.scripts.prepare_data
```

This downloads the source CSVs at a pinned revision, verifies their digests,
and writes two trees: `hard-1k`, each training maze in its eight dihedral
views with the stored label moved along, and `hard-1k-canonical`, the same
views relabeled. The test split is identical in both.

### Experiments

```bash
uv --quiet run --frozen python -m priml priml.baselines.maze.experiments.exp000
```

Replace `exp000` with the experiment's name. No experiment reads another's
checkpoint, so any order works once the trees exist. `exp_smoke` checks an
installation in seconds and is not scored.

| Experiment | Reads | Budget | Recorded exact, 1xH100 |
|---|---|---|---|
| `exp000` | `hard-1k` | 1 hour | 0.007 |
| `exp001` | `hard-1k-canonical` | 1 hour | 0.940 |
| `exp002` | `hard-1k-canonical` | 1 hour | 0.949 |
| `exp003` | `hard-1k-canonical` | 4 hours | 0.952 |

### Where files go

Runs resolve beneath `base_dir`, which defaults to `/opt/scratch`: experiment
`expNNN` reads `<base_dir>/datasets/maze/` and writes its checkpoints and
`metrics.json` to `<base_dir>/runs/maze/expNNN/`. To use another root, pass it
to both steps:

```bash
uv --quiet run --frozen python -m priml.baselines.maze.scripts.prepare_data --directory /data/scratch/datasets/maze
uv --quiet run --frozen python -m priml priml.baselines.maze.experiments.exp_smoke --override base_dir=/data/scratch
```

## Reproducing the results

Every rung trains on one H100 against a training-time budget: a change that
buys steps and a change that improves a step show up in the same number.
`exp001` and `exp002` move the record; `exp003` runs the one-hour best's
recipe for four hours.

`exp001` forks the reference recipe with every change the record's best line
made on the way to it. The record reached it through two record-raising rungs
on a hierarchical model this baseline does not carry, so the fork lists the
whole configuration difference.

Δ compares each recipe with its parent at the same budget. `—` means no
measured comparison is available. Results are the final checkpoint.

### H100, 1 hour

| Experiment | Change | Exact | Δ exact | Seeds |
|---|---|---:|---:|---|
| `exp000` | Reference TRM: two latents, 2-D rotary positions, AdamATan2, stored labels | 0.007 | — | 1 (0) |
| `exp001` | Canonical labels; URM recurrence with ConvSwiGLU; Muon; boundary loss | 0.940 | +0.933 | 1 (0) |
| `exp002` | One eight-pass application, five with gradient; fused shift taps | **0.949** | +0.009 | 1 (0) |

A replicate of `exp002` with every training seed at 1 finished at 0.937
(best 0.939).

### H200, 1 hour

The same configurations on one H200 each, seed 0. The H100 column is the
result above; best is the highest of the evaluations every 500 steps.

| Experiment | Exact | Best (step) | Steps | H100 |
|---|---:|---:|---:|---:|
| `exp000` | 0.000 | 0.003 (11,500) | 16,000 | 0.007 |
| `exp001` | 0.932 | 0.941 (12,500) | 15,575 | 0.940 |
| `exp002` | 0.945 | 0.952 (12,500) | 13,678 | 0.949 |

### H100, 4 hours

| Experiment | Change | Exact | Δ exact | Seeds |
|---|---|---:|---:|---|
| `exp003` | Three passes with gradient; four-hour horizon | **0.952** | — | 1 (0) |

`exp003` peaked at 0.957 at step 58,500. Its parent was not run for four
hours.

### References

- Jolicoeur-Martineau. [Less is More: Recursive Reasoning with Tiny
  Networks](https://arxiv.org/abs/2510.04871). The paper reports 0.853 on
  Maze-Hard under its own training budget.
- Gao et al. [Universal Reasoning Model](https://arxiv.org/abs/2512.14693).
- Jordan et al. 2024. [Muon: an optimizer for hidden
  layers](https://kellerjordan.github.io/posts/muon/).

## Evaluation

`eval/exact_accuracy` is the fraction of the 1,000 test mazes whose predicted
grid matches the stored label in every cell, scored every 500 steps from the
EMA weights. A valid shortest path other than the stored one counts as wrong.
The test split is never transformed or relabeled, so every rung is scored on
the same grids.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/maze
```

Every test runs on CPU. `train_step_test.py` replays `exp000`, `exp001`, and
`exp002`, shrunk by size only, bit for bit; `experiments_test.py` checks each
fork's configuration difference against its docstring.

## Files

| File | Responsibility |
|---|---|
| `experiments.py` | Experiment factories and the training loop's slots |
| `train_step.py` | The reference TRM step plus the boundary loss |
| `loss.py` | The path-boundary mask and its loss |
| `data.py` | Tokens, dihedral views, canonical labels, and the loader |
| `scripts/prepare_data.py` | Pinned download, verification, and both prepared trees |
