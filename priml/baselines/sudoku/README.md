# Sudoku

Solving 9x9 Sudoku-Extreme as structured prediction with a tiny recursive
model (TRM): read a grid of tokens, predict the completed grid. A puzzle counts
only if every cell is right. Experiment factories expose the model, optimizer,
data, search, and lock settings in one configuration tree.

For our [100% Sudoku blog
post](https://rekursiv.ai/blog/100-percent-accuracy-on-sudoku/), see
[Reproducing the blog post](#reproducing-the-blog-post) for the experiment
chain, results, and hardware.

## Table of contents

- [Run the baseline](#run-the-baseline)
- [Reproduce](#reproduce)
- [Reproducing the blog post](#reproducing-the-blog-post)
  - [Training, 2K proxy](#training-2k-proxy)
  - [Full set](#full-set)
  - [Reproduced on 1x H200](#reproduced-on-1x-h200)
  - [References](#references)
- [Evaluation](#evaluation)
- [Hard catalogs](#hard-catalogs)
- [Tests](#tests)
- [Files](#files)

## Run the baseline

From a checkout with dependencies installed:

```bash
uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.sudoku.experiments.exp000
```

This prepares Sudoku-Extreme under `/opt/scratch/datasets/sudoku-extreme`:
1,000 training puzzles, each expanded with 1,000 validity-preserving
transformations, and the 422,786-puzzle test split verbatim. `exp_smoke`
answers only "is the data prepared and does the loop run"; its accuracy is
meaningless.

## Reproduce

Every input below is produced by an earlier step or downloaded at a pinned
revision with a checked sha256; no step needs a file from the original runs.

**Prerequisites.** One NVIDIA GPU with bf16 (Ampere or newer) per rung; the
records ran on one H100 80GB each. No credentials. Disk: 1.8 GB for the
prepared dataset, under 1 GB per training run (`exp008` and `exp012` keep a
checkpoint per 1,000 steps, about 2 GB each).

**Data.** Sudoku-Extreme from Hugging Face at a pinned revision, digest
checked:

```bash
uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.prepare_data
```

**Smoke.** Check the prepared data and the loop in seconds, on any device;
its accuracy is meaningless:

```bash
uv --quiet run --frozen python -m priml priml.baselines.sudoku.experiments.exp_smoke
```

**Experiments.** Run each with

```bash
uv --quiet run --frozen python -m priml priml.baselines.sudoku.experiments.exp000
```

and the rung's name in place of `exp000`. Run a rung's inputs first; outputs
land beneath `/opt/scratch/runs/`:

| Experiment | Reads (run first) | Writes | Record, 1x H100 |
|---|---|---|---|
| `exp000`-`exp006` | the dataset | `runs/exp00N/` | 2K exact, table below |
| `exp007` | `exp006`'s final checkpoint | `runs/exp007/dumps/` | 0.99920 full set, about 2 h |
| `exp008` | `exp006`'s final checkpoint | `runs/exp008/`, `runs/harvest/harvest/`, `runs/exp008_generator/`, `runs/exp008_verifier_s{0,1,2}/` | 1.0 full set, 0 wrong locks |
| `exp009`, `exp010` | the dataset | `runs/exp009/`, `runs/exp010/` | 0.9650, 0.9770 2K |
| `exp011` | `exp006`, `exp010` final checkpoints; `exp008`'s verifiers | `runs/exp011/dumps/` | 1.0 full set, 1,598.6 GPU-s |
| `exp012` | the dataset (trains its own generator) | `runs/exp012/`, `runs/exp012_generator/` | 1.0 full set, 7.0 GPU-h |
| `exp013` | `exp006`, `exp010` final checkpoints; `exp008`'s harvest corpus | `runs/exp013/dumps/`, `runs/exp013_verifier_s{0,1,2}/` | 1.0 full set, 1,613.6 GPU-s |

A consumer derives each path from its parent's config: the parent's
`experiment_name` and final step, `runs/<name>/checkpoints/step_<steps>.pt`.

**The base_dir knob.** `--override base_dir=/your/root` relocates a rung's
runs and its dataset root together: runs land in `/your/root/runs/` and the
dataset is read from `/your/root/datasets/sudoku-extreme`. Prepare the data
there with `--directory /your/root/datasets/sudoku-extreme`, and pass the same
`base_dir` to every rung of a chain.

## Reproducing the blog post

The chain keeps the baseline, every experiment that set a new best, the
cheapest 100%, and three ways to lock an answer: a learned verifier committee
(`exp008`), agreement between symmetric views (`exp012`), and a committee
that encodes no Sudoku structure (`exp013`). Each factory forks a named
parent and states its changes, hypothesis, and measured result.

```
exp000  attention TRM baseline
  +-- exp001 .. exp006  one training mechanism per rung
        +-- exp007  hypothesis-pinning search at evaluation
        +-- exp008  verifier-committee sieve (harvest, fit, train, sieve)
        +-- exp009  winner-take-all latent hypotheses
        |     +-- exp010  winner recompute, three hypotheses, batch 1,152
        |           +-- exp011  staircase sieve: the cheapest 100%
        |                 +-- exp013  prior-free committee lock
        +-- exp012  nine-view agreement lock (train, then lock)
```

Every number was measured on one H100. Δ compares each rung with its parent;
`—` means no measured comparison is available.

### Training, 2K proxy

Exact accuracy on the first 2,000 test puzzles, scored during training. The
proxy underestimates the full set.

| Experiment | Change | Exact (2K) | Δ | Seeds |
|---|---|---:|---:|---|
| `exp000` | Attention TRM, 62k-step anneal | 0.61 (unstable; see below) | — | 1 |
| `exp001` | Row, column, and box position tables | 0.70 | +0.09 | 1 |
| `exp002` | QK-norm, 12k-step anneal | 0.71 at 10k (0.15 without) | ~5x faster | 1 |
| `exp003` | ACT 32, no smoothing, EMA 0.999, cardinality loss, digit-only augmentation | 0.78 (last 5) | +0.07 | 1 |
| `exp004` | Corrupted-feedback repair | 0.82 (last 5) | +0.043 to +0.059 (paired) | 3 |
| `exp005` | Deep recurrence, slow 6 / fast 9 | 0.8634, 0.8972 (last 5) | +0.059 to +0.075 (paired) | 2 |
| `exp006` | 19,500-step anneal | 0.9300 (last 5) | — | 1 |
| `exp009` | Two latent hypotheses, winner take all, 22k-step anneal | 0.9650 | +0.035 | 1 |
| `exp010` | Winner recompute, three hypotheses, batch 1,152, 10k-step anneal | 0.9770 | +0.012 | 1 |

### Full set

Exact accuracy on all 422,786 test puzzles. Evaluation rungs score frozen
checkpoints; `exp008` and `exp012` also train their own generator.

| Experiment | Change | Exact (full set) | Δ | Seeds |
|---|---|---:|---:|---|
| `exp006` | Deterministic pass | 0.9624 (406,892) | — | 1 |
| `exp007` | Hypothesis-pinning search, learned-q acceptance | 0.99920 (422,447; 321 wrong accepts) | +0.0368 | 1 |
| `exp008` | Verifier-committee sieve, 3,274 GPU-s | 1.0 (0 wrong locks) | +0.0376 | 1 |
| `exp009` | Deterministic pass | 0.9876297 (417,556) | +0.0252 | 1 |
| `exp010` | Deterministic pass | 0.9902007 (418,643) | +0.0026 | 1 |
| `exp011` | Staircase sieve under `exp008`'s committee, 1,598.6 GPU-s | 1.0 (0 wrong locks) | +0.0098 | 1 |
| `exp012` | Nine-view agreement lock, 7.0 GPU-h with training | 1.0 (0 wrong locks) | +0.0376 | 1 |
| `exp013` | Prior-free committee lock, 1,613.6 GPU-s | 1.0 (0 wrong locks) | 0 | 1 |

`exp011` reaches `exp008`'s 100% at 2.05x lower cost. `exp008`'s cost is its
sieve on `exp006`'s frozen checkpoint. `exp012`'s 7.0 GPU-h splits into
2,985.8 GPU-s of training, 1,121.8 of screens, and 20,973.4 of the full-set
lock. `exp013`'s recorded committee was fit on an earlier harvest corpus; the
factory fits its members on `exp008`'s.

### Reproduced on 1x H200

The same factories on one H200 each, each factory's own seed; seed-1 rows add
`--override seed=1`. The H100 column is the record above; last 5 is the mean
of the final five 2K evaluations.

| Experiment | Exact (2K, final) | Last 5 | Seconds | H100 |
|---|---:|---:|---:|---:|
| `exp000` | 0.173 | 0.173 | 3,121 | 0.61 |
| `exp000`, seed 1 | 0.1385 | 0.135 | 3,116 | 0.61 |
| `exp001` | 0.7105 | 0.708 | 3,372 | 0.70 |
| `exp002` | 0.7175 (0.698 at 10k) | 0.712 | 643 | 0.71 at 10k |
| `exp003` | 0.7185 | 0.727 | 694 | 0.78 |
| `exp003`, seed 1 | 0.770 | 0.781 | 834 | 0.78 |
| `exp004` | 0.7905 | 0.793 | 906 | 0.82 |
| `exp004`, seed 1 | 0.7955 | 0.803 | 733 | 0.82 |
| `exp005` | 0.8915 | 0.887 | 2,452 | 0.8634, 0.8972 |
| `exp006` | 0.923 | 0.9245 | 3,825 | 0.9300 |
| `exp009` | 0.971 | 0.967 | 7,282 | 0.9650 |
| `exp010` | 0.968 | 0.963 | 13,729 | 0.9770 |

Full-set rungs score every one of the 422,786 test puzzles:

| Experiment | Exact (full set) | Wrong | Seconds | H100 |
|---|---:|---:|---:|---:|
| `exp007` | 0.99957 (422,604) | 142 accepted | 8,337 | 0.99920 (422,447; 321 wrong) |

- `exp000` is unstable: its 0.61 record is one clean seed. Both reproduced
  seeds diverged (seed 0 at 18k steps, 0.247 to 0.012; seed 1 at 8k, 0.128 to
  0.0) and only partly recovered. `exp002`'s QK-norm removes the instability.
- `exp003` and `exp004` seed 0 land 5.3 and 2.7 pp below their H100 records
  (last 5); seed 1 lands within 0.1 and 1.7 pp, so the gap is seed spread.
  Repair adds +6.6 and +2.2 pp over `exp003`, paired by seed.
- `exp008` and `exp011` through `exp013` are running; their rows land here.

### References

- Jolicoeur-Martineau. [Less is More: Recursive Reasoning with Tiny
  Networks](https://arxiv.org/abs/2510.04871).
- Henry et al. [Query-Key Normalization for
  Transformers](https://arxiv.org/abs/2010.04245).
- Yang, Lee & Park. [Learning to Solve Constraint Satisfaction Problems with
  Recurrent Transformer](https://arxiv.org/abs/2307.04895).
- Austin et al. [Structured Denoising Diffusion Models in Discrete
  State-Spaces](https://arxiv.org/abs/2107.03006).
- [Speed is Confidence](https://arxiv.org/abs/2601.19085).

## Evaluation

A puzzle is solved only if every cell matches its unique solution. No solver
reads a label: labels join only in the reported metrics.

- **Hypothesis-pinning search** (`exp007`) rolls the model to full depth,
  accepts a grid when its halt logit clears 7.875 at ACT steps 24, 28 and 32,
  and otherwise pins the most uncertain cell's likeliest digits as extra
  givens and re-runs.
- **Locks** decide which grids retire as final answers. The verifier
  committee accepts a grid only when all three verifiers score it above 0;
  the agreement lock accepts a grid two symmetric views produce identically.
  A wrong lock is a retired grid that is not the solution.
- **Sieves** (`exp008`, `exp011`, `exp013`) offer each round's grids to the
  lock and run the next, costlier round only on the puzzles still open; a
  puzzle still open after the last round takes the modal grid over every
  round.

## Hard catalogs

The blog's 99,768-puzzle stress set assembles five public hardest-sudoku
catalogs into one evaluation dataset with per-row provenance:

```bash
uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.build_hard_catalogs --workers 32
```

Exact duplicates across catalogs keep one row, so the published dataset holds
only unique puzzles. An exhaustive enumerator labels every row and certifies
it has exactly one solution.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/sudoku
```

Tensor goldens replay the representative recipes' training steps (`exp000`,
`exp006`, `exp009`, `exp010`) and each evaluation runner bit for bit at tiny
size on CPU. `exp009`, `exp010`, and the staircase sieve's selection round
matched the implementations that produced the recorded results before their
goldens were minted.

## Files

| File | Responsibility |
|---|---|
| `experiments.py` | Experiment factories |
| `trainer.py` | The TRM trainer: ACT pool, optimizers, EMA, evaluation |
| `routing.py`, `losses.py` | The trainer's routing slot and its training losses |
| `latent_recursion.py` | The tiny recursive model |
| `hypotheses.py` | Winner-take-all training over several latent hypotheses |
| `views.py` | Exact Sudoku symmetries and the frozen nine-view table |
| `rollout.py`, `search.py`, `search_dump.py` | Frozen-model rollouts, hypothesis-pinning search, and its packed dump rows |
| `verifier.py`, `verifier_data.py`, `verifier_fit.py` | The learned verifier, its training stream, and its fit |
| `harvest.py` | The verifier's candidate corpus, harvested from a frozen generator |
| `hps_eval.py`, `agreement_lock.py`, `sieve.py` | The full-set evaluation runners |
| `reproduction.py` | The train-then-lock pipelines behind `exp008` and `exp012` |
| `runner.py` | Checkpoint loading and output plumbing the runners share |
| `testing.py` | Fixtures and tiny configs the runners' tests share |
| `puzzle_data.py` | Puzzle datasets and augmentation |
| `act.py`, `embedding.py`, `model.py`, `prefix.py`, `train_step.py`, `data.py`, `metric.py` | Pieces shared with other baselines |
| `scripts/prepare_data.py` | Download and build Sudoku-Extreme |
| `scripts/build_hard_catalogs.py` | Build the hard-catalog evaluation dataset |
