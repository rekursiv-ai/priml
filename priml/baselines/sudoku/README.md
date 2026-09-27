# Sudoku

Solving 9x9 sudoku as a structured-prediction task: read a grid of tokens,
predict the completed grid. A puzzle counts only if every cell is right, which
makes this a clean test of whether a network can hold a global constraint
rather than score well on average.

The dataset is Sudoku-Extreme, subsampled to few distinct puzzles and expanded
with many validity-preserving transformations of each. That is the point of the
benchmark: with a thousand puzzles and a million copies, memorizing instances
does not help and learning the rules does.

## The 2x2

Two mechanisms vary independently, so `exp000`-`exp003` form a 2x2 rather
than a chain:

|           | transformer | MLP-mixer |
|-----------|-------------|-----------|
| plain     | `exp000`    | `exp001`  |
| recurrent | `exp002`    | `exp003`  |

Both axes are config VALUES -- a slot filled differently -- so all four share
one model class, one train step, and one dataset. Nothing differs except the
thing each experiment names.

`exp000` is the naive recipe and is never edited; improvements are forks, so a
number measured against it stays comparable.

## Recurrence

`exp002` and `exp003` re-apply the block stack over a carried latent state
instead of running it once. Constraint propagation is iterative -- filling one
cell licenses filling the next -- so a fixed-depth network must learn in one
pass what a recurrence can unroll.

Two properties make it affordable, both pinned by tests:

- **Gradient cost is flat in depth.** All but the last cycle run under
  `no_grad`, so a 32-cycle forward backpropagates through one cycle. Measured:
  the backward graph is the same size at 1, 8, and 32 cycles.
- **Blocks are post-norm.** A recurrence feeds a block its own output, and
  pre-norm leaves the residual stream unnormalized -- harmless in one pass,
  compounding when fed back. Measured at hidden 32 over 5 steps: pre-norm drove
  the carried latent to 413.6 and the loss to 4473; post-norm held it at 2.9
  with the loss falling monotonically.

Recurrent runs also spend a variable number of steps per puzzle. Each occupies
a slot in a pool and takes one step per call, carrying its state forward until
the model's halt head says it is done; an easy puzzle leaves early, a hard one
keeps its slot, and the batch shape never changes.

## The recursive-model ladder

`exp004`-`exp010` build the tiny recursive model (TRM) recipe, one
mechanism per experiment. `exp011`-`exp014` fork the finished recipe and
change only how an answer is produced and accepted.

| Exp | Parent | Change | Result |
|---|---|---|---|
| `exp004` | -- | TRM baseline, 62k-step anneal | 0.61 (2K) |
| `exp005` | `exp004` | + row/column/box position tables | 0.70 (2K) |
| `exp006` | `exp005` | + QK-norm, 12k steps | ~5x faster |
| `exp007` | `exp006` | + ACT 32, EMA .999, cardinality loss, digit-only augmentation | 0.78 (2K) |
| `exp008` | `exp007` | + corrupted-feedback repair | 0.82 (2K) |
| `exp009` | `exp008` | + deep recurrence (slow 6 / fast 9) | 0.86-0.90 (2K) |
| `exp010` | `exp009` | + 19,500-step anneal | 0.93 (2K); 0.9624 full |
| `exp011` | `exp010` | + hypothesis-pinning search at eval | 0.99920 full |
| `exp012` | `exp010` | + nine-view agreement lock at eval | 1.0 full |
| `exp013` | `exp010` | + verifier-committee sieve at eval | 1.0 full (frozen `exp010`) |
| `exp014` | `exp012` | three training seeds instead of nine views | TBD |

2K is the first 2,000 test puzzles, which underestimates the full set of
422,786. Each experiment's docstring states its hypothesis and the
measurement behind its result.

`exp011` scores `exp010`'s final checkpoint. To skip training, place the
released one where it reads:

```bash
uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.download_checkpoints --names exp006:exp010
```

`exp012`-`exp014` train their generator, then evaluate it. `exp013` also
harvests its verifiers' training candidates from `exp010`'s final
checkpoint, so it needs that checkpoint first.

## Slots worth knowing

- `step.model.embedding.channels_out` -- what is added to the token embedding.
  Sudoku uses learned row/column/box positions, and the recurrent rungs add the
  previous step's own prediction. A differently shaped puzzle is a different
  channel list, not an edit here.
- `step.model.block` -- how tokens mix. Any module taking `(x, *args, **kwargs)`.
- `step.model.recurrence` -- `None` runs the stack once.
- `step.act` -- the pool. Every knob meaningless without recurrence lives here,
  so a plain run's config does not carry them.
- `step.optimizer` -- Muon on the reasoning matrices, AdamW on tables and heads,
  partitioned by name.

## Running

```bash
uv --quiet run --frozen python -m priml.baselines.sudoku.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.sudoku.experiments.exp000
```

`exp_smoke` answers only "is the data prepared and does the loop run"; its
accuracy is meaningless.
