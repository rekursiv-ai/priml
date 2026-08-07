---
name: priml-baseline
description: ALWAYS invoke this skill when writing, extending, or reviewing a priml baseline -- a dataset pipeline under priml/baselines/ with its data, model, train_step, experiments, and expNNN fork chain. Do not write baseline code -- invoke first.
---

# Priml Baselines

A baseline is one dataset end to end, written so that reading one is enough to
navigate the rest. It is public, pedagogical code: a reader learns priml from
it, and downstream work forks its `exp000` as a control.

## Core rule

**A reader must see WHAT CHANGED without computing anything.** Every rule below
follows from that. When two rules conflict, the one that makes the delta
visible wins -- including over DRY.

## Layout

One directory per dataset, named after the dataset. Required files, meaning the
same thing in every baseline:

```
priml/baselines/<dataset>/
  README.md
  __init__.py
  data.py            data_test.py
  model.py           model_test.py
  train_step.py      train_step_test.py
  experiments.py     experiments_test.py
  goldens/           bit-for-bit snapshots
  scripts/           prepare_data.py and other one-time tools
```

Extra modules are welcome when a file would otherwise sprawl -- `optimizer.py`,
`attention.py`, `metric.py`. Mirror priml's own layout when you name them, so a
reader who knows `priml/optimizers/` guesses `baselines/<dataset>/optimizer.py`
correctly.

Data staging lives in `scripts/prepare_data.py`, never in a config. Assume
`/opt/scratch` exists.

## exp000 and the fork chain

`exp000` is the **best naive recipe** -- what a strong first-year graduate
student writes without exotica. Not a toy, and not the state of the art. It is
frozen at birth: improvements land as forks, never as edits, so a number
measured against it stays comparable across releases.

Every later experiment forks a NAMED parent and applies ONE change:

```python
def exp002() -> Experiment:
    """exp001 + Muon on the convolution weights, SGD on the rest.

    Hypothesis:
      Muon orthogonalizes each update, making the step scale-invariant with
      respect to the weight matrix, which should suit convolution kernels.

    References:
      https://kellerjordan.github.io/posts/muon/
      Jordan et al. 2024. Muon: an optimizer for hidden layers.

    Results:
      TBD.
    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
    ...
```

- The summary line names the parent. The chain must be reconstructable without
  running anything.
- **Hypothesis / References / Results** are all three required. `Results: TBD.`
  until measured on reference hardware -- never a number from a laptop GPU.
- One delta per experiment. Two fields are one change only when they are
  inseparable (an optimizer and the schedule its published recipe prescribes);
  say so in the Hypothesis.
- A `Experiment` config class narrows the loop's slots once, so no factory
  needs an `isinstance` narrow to reach a field it is about to set:

```python
class Experiment(Makes["TrainLoop"], TrainLoop.Config):
    step: Cifar10TrainStep.Config = field(default_factory=Cifar10TrainStep.Config)
    dataset: Cifar10Data.Config = field(default_factory=Cifar10Data.Config)
```

## Configgle style

Read the `configgle` skill first; it is the authority. The rules baselines get
wrong most often:

**Inject capability, don't enumerate it.** A `Literal` naming implementations
is a missing injection -- it closes the set, so a caller wanting Lion must
patch the library:

```python
# Bad
optimizer_kind: Literal["adamw", "muon"] = "adamw"
conv_momentum: float = 0.6  # noise for every run that is not Muon

# Good
optimizer: Makeable[OptimizerBuilder] = field(default_factory=...)
```

`Literal` is right for a genuine mode (`"reflect"` vs `"constant"` padding),
wrong for anything a third party might implement.

**Mutate; do not nest.** Same line count, and each line's PATH says what it
configures:

```python
# Bad
cfg.step.optimizer = PartialConfig(
    muon_and_sgd,
    build_matrix=Muon.Config(),  # ... then lr, momentum
)

# Good
muon = cfg.step.optimizer = PartialConfig(muon_and_sgd)
muon.build_matrix = Muon.Config()
muon.build_matrix.lr = 0.24
muon.build_matrix.momentum = 0.6
```

Bind a local only to shorten a genuinely long path, never to rename a short
one. The chained form introduces a slot and names it in one breath:

```python
topk = cfg.metrics["accuracy"] = TopK.Config()
topk.k_values = [1]
```

**Read the parent's value; never restate a literal.** A recomputed constant is
indistinguishable from a deliberate override and goes stale silently:

```python
# Bad: is 512 this variant's change, or an echo of exp000?
cfg.max_steps = 8 * 50_000 // 512

# Good
cfg.max_steps = cfg.step.total_train_steps = 8 * cfg.num_steps_eval
```

**No pass-through locals.** `epochs = 8; cfg.max_epochs = epochs` adds a hop
for nothing -- assign the literal and read `cfg.max_epochs` afterward.

**No helper hiding the delta.** A `set_epochs(cfg, 8)` mutating four fields is
fewer lines and strictly worse: the reader must open it to learn whether the
schedule horizon moved. Inline the assignments in every factory, repetition and
all.

**Module constants need `Final`,** and only earn a name when the name carries
meaning the number does not (`NUM_TRAIN_SAMPLES`, not a bare batch size -- that
belongs on the config).

**Leave `seed` at its default** unless the experiment is specifically about
seed variance.

## Tests and goldens

CPU only; CI has no GPU. Target well under 100ms per test.

**Shrink by mutating the config.** It is configgle -- there is no framework to
build:

```python
config = Cifar10TrainStep.Config()
config.model = ResNet.Config()
config.model.channels = (8, 16)
config.model.blocks_per_stage = 1
config.device = "cpu"
```

Shrink SIZE only -- widths, depth, batch, steps. Never touch the recipe
(optimizer, schedule, loss, init), or the test stops covering the experiment.

**Goldens** (`priml/testing/bfb.py`) pin the arithmetic, which is what makes
`exp000` frozen in practice rather than by convention:

- one per model: `state_dict` after init, plus the forward output;
- one per optimizer stack: a few train steps, so loss, augmentation draws,
  schedule, and optimizer all reach the compared post-state.

Mint and replay inside `host_agnostic_numerics()`; assert `torch.equal`, never
`allclose`. Verify a golden BITES before trusting it: perturb a constant and
confirm the failure. Regenerating is a deliberate act -- it means the recipe
changed and its recorded result no longer describes the code.

**Experiment tests** assert the DELTA -- exactly which fields each fork
changes -- so "one change per experiment" is enforced rather than claimed. They
also assert construction reads no files: a config must build on a laptop with
neither the dataset nor a GPU.

## Commands

Every Python invocation, in code, docs, and your own shell, goes through
`uv --quiet run --frozen`. Never `python -c`, never a heredoc -- use Edit and
Write for file changes, and a repo-local script under `scripts/` for probes.

Commands in docstrings and READMEs stay on ONE line so they can be
copy-pasted; the line-length rule is exempt (`# noqa: E501` naming the reason).

In an exported `.md`, write the PUBLIC module path (`priml.baselines...`): the
copybarista transform rewrites the slash form in markdown but not the dotted
form, so the monorepo spelling would trip the leak check.

## CLI overrides are for the environment

`--override PATH=VALUE` adapts a run to the machine it lands on -- where data
lives, where output goes. Path fields are logical, resolved beneath `base_dir`,
so pass `/datasets/...` and not a full on-disk path.

Overriding a HYPERPARAMETER is bad form: it produces a result whose config
exists nowhere in the code, so it cannot be rerun or compared. Write a fork.
Do not show hyperparameter overrides in documentation.

## Comments

Is every word load-bearing? Delete the sentence that restates the code.
`cfg.num_steps_eval = steps_per_epoch` needs no `# evaluate once per epoch`. A
comment earns its place only when the reason is invisible from the code AND its
absence invites a plausible wrong edit -- then state the consequence:

```python
# Floors: a short final batch is not a whole step.
steps_per_epoch = NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
```

## Before finishing

1. Ask **"does this change imply changes elsewhere?"** A rule applied at one
   site and not its siblings is not applied. Grep for the pattern.
2. Run the gates: `ruff`, `ty`, `basedpyright`, `pytest`, then `pre-commit run
   --files ...` over everything changed.
3. Never weaken a gate to pass it.

## Common mistakes

| Mistake | Fix |
|---|---|
| `Literal` selecting an implementation | Inject a `Makeable[Protocol]` slot |
| Nested `PartialConfig(...)` kwargs | Mutate through the dotted path |
| Fork restates a parent's literal | Read it off the config |
| Helper mutating several config fields | Inline it in each factory |
| `x = 8; cfg.field = x` | Assign the literal directly |
| Docstring result from a dev laptop | `Results: TBD.` until reference hardware |
| Smoke config still full-size | Shrink the model too; it proves wiring, not accuracy |
| Building a test-shrink framework | Mutate the config inline |
| Loosening `torch.equal` to `allclose` | Find the real divergence |
| Bare `python -m ...` anywhere | `uv --quiet run --frozen python -m ...` |
