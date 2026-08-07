---
name: priml
description: ALWAYS invoke this skill when using or editing priml -- adding a model/optimizer/loss/metric, writing a baseline, designing an expNNN chain, or deciding which priml layer code belongs in. Do not write priml code -- invoke first.
---

# priml

ML building blocks for training experiments. Read the `configgle` skill for the
Config mechanics (`Fig`, `make()`, `finalize()`, `pprint`); this skill covers
what priml's Configs MEAN and where new code belongs.

## 0. Why priml is shaped this way

An experiment's entire state is one printable, swappable tree. Every rule below
follows from one of those two properties.

**Pprintable.** `cfg.pprint()` shows what this experiment changed;
`cfg.pprint(hide_default_values=False)` shows everything it is. A value that
affects the result and does not appear there makes the run unreproducible from
its config, and the config is the only artifact that outlives the process.

**Swappable.** Every node is a slot a third party can fill. Someone wanting
Lion, a different PCA backend, or their own block supplies one. They never patch
priml to get it.

So the config tree is not a description of the experiment. It IS the
experiment; rebuilding the tree rebuilds the run.

Unprintable: a module-level constant holding a tunable, an `os.environ` read, a
`finalize()` that stages data, a monkeypatch in a test. Each puts state where
the tree cannot show it.

Unswappable: a `Literal` naming implementations, a `bool` selecting an
algorithm, a helper returning `"power"`/`"eigh"`, a class constructed inside
`__init__`. Each closes a set the library had no business closing.

A `Literal` selecting an implementation fails both, which is why it is the error
to recognize first. `Literal` is right for a genuine MODE (`"reflect"` vs
`"constant"` padding): a closed set the library defines, not an open set of
implementations.

## 1. Where does this change go?

**Q1 -- knob or concept?** A scalar parameter of an existing thing, or a new
behavior with state of its own? A field meaningful only when ANOTHER field has a
particular value is a concept: `conv_momentum` is noise on every AdamW run.

**Q2 -- general or specific?** State what it does without naming your dataset,
your model's layer names, or your experiment. If you cannot, it is not library
code.

```
                 │ experiment-local           │ priml
─────────────────┼────────────────────────────┼──────────────────────────────
add an attribute │ subclass the Config. Rare: │ a real parameter of that
                 │ usually you wanted a class │ concept, at every value of
                 │                            │ the other fields
─────────────────┼────────────────────────────┼──────────────────────────────
new class        │ DEFAULT. A small Config    │ once a SECOND caller needs
                 │ injected into a slot       │ it, and the mechanism only
```

Bias toward the bottom-left. A new injected Config is cheap, local, and prints;
a new attribute on a shared Config is permanent and taxes every other user.

Start local. Promote the mechanism, never the recipe:

```python
# priml/optimizers/partition.py  # error -- general name, one model inside
def muon_and_sgd(model, *, matrix_excludes=("head",)): ...
```

The name promised generality while `("head",)` encoded one model's layer names.
The surviving split cuts THROUGH that file, into three homes:

- **mechanism** -> `optimizers/composite.py`: `Selector`, `everything`,
  `excluding`, `complement`. Routing is general.
- **the algorithm's own claim** -> `optimizers/muon.py`:
  `Muon.eligible_tensor` is `ndim >= 2`, Muon's constraint and nothing else.
- **recipe** -> the experiment: `excluding(Muon.eligible_tensor, "head")`, which
  names a layer, so it lives where layers are named.

The general/specific line runs through code, not around it.

Where a thing lives once you know it is general:

```
math/        pure functions: tensors in, tensors out. No state, Config, or I/O.
model/       nn.Module + Config, consuming math/.
optimizers/  Optimizer + Config.
loss/        configgleable losses wrapping math/loss.py.
metrics/     configgleable metrics.
data/        datasets and input pipelines.
train/       TrainLoop, TrainStep, schedules, checkpointing.
baselines/   one dataset end to end.
testing/     bit-for-bit goldens, fixtures.
```

`math/` is the floor and the rule runs both ways: a pure function found
elsewhere belongs there, and stateful code inside it is the same defect
inverted. Facades are thin -- import from the module that DEFINES a symbol
(`from priml.model.init import dirac`), never from a re-exporter.

Before writing anything, grep priml: ~40 exports in `model/`, 15 in
`optimizers/`, plus ten `math/` modules.

## 2. Writing a Config field

Most errors happen here. Could someone reasonably want a value not in your list?
Then it is an injection point, not an enumeration.

### A name is not a behavior

```python
optimizer_kind: Literal["adamw", "muon"] = "adamw"  # error
learning_rate: float = 1e-3
conv_momentum: float = 0.6  # noise on every adamw run
ns_steps: int = 3  # noise on every adamw run
```

Adding Lion means editing the train step. The print shows `"muon"`, not what
Muon does.

#### Use instead

```python
optimizer: Makeable[Callable[..., Optimizer]] = field(
    default_factory=lambda: CompositeOptimizer.Config(...),
)
"""Builds the optimizer from the model."""

muon = Muon.Config()  # in the experiment, beside the values it explains
muon.lr = 0.24
muon.ns_steps = 3
```

The per-branch hyperparameters leave the shared Config. Every run prints exactly
the optimizer it used, and a stranger's optimizer fills the same slot.

### A flag is a closed set of two

```python
dirac_init: bool = False  # error


def whiten_algorithm(device: torch.device) -> str:  # error
    return "power" if device.type == "mps" else "eigh"
```

#### Use instead

```python
init_conv: InitFn | None = None
"""Re-initializes every convolution; None keeps torch's."""

whiten_decompose: PcaDecompose = pca_eigh
"""Default reaches linalg.eigh, which MPS lacks; pass pca_power there."""
```

### A sentinel is a policy with no owner

```python
logit_scale: float = 0.0
"""Output multiplier; 0 means 1 / fan_in."""  # error
```

The rule lives in the parent's `forward` as an `if scale > 0` branch, so a
caller cannot turn it off without knowing the sentinel.

#### Use instead

```python
proj_out: Makeable[nn.Module] = field(default_factory=ScaledLinear.Config)
"""Output projection; owns its own scaling."""
```

`ScaledLinear` carries the `1 / fan_in` rule with the weights it divides. The
branch in `forward` disappears, and a plain `nn.Linear` drops in. Same shape as
`CausalLM.Config.lm_head`.

Two more of the same shape: a field meaningful only when another field has a
given value belongs on the injected piece; a count beside a list of the same
thing is redundant, because the list's length IS the count.

Declare what a slot must DO, not what it must BE -- `Makeable[Protocol]` over a
concrete class, so an implementation priml has never seen still fits.

Constants that are facts, not tunables, take `Final`:
`NUM_TRAIN_SAMPLES: Final = 50_000` is a property of the dataset. A batch size is
not; it is a tunable and belongs on the config, which `check-config-globals`
enforces. Name a constant only when the name carries meaning the number does not.

## 3. Writing a module

Every building block speaks the same shape fields, `-1` meaning "infer":

```python
channels_in: int = -1  # -1 to infer from channels_out
channels_out: int = -1  # -1 to infer from channels_in
channels_hidden: int  # or tuple[int, ...] for a multi-stage net
```

A parent pushes dimensions down in `finalize()` with `propagate_attr`
(`model/custom_types.py`), gated on a runtime-checkable Protocol:

```python
propagate_attr(self.block, "channels_in", channels, protocol=ChannelsIn)
propagate_attr(self.block, "depth", self.depth)  # protocol=None: best effort
```

The Protocol is load-bearing: a child implementing `ChannelsIn` MUST accept the
value, so a typo raises instead of silently building with the `-1` sentinel; a
child that does not implement it opts out. Never `getattr(cfg, "f", None)` --
it bypasses the checker and hides the contract.

`depth` is the block's INDEX in the stack, for depth-scaled init (see
`model/transformer.py`, `model/linear.py`), not "how deep the network is". Do
not reuse the name for a count.

A repeated sub-module is a template or a list -- the shape in
`model/causal_lm.py`, `model/sequential.py`, and `baselines/cifar10/model.py`:

```python
block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
    default_factory=TransformerBlock.Config,
)
"""Block template (broadcast num_layers times) or explicit per-layer list."""
```

One entry stays simple; a list gives per-layer control and its length is the
count. Copy the template per slot (`copy_tree`) or every block ends up carrying
the last stage's width.

A function-shaped slot takes a named type, not a bare `Callable`. Read
`math/custom_types.py` and `model/custom_types.py` before declaring one: their
docstrings carry the distinctions, including why a `TensorableFn` is not a
`TensorFn`. `InitFn`, `Schedule`, and `PcaDecompose` live beside their consumers
in `model/init.py`, `train/schedules.py`, and `math/stats.py`.

Never alias one; one spelling per concept, since it is a monorepo. A one-domain
type lives beside its consumer, not in a shared `custom_types.py` next to the
universal ones. Annotate with `from torch import Tensor`, not `torch.Tensor`.

Anything a caller might inject or subclass is public: `ResidualBlock`,
`ConvBlock`, `ScaledLinear` carry no underscore. Reserve `_` for implementation
a caller can never reach, and place it after its public caller.

`X.Config().make()` on an optimizer yields a deferred CONSTRUCTOR, not the
optimizer -- parameters do not exist at config time:

```python
optimizer: Makeable[partial[Optimizer]]  # error -- partial is invariant
optimizer: Makeable[Callable[..., Optimizer]]
```

`partial[Muon]` is not a `partial[Optimizer]`, so no concrete member fits;
`Callable[..., T]` is covariant in its return. A split recipe presents as ONE
optimizer via `CompositeOptimizer`, so the train step holds `self.optimizer`,
not a list. Selectors must be comparable objects, never closures -- a closure's
`repr` carries an address, so a config holding one never equals its parent and
experiment diffing breaks.

### Numerics

`torch.where` evaluates BOTH branches, and the dead one's gradient still flows:

```python
torch.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)  # error
```

At `x = 1.0`, an ordinary logit, the unused branch reaches `1e30`, squared in
backward to `inf`, and `inf * 0` is `NaN`. Measured: `grad=[nan, -0.053, 0.347]`.

#### Use instead

Mask each branch's ARGUMENT -- the double-where trick. Broadcast a scalar, not
`zeros_like`, to preserve dtype:

```python
negative = x < 0
torch.where(
    negative,
    -torch.log1p(-torch.where(negative, x, 0.0)),
    torch.log1p(torch.where(negative, 0.0, x)),
)
```

`x.clamp(max=0.0)` is bit-identical and shorter when the safe value IS the
boundary; not general, since `1/x` wants a safe input of `1.0` at boundary
`0.0`. Better still, delete the branch -- `where` does not short-circuit on
vector hardware, so both `log1p` calls cost:

```python
sign = torch.where(x < 0, -one, one)  # `one` carries x's dtype
sign * torch.log1p(sign * x)  # sign * x is |x|, differentiably
```

Prefer a stable identity to a stabilized expression: `log_stablemax` is
`log_softmax(log_modulus(x))`, because `log_softmax` subtracts the row max
exactly where the manual form overflowed. Any implementation that is not the
literal definition earns a `Derivation:` block.

Never auto-cast:

```python
logits = logits.to(torch.float64)  # error -- the caller cannot decline
```

That upcast protected the CALLER's reduction, not this function: ~900 bf16 terms
summed in bf16 cost `1.16e-2` against bf16's own `5.8e-5` representation error.
The accumulation is the caller's, so the decision is too --
`stablemax_cross_entropy(logits.double(), labels)`. State the requirement in
`Returns:`. Removing an implicit cast is a contract change: every call site, or
none.

Test the gradient, not just the value -- a finite-gradient assertion passes for
every wrong variant. Pin both to closed forms at the interesting points
(`log_modulus` at `[-1, 0, 1]`: values `[-log 2, 0, log 2]`, gradients
`[0.5, 0.0, 0.5]`), then confirm a plausible-but-wrong spelling goes red.
`sign(x) * log1p(|x|)` matches on every value and differs on the gradient at the
origin.

Is every word of a comment load-bearing?

```python
cfg.num_steps_eval = steps_per_epoch  # evaluate once per epoch  # error

# Floors: a short final batch is not a whole step.
steps_per_epoch = NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
```

A comment earns its place only when the reason is invisible from the code AND
its absence invites a plausible wrong edit. Then state the consequence.

## 4. Writing an experiment

Directories are named after the dataset -- `mnist`, `cifar10`, `arcagi1`.
`nanochat` and `sudoku` are named exceptions, better known than their datasets.
Layout mirrors priml so a reader navigates by analogy:

```
priml/baselines/<dataset>/
  README.md   __init__.py
  data.py     model.py    train_step.py   experiments.py   (+ *_test.py each)
  goldens/    scripts/prepare_data.py
```

Extra modules welcome when a file would sprawl. Data staging lives in
`scripts/prepare_data.py`, never in a config -- `finalize()` declares, it never
does I/O. Assume `/opt/scratch` exists.

`exp000` is the best NAIVE recipe -- what a strong first-year graduate
student writes without exotica. Not a toy, not the state of the art. Frozen at
birth: improvements are forks, never edits, so a number measured against it
stays comparable across releases. Every later experiment forks a NAMED parent
and applies ONE change:

```python
def exp002() -> Cifar10TrainLoop:
    """exp001 + Muon on the convolution weights, SGD on the rest.

    Hypothesis:
      Muon orthogonalizes each update, making the step scale-invariant with
      respect to the weight matrix, which should suit convolution kernels.

    References:
      https://kellerjordan.github.io/posts/muon/

    Results:
      TBD.
    """
    cfg = exp001()
    cfg.experiment_name = "exp002"
```

- The summary names the parent; the chain reconstructs without running anything.
- Hypothesis, References, and Results are all three required. `Results: TBD.`
  until measured on reference hardware -- never a number from a dev laptop.
- One delta. Two fields are one change only when inseparable (an optimizer and
  the schedule its published recipe prescribes); say so in the Hypothesis.
- Leave `seed` at its default (`None`) unless the experiment is about seed
  variance; pinning it hides the variance you would want to measure.

Narrow the loop's slots once, so no factory needs an `isinstance` to reach a
field it is about to set:

```python
class Cifar10TrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    step: Cifar10TrainStep.Config = field(default_factory=Cifar10TrainStep.Config)
    dataset: Cifar10Data.Config = field(default_factory=Cifar10Data.Config)
```

Keep horizon and budget equal (`cfg.max_steps = cfg.step.total_train_steps`), or
the schedule anneals past the end of training or short of it.

A smoke experiment is not a result. It answers one question -- is the data
prepared and does the loop run -- so cut every axis that costs time without
bearing on it: one epoch AND a network narrow enough to finish in seconds.
Shrinking only the step count wastes compute proving nothing.

Launch with the module path, not a config file:

```bash
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000
```

```bash
--override step.learning_rate=3e-4          # error -- a config that exists nowhere
--override dataset.working_dir=/datasets/x  # environment, not experiment
```

An overridden hyperparameter produces a result whose config is in no file, so it
cannot be rerun or compared. Write a fork. Do not show one in documentation.
`PATH=VALUE` is repeatable, dotted, JSON-parsed then cast to the field's type;
path fields are logical, resolved beneath `base_dir`.

Every Python invocation -- code, docs, your own shell -- goes through
`uv --quiet run --frozen`. Never `python -c`, never a heredoc; use Edit/Write
for file changes and a repo-local `scripts/` module for probes. Commands in
docstrings and READMEs stay on ONE line for copy-paste; the line-length rule is
exempt (`# noqa: E501` naming the reason).

## 5. Tests and goldens

CPU only; CI has no GPU. priml has no slow tests -- target well under 100 ms
each. A test running 100 training steps to watch a learning-rate schedule is
testing a pure function the hard way: sample the function.

Shrink by mutating the config. It is configgle; there is no framework to build:

```python
config.model.channels_hidden = (4, 8)  # size: yes
config.model.blocks_per_stage = 1
config.device = "cpu"

config.optimizer = plain_sgd  # error -- recipe, not size
```

Shrink SIZE only; touching the recipe (optimizer, schedule, loss, init) means
the test stops covering the experiment.

Goldens (`priml/testing/bfb.py`) make `exp000` frozen in practice rather than by
convention: one per model (`state_dict` after init, plus the forward output) and
one per optimizer stack (a few train steps, so loss, augmentation draws,
schedule, and optimizer all reach the compared post-state).

Mint and replay inside `host_agnostic_numerics()` -- it upcasts fp32 to fp64, so
AVX2 and AVX-512 agree. Assert `torch.equal`, never `allclose`. Verify a golden
BITES before trusting it: perturb a constant and confirm the failure.
Regenerating is deliberate; it means the recipe changed.

Experiment tests assert the DELTA -- exactly which fields each fork changes --
so "one change per experiment" is enforced, not claimed. They also assert
construction reads no files: a config must build on a laptop with neither the
dataset nor a GPU.

## 6. When you are corrected

- Apply the rule everywhere it holds, not where it was pointed. One `Literal`
  named means auditing every `Literal`. Grep for the shape.
- A contract change takes every consumer, all or none. A dtype or signature
  change with 26 call sites and 24 updated is worse than not starting.
- Inline the one-line helper; drop the pass-through local; do not extract a
  private factory two callers barely share. Readability outranks DRY here: the
  reader must see WHAT CHANGED without opening anything else.
- Churn is a failure, not effort. Changing a spelling twice and landing back
  where you started costs more than one wrong answer. Changing the same
  mechanism a third time means the design is wrong -- stop and say so.
- Run the gates: `ruff`, `ty`, `basedpyright`, `pytest`, then `pre-commit run
  --files ...` over everything changed.
