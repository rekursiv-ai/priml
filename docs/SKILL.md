---
name: priml
description: ALWAYS invoke this skill when using or editing priml -- adding a model/optimizer/loss/metric, writing a baseline, designing an expNNN chain, or deciding which priml layer code belongs in. Do not write priml code -- invoke first.
---

# priml

ML building blocks for training experiments. Read the `configgle` skill for the
Config mechanics (`Fig`, `make()`, `finalize()`, `pprint`); this skill covers
what priml's Configs MEAN and where new code belongs.

## 0. Principles

An experiment's entire state is one config tree. Four properties make that worth
doing, and every rule below follows from one of them.

1. **Hierarchical** -- a reader holds one node at a time. A slot's contents stay
   unread until opened, so complexity stays bounded however large the tree
   grows.
2. **Injectable** -- leaves own churn, not ancestors. Adding an optimizer must
   not edit the train step; the root holds a slot and no knowledge of what
   fills it.
3. **Hermetic** -- the tree is the whole input. No environment read, no global,
   no staging at build time, so the same tree gives the same run anywhere.
4. **Printable** -- the whole run is inspectable without running it. A value
   affecting the result that is not in the print makes the run unreproducible
   from its config, and the config outlives the process.

```python
exp001().pprint()  # PPrints fields differing from Config class.
exp001().pprint(hide_default_values=False)
```

Both `copy_tree().finalize()`, so you see the tree after propagation. Eg,
`working_dir` resolved beneath `base_dir`, `channels_in` pushed into each block,
sentinels filled in. That is why it is the debugging tool -- it shows what
`make()` will use, not what you typed. Pass `finalize=False` for raw input.

The default view is already a diff against the class defaults, so comparing two
experiments is the same call on each:

```python
difflib.unified_diff(
    exp000().pformat().splitlines(),
    exp001().pformat().splitlines(),
    "exp000",
    "exp001",
    lineterm="",
)
```

`exp().pprint()` is the single best way to debug an experiment!

Experiments should be entirely defined by their `Config`. NEVER USE: a
module-level constant holding a tunable, an `os.environ` read, etc.

Since `finalize()` is for config inspection, it should be lightweight. NEVER use
it for durable side-effects, staging data, monkeypatching, etc.

Future-proof `Config`s by facilitating dependency injection for likely-to-change
components. Eg, rather than `"relu"` just assign the field as `torch.relu`.
Using `Enum` or `Literal` means future changes requires upstream changes which
introduces the possibility for error.

## 1. What a tree looks like

Construct, then mutate. Every line names its full path, so the file reads as a
config and _what changed_ reads linearly, without parsing indents to place a
value.

Be sure that every docstring explicitly names "Hypothesis", "References", and
"Results" (the last filled in after the fact). Exception: it is permissible to
drop "Hypothesis" and/or "References" on grid sweeps.

The canonical example is `priml.baselines.cifar10.experiments`; excerpt:

```python
NUM_TRAIN_SAMPLES: Final = 50_000


def exp000() -> Cifar10TrainLoop:
    """Pre-activation ResNet trained with AdamW and cosine decay.

    trax experiment 42

    Hypothesis:
      A residual network with AdamW and cosine decay is the strongest recipe
      using nothing exotic -- the bar any addition must clear.

    References:
      https://arxiv.org/abs/1603.05027

    Results:
      TBD.
    """
    cfg = Cifar10TrainLoop()
    cfg.study_name = "cifar10"
    cfg.experiment_name = "exp000"

    cfg.step.model = ResNet.Config()
    cfg.step.model.channels_hidden = (64, 128, 256)
    cfg.step.model.blocks_per_stage = 2
    cfg.step.model.block = ResidualBlock.Config()

    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 1e-3
    adamw.weight_decay = 5e-2
    cfg.step.optimizer = CompositeOptimizer.Config()
    cfg.step.optimizer.optimizers = [adamw]
    cfg.step.schedule = PartialConfig(cosine)

    cfg.dataset.batch_size = 512
    cfg.dataset.working_dir = "/datasets/cifar10"

    topk = cfg.metrics["accuracy"] = TopK.Config()
    topk.k_values = [1]

    cfg.num_steps_eval = NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
    cfg.max_steps = cfg.step.total_train_steps = 30 * cfg.num_steps_eval
    return cfg


def exp001() -> Cifar10TrainLoop:
    """exp000 + Muon on the convolution weights, SGD on the rest.

    trax experiment 43

    Hypothesis:
      Muon orthogonalizes each update, making the step scale-invariant in the
      weight matrix, which should suit convolution kernels. Its recipe anneals
      polynomially, so the two travel together.

    References:
      https://kellerjordan.github.io/posts/muon/

    Results:
      TBD.
    """
    cfg = exp000()
    cfg.experiment_name = "exp001"

    sgd, muon = cfg.step.optimizer.optimizers = [
        PartialConfig(torch.optim.SGD),
        Muon.Config(),
    ]
    sgd.lr = 0.67
    muon.lr = 0.24
    on_muon = excluding(Muon.eligible_tensor, "head")
    cfg.step.optimizer.select = [complement(on_muon), on_muon]

    cfg.step.schedule = PartialConfig(polynomial)
    cfg.step.schedule.power = 1.2
    return cfg
```

`exp001` reads as a diff: call the named parent, change one thing, return. The
swap touches no other node, because `optimizer` is a slot rather than a
`Literal` the train step branches on. Nothing here is a mode string: `model`,
`block`, `optimizer`, and `schedule` hold a config or a function, so a variant
is a different VALUE.

`Cifar10TrainLoop` pre-narrows the `step` and `dataset` slots (section 6), so no
line needs `isinstance`. This NARROWS the `configgle` skill's guidance rather
than contradicting it: that skill's per-factory `assert isinstance(...)` is
correct wherever a slot's type varies between experiments. In a baseline it does
not -- every experiment in the directory shares one step and one dataset -- so
the assert belongs once, on the class.

`CompositeOptimizer` is the adapter even for one member: the slot is called with
the MODEL, so something must route parameters to optimizers -- a bare
`PartialConfig` raises `'ResNet' object is not iterable`. `max_steps` and
`total_train_steps` move together because a schedule whose horizon differs from
the budget anneals past the end of training or short of it.

Bind a local only to shorten a long path (`adamw`, `topk`), never to rename a
short one. Set only what differs from the default: `cfg.x = <default>` is noise
and pollutes the `pprint` diff section 0 relies on.

## 2. Where does this change go?

```
                 │ study-local                   │ priml-major
─────────────────┼───────────────────────────────┼─────────────────────────────
add an attribute │ Subclass the Config, add the  │ An essential arm of a core
                 │ field. RARE -- classes should │ concept and/or error prone.
                 │ be dependency injectable.     │
─────────────────┼───────────────────────────────┼─────────────────────────────
new class        │ DEFAULT. A small Config in    │ A defensible ML primitive
                 │ your study dir into an        │ absent from priml but likely
                 │ existing slot.                │ needed by others.
```

Bias toward the bottom-left. A new injected Config is local and prints; a new
attribute on a shared Config is permanent and every other user carries it.

Which column: state what the code does without naming your dataset, your model's
layer names, or your experiment. If you cannot, it is not library code. Which
row: a scalar parameter, or a behavior with state of its own? A field meaningful
only when ANOTHER field has a particular value is the second --
`conv_momentum` is noise on every AdamW run.

**Write it local, promote it later.** New code starts in the baseline that needs
it, where a wrong guess reaches one directory. Two triggers move it into priml;
nothing else does:

- **A second caller needs it.** Two baselines reaching for the same thing is
  evidence; one baseline and an intention is not. The second caller reveals
  which parts were general.
- **It is a fundamental operation** -- a definition rather than a choice. A
  careful numeric transform, an eigendecomposition, a routing protocol: right
  independent of any experiment, so waiting only means the first caller wrote
  it worse.

Promote on a guess and the code carries its origin's assumptions in a default,
where its position in `priml/` presents them as general.

What gets promoted is a CONFIGGLEABLE CLASS, never a function carrying keyword
arguments.

```python
# priml/optimizers/partition.py
# Bad -- the kwarg is invisible to pprint, --override, a fork, and a diff.
def muon_and_sgd(model, *, matrix_excludes=("head",)): ...


class MuonAndSgd:  # Good -- the same value is a node
    class Config(Fig["MuonAndSgd"]):
        matrix_excludes: list[str] = field(default_factory=list)
        """Parameter-name substrings routed to SGD instead of Muon."""
```

A function's keyword arguments are not nodes in the tree. They do not print, so
`matrix_excludes=("head",)` never appears in `pprint` and the run stops being
reproducible from its config; they cannot be reached by `--override`, mutated by
a fork, or diffed against a parent. The default also asserts that every model
names its classifier `head` -- one model's recipe under `baselines/cifar10/`,
a library-wide requirement under `priml/optimizers/`.

A `Config` fixes both: its fields are nodes, so they print and diff, and each is
a slot a caller can fill. Promotion then also cuts THROUGH the original
function: routing is general (`composite.py`'s `Selector`, `excluding`,
`complement`), `ndim >= 2` is Muon's own claim (`Muon.eligible_tensor`), and
`excluding(Muon.eligible_tensor, "head")` names a layer, so it stays in the
experiment.

**`math/` is the one deliberately anti-configgle layer.** Nothing in it defines
a `Config`, and nothing should: a function there takes tensors and returns
tensors, so there is no state to configure, nothing to print, and no slot to
inject. Keyword arguments are fine and expected (`dim=-1`, `eps=5e-4`,
`whiten=True`) because they are arguments to a computation, not hidden
experiment state -- a caller passes them explicitly at every call, or binds them
in a `PartialConfig`, which puts them back IN the tree.

That is why the layer inverts: pure functions belong there, and the configgleable
class that consumes one belongs in `loss/`, `model/`, or `optimizers/`.
`stablemax_cross_entropy` lives in `math/loss.py`; the `SimpleLoss` that wraps
it, with its `ignore_index` and reduction, lives in `loss/`. So a `Config`
inside `math/` is wrong, and so is a kwarg-carrying function anywhere else.

## 3. The layers

One axis: how much state does this hold, and who owns it.

| Dir | Holds | Why separate |
|---|---|---|
| `math/` | pure functions, `Tensor`/`Tensorable` in and always `Tensor`(s) out | No state, Config, or I/O -- testable without a model, device, or config. |
| `model/` | `nn.Module` + `Config` | Parameters are state. Calls `math/`; keeps only what must persist. |
| `optimizers/` | `Optimizer` + `Config` | Update state belongs to the algorithm, not the model it updates. |
| `loss/` | configgleable losses | Math is in `math/loss.py`; these carry configuration only. |
| `metrics/` | configgleable metrics | Accumulate across batches, so they hold state. |
| `data/` | datasets, input pipelines | The only layer touching disk or network -- so a config BUILDS without either. |
| `train/` | `TrainLoop`, `TrainStep`, schedules | Owns the run: steps, RNG, checkpoints. Everything above is a slot it fills. |
| `inference/` | generation, serving | Read-only use of a trained model. |
| `testing/` | goldens, fixtures | Test-only; production never imports it. |
| `baselines/` | one dataset end to end | The only place a dataset name, layer name, or recipe may appear. |

Two directions of one rule: a pure function outside `math/` should move down; a
stateful one inside it makes the floor unreliable for everyone above.
Dependencies point down, never up -- `math/` imports nothing from priml, and a
`model/` module reaching into `train/` is a layering error even when it
type-checks.

Import from the module that DEFINES a symbol
(`from priml.model.init import dirac`), never a re-exporter. Before writing
anything, grep priml: ~40 exports in `model/`, 15 in `optimizers/`, ten `math/`
modules.

## 4. Writing a Config field

Most errors happen here. Could someone reasonably want a value not in your list?
Then it is an injection point, not an enumeration.

### An enum makes ancestors own churn; injection leaves it with the leaf

```python
optimizer_kind: Literal["adamw", "muon"] = "adamw"  # Bad -- closed set
conv_momentum: float = 0.6  # Bad -- noise on every adamw run
ns_steps: int = 3  # Bad -- noise on every adamw run

optimizer: Makeable[Callable[..., Optimizer]] = field(  # Good -- one slot
    default_factory=CompositeOptimizer.Config,
)
```

Under the enum, every option's parameters pile onto the shared node and adding
Lion edits the train step -- an ANCESTOR -- to teach it a name; the print shows
`"muon"`, a label, not what Muon does. Under the slot the root holds no
knowledge of what fills it, Muon's parameters live on Muon where they are always
meaningful, and Lion is added without editing any node above it.

### A flag is a closed set of two

```python
dirac_init: bool = False  # Bad -- a closed set of two
init_conv: InitFn | None = None  # Good -- None keeps torch's


def whiten_algorithm(device: torch.device) -> str:  # Bad -- impl behind a string
    return "power" if device.type == "mps" else "eigh"


whiten_decompose: PcaDecompose = pca_eigh  # Good -- pass pca_power on MPS
```

### A sentinel is a policy with no owner

```python
logit_scale: float = 0.0
"""Output multiplier; 0 means 1 / fan_in."""  # Bad -- unowned policy

proj_out: Makeable[nn.Module] = field(default_factory=ScaledLinear.Config)
"""Output projection; owns its own scaling."""  # Good
```

The sentinel's rule lives in the parent's `forward` as an `if scale > 0` branch,
so a caller cannot turn it off without knowing the sentinel. `ScaledLinear`
carries the `1 / fan_in` rule with the weights it divides: the branch disappears
and a plain `nn.Linear` drops in. Same shape as `CausalLM.Config.lm_head`.

Two more of the same shape: a field meaningful only when another field has a
given value belongs on the injected piece; a count beside an EXPLICIT list is
redundant, since the list's length IS the count. (A count is right when the slot
holds one template to broadcast -- `blocks_per_stage` beside a single `block`.)
Declare what a slot must DO, not what it must BE.

Constants that are facts, not tunables, take `Final`:
`NUM_TRAIN_SAMPLES: Final = 50_000` is a property of the dataset; a batch size
is a tunable and belongs on the config, as `check-config-globals` enforces.

### Prefer mutable containers

A config is EDITED, not rebuilt, so a collection field should be mutable:

```python
optimizers: tuple[Makeable[Optimizer], ...] = ()  # Bad -- a fork rebuilds it
optimizers: list[Makeable[Optimizer]] = field(default_factory=list)  # Good
```

With a tuple, changing one member means restating all of them, and which element
the fork meant to change is buried in a rewritten literal. With a list, a fork
says
`cfg.step.optimizer.optimizers[1] = Muon.Config()` or appends, and both the diff
and the `pprint` show exactly the one thing that moved. Same for `dict` over a
frozen mapping: `cfg.metrics["accuracy"] = TopK.Config()` adds a metric without
touching the others.

`default_factory` is required for any mutable default -- a bare
`optimizers: list = []` shares one list across every instance, which
`dataclasses` rejects outright. Use the class itself where the default is a
Config (`default_factory=Topping.Config`), and a `lambda` only when the default
needs constructor arguments.

Immutable is right for a tuple-of-scalars whose length is part of the shape --
`channels_hidden: tuple[int, ...] = (64, 128, 256)` is one value describing the
stage widths, and a fork that changes the width count changes the architecture,
so restating the whole tuple states the change.

### Blessed nouns

A field is reusable only if a stranger's config can be talked about in the same
words. Use these; do not invent a synonym.

| Noun | Means |
|---|---|
| `channels_in` / `channels_out` | Feature widths at the boundary. `-1` infers from the other. |
| `channels_hidden` | Internal width -- an `int`, or a `tuple` per stage. |
| `channels_head` / `heads` | Per-head width and head count. |
| `depth` | The block's INDEX in the stack, for depth-scaled init. Never a count. |
| `num_layers` / `blocks_per_stage` | Counts. Omit when a block LIST already implies one. |
| `block` | The injected sub-module: one template, or a list. |
| `proj_out` / `lm_head` | Output projection slot; owns its own scaling and init. |
| `norm` / `norm_qk` / `norm_out` | Normalization slots, named for their position. |
| `init_weight` / `init_bias` | `InitFn` slots. |
| `activation` | `ActivationFn`: a config, a Module, or a plain function. |
| `eps` / `bias` / `dropout` / `stride` / `kernel_size` | Torch's own spellings; do not rename. |
| `device` / `dtype` / `dtype_autocast` | Placement and precision. |
| `base_dir` / `working_dir` | Resource root and the logical path resolved beneath it. |
| `study_name` / `experiment_name` | Run identity; they build `working_dir`. |
| `max_steps` / `total_train_steps` / `num_steps_eval` | Budget, schedule horizon, eval period. |
| `seed` | `None` unless the experiment is about seed variance. |
| `shard` | Tensor-parallel style for a block. |

Batches are keyed `media` and `label`; a train step returns `loss` and `model`
(`TrainStepOutput`). A name ending `_fn` is a callable slot; `_dir` is a path;
`num_*` is a count; `*_kind` or `*_algorithm` is the enum smell above.

## 5. Writing a module

A parent pushes the shape nouns down with `propagate_attr`
(`model/custom_types.py`), gated on a runtime-checkable Protocol:

```python
propagate_attr(self.block, "channels_in", channels, protocol=ChannelsIn)
propagate_attr(self.block, "depth", self.depth)  # protocol=None: best effort
```

The Protocol is load-bearing: a child implementing `ChannelsIn` MUST accept the
value, so a typo raises instead of building with the `-1` sentinel and no
diagnostic, while a child that does not implement it opts out. Never
`getattr(cfg, "f", None)` -- it bypasses the checker and hides the contract.

A repeated sub-module is a template or a list (`model/causal_lm.py`,
`model/sequential.py`, `baselines/cifar10/model.py`):

```python
block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
    default_factory=TransformerBlock.Config,
)
"""Block template (broadcast num_layers times) or explicit per-layer list."""
```

One entry stays simple; a list gives per-layer control and its length is the
count. Copy the template per slot (`copy_tree`) or every block carries the last
stage's width.

A function-shaped slot takes a named type, not a bare `Callable`: read
`math/custom_types.py` and `model/custom_types.py` first, since their docstrings
carry the distinctions (including why a `TensorableFn` is not a `TensorFn`).
Never alias one; annotate with `from torch import Tensor`.

Anything a caller might inject or subclass is public -- `ResidualBlock`,
`ConvBlock`, `ScaledLinear` carry no underscore. Reserve `_` for implementation
a caller can never reach, placed after its public caller.

An optimizer's `Config.make()` yields a deferred CONSTRUCTOR, since parameters
do not exist at config time. Annotate that slot
`Makeable[Callable[..., Optimizer]]`, never `Makeable[partial[Optimizer]]`:
`partial` is invariant, so `partial[Muon]` is not a `partial[Optimizer]`.
Selectors must be comparable objects, never closures -- a closure's `repr`
carries an address, so a config holding one never equals its parent.

### Numerics: work in the log domain

`log_stablemax` normalizes the surrogate `s(x) = 1/(1-x)` if `x < 0` else
`1 + x`. Four passes, each fixing what the last got wrong:

```python
# Bad -- NaN gradient
s_x = torch.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)
torch.log(s_x / s_x.sum(dim, keepdim=True))

# Good -- the DOUBLE-WHERE trick: an outer `where` picks the branch, an inner
# one sanitizes each branch's ARGUMENT so neither is evaluated off its domain.
negative = x < 0
torch.where(
    negative,
    -torch.log1p(-torch.where(negative, x, 0.0)),
    torch.log1p(torch.where(negative, 0.0, x)),
)

# Better -- no `where` at all
sign = torch.where(x < 0, -one, one)
sign * torch.log1p(sign * x)

# Best -- it has a name, and the whole function collapses
torch.log_softmax(log_modulus(x), dim=dim)
```

The first form NaNs because `where` evaluates BOTH branches and the dead one's
gradient still flows: at `x=1.0` the unused branch reaches `1e30`, squared in
backward to `inf`, and `inf * 0` is `NaN` (measured
`grad=[nan, -0.053, 0.347]`). The double-where is the general fix -- mask the
ARGUMENT, not the result -- and broadcasting a scalar rather than `zeros_like`
preserves dtype.

Everything after that came from the log domain, which is the transferable
lesson:

1. **Sub-structure becomes recognizable.** `log(s(x))` is `sign(x) log1p(|x|)`,
   the log-modulus transform of John & Draper (1980). In the linear domain it
   was a nameless two-branch expression.
2. **Fusion opportunities appear.** Once each term is a log, normalizing is
   `logsumexp` -- so the whole thing is `log_softmax`, one fused primitive
   replacing a divide and a sum.
3. **Numerics improve for free.** `log1p` is accurate near zero where `log(1+z)`
   cancels, and `log_softmax` subtracts the row max exactly where the
   `s`-then-divide form overflowed.
4. **The branch disappears.** In logs the halves differ only by sign, so
   hoisting it removes the `where`. `where` does not short-circuit on vector
   hardware, so both branches execute.

Try the log domain before optimizing a branch. And when an expression survives
three rounds of tuning, search for its published name: that brings a citation, a
stated domain, and the edge cases its authors already found. Put it in
`math/numeric.py` beside its relatives. Any implementation that is not the
literal definition carries a `Derivation:` block.

Never auto-cast:

```python
logits = logits.to(torch.float64)  # Bad -- the caller cannot decline
```

That upcast protected the CALLER's reduction: ~900 bf16 terms summed in bf16
cost `1.16e-2` against bf16's own `5.8e-5` representation error. The
accumulation is the caller's, so the decision is too -- state it in `Returns:`.
Removing an implicit cast is a contract change: every call site, or none.

Test the gradient, not just the value: a finite-gradient assertion passes for
every wrong variant. Pin both to closed forms, then confirm a wrong spelling
goes red.

Write a comment only when the reason is invisible from the code AND its absence
invites a plausible wrong edit. Then state the consequence.

```python
cfg.num_steps_eval = steps_per_epoch  # evaluate once per epoch  # Bad
steps_per_epoch = NUM_TRAIN_SAMPLES // cfg.dataset.batch_size
# Good -- Floors: a short final batch is not a whole step.
```

## 6. Writing an experiment

Directories are named after the dataset -- `mnist`, `cifar10`, `arcagi1`;
`nanochat` and `sudoku` are exceptions, better known than their datasets. Layout
mirrors priml: `data.py`, `model.py`, `train_step.py`, `experiments.py`, each
with a `*_test.py`, plus `goldens/` and `scripts/prepare_data.py`. Data staging
lives in that script, never in a config -- `finalize()` declares, it never does
I/O. Assume `/opt/scratch` exists.

`exp000` is the best NAIVE recipe -- what a strong first-year graduate student
writes without exotica. Not a toy, not the state of the art. Frozen at birth:
improvements are forks, never edits, so a number measured against it stays
comparable across releases. Every later experiment forks a NAMED parent and
applies ONE change, as `exp001` does in section 1:

- Names are `expNNN`, sequential and never adorned: `exp007`, not `exp007_muon`,
  `exp007b`, or `exp_muon_v2`. The number is an identity, not a description -- a
  slug goes stale when the experiment is revised, collides when two people pick
  the same word, and breaks sorting. `exp_smoke` is the one exception.
- The summary names the parent; the chain reconstructs without running anything.
- Hypothesis, References, and Results are all three required. `Results: TBD.`
  until measured on reference hardware -- never a number from a dev laptop.
- One delta. Two fields are one change only when inseparable (an optimizer and
  the schedule its recipe prescribes); say so in the Hypothesis.
- Leave `seed` at its default (`None`) unless studying seed variance.

Narrow the loop's slots once, so no factory needs an `isinstance` to reach a
field it is about to set:

```python
class Cifar10TrainLoop(Makes["TrainLoop"], TrainLoop.Config):
    step: Cifar10TrainStep.Config = field(default_factory=Cifar10TrainStep.Config)
    dataset: Cifar10Data.Config = field(default_factory=Cifar10Data.Config)
```

`exp_smoke` answers one question -- is the data prepared and does the loop run
-- so cut every axis that adds time without bearing on it: one epoch AND a
network narrow enough to finish in seconds. Shrinking only the step count still
builds the full-width model.

```bash
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000
--override step.learning_rate=3e-4          # Bad -- a config that exists nowhere
--override dataset.working_dir=/datasets/x  # Rare -- environment (as needed); uncommon.
```

The launcher applies either: `PATH=VALUE` is repeatable, dotted, JSON-parsed,
cast to the field's type, and reaches any node. Use that reach for DIRECTORIES
-- where data lives is a property of the machine, so one config stays runnable
everywhere. Overriding a hyperparameter instead produces a result whose config
exists in no file: it cannot be rerun, diffed, or identified afterward. Write a
fork, and never show an override in documentation; examples get copied.

Every Python invocation -- code, docs, your own shell -- goes through
`uv --quiet run --frozen`; never `python -c` or a heredoc. Commands in
docstrings stay on ONE line for copy-paste, with `# noqa: E501` naming why.

## 7. Tests and goldens

CPU only; CI has no GPU. **No test exceeds 100 ms**, and that is always
reachable: depth, width, batch, and step count are the only things making a test
slow, and shrinking all four preserves coverage. A 2-layer, 8-channel net run
for 3 steps exercises the same paths as the real one -- wiring, shapes,
optimizer arithmetic -- because the recipe is unchanged. A test needing 100 steps
to watch a schedule is testing a pure function the hard way: sample it.

Shrink by mutating the config. It is configgle; there is no framework to build:

```python
config.model.channels_hidden = (4, 8)  # Good -- size
config.model.blocks_per_stage = 1
config.device = "cpu"

config.optimizer = plain_sgd  # Bad -- recipe, not size
```

Touching the recipe (optimizer, schedule, loss, init) means the test no longer
covers the experiment. Changing size never does that.

Goldens (`priml/testing/bfb.py`) make `exp000` frozen in practice rather than by
convention: one per model (`state_dict` after init, plus the forward output) and
one per optimizer stack (a few train steps, so loss, augmentation draws,
schedule, and optimizer all reach the compared post-state). Mint and replay
inside `host_agnostic_numerics()` -- it upcasts fp32 to fp64, so AVX2 and
AVX-512 agree. Assert `torch.equal`, never `allclose`. Verify a golden BITES
before trusting it: perturb a constant and confirm the failure. Regenerating one
means the recipe changed.

Experiment tests assert the DELTA -- exactly which fields each fork changes --
which is what enforces "one change per experiment". That construction reads no
files: a config must build with neither the dataset nor a GPU.

Not every experiment needs one. A fork is a composition: if the parent and the
injected piece are both tested, the fork adds no untested code, only an untested
COMBINATION. Judgement call -- write a test when the fork wires two things
together in a way neither covers (a new selector routing parameters, a schedule
interacting with the budget); skip it when the fork swaps a value into a slot
both sides already exercise. `exp000` carries goldens regardless, being the
control.

## 8. When you are corrected

- Apply the rule everywhere it holds, not where it was pointed. One `Literal`
  named means auditing every `Literal`. Grep for the shape.
- A contract change takes every consumer, all or none. A dtype or signature
  change with 26 call sites and 24 updated is worse than not starting.
- Inline the one-line helper; drop the pass-through local. Readability outranks
  DRY: the reader must see WHAT CHANGED without opening anything else.
- Run gates: `uv --quiet run --frozen --group pre-commit pre-commit run --files ...`
