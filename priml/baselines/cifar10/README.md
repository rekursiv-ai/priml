# CIFAR-10

Image classification on 50000 32x32 photographs in 10 classes. Small enough
to train end to end in minutes on one GPU, which makes it the right place to
read a complete `priml` pipeline before reaching for a larger one.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.cifar10.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000
```

The first command downloads the dataset and caches it as normalized tensors
under `/opt/scratch/datasets/cifar10`; it is idempotent, so re-running costs
nothing. The second trains.

`--override PATH=VALUE` adapts a run to the machine it lands on -- where the
data lives, where output goes:

```bash
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000 --override dataset.working_dir=/datasets/my-cifar10
```

Path fields are *logical*: `dataset.working_dir` resolves beneath the run's
`base_dir` (`/opt/scratch` by default), so pass `/datasets/my-cifar10`, not the
full on-disk path. Override `base_dir` to move the whole tree at once.

Hyperparameters are deliberately not shown here. Changing one at launch
produces a result whose config exists nowhere in the code -- it cannot be
rerun, and it cannot be compared against `exp000`. Write a fork instead.

To check an installation before committing to a full run, `exp_smoke` is
`exp000` cut to a single epoch.

## Experiments

`exp000` is the baseline. It is the strongest recipe that uses nothing
exotic -- the network a competent practitioner writes first -- and it is
frozen: improvements land as forks, never as edits, so a number measured
against it stays comparable.

Each later experiment forks a named parent and applies one change. Its
docstring carries a **Hypothesis** (what the change should buy and why),
**References** (where it comes from), and **Results** -- `TBD` until the
variant has been measured on reference hardware.

| Experiment | Parent | Change |
|---|---|---|
| `exp000` | -- | Pre-activation ResNet, AdamW, cosine decay, 30 epochs |
| `exp001` | `exp000` | SpeedNet architecture: PCA whitening, wide and shallow, 8 epochs |
| `exp002` | `exp001` | Muon on the convolution weights |
| `exp003` | `exp002` | Test-time augmentation over mirrored, shifted crops |
| `exp004` | `exp003` | Identity (Dirac) initialization of the convolutions |

## Layout

| File | Contents |
|---|---|
| `data.py` | Download, cache, and serve batches from device memory |
| `model.py` | `ResNet` (exp000) and `SpeedNet` (exp001 on) |
| `train_step.py` | Optimizers, schedule, augmentation, evaluation |
| `experiments.py` | The configs above |
| `scripts/prepare_data.py` | One-time dataset preparation |

Augmentation lives in the train step rather than the input pipeline because it
is an experimental variable: `exp003` changes evaluation-time augmentation by
setting one field, with no second copy of the data.

The optimizer and the learning-rate schedule are *injected*, not chosen from a
fixed set -- `exp002` supplies a `PartialConfig` naming the callable and its
arguments. Trying Lion or Adafactor is therefore a new experiment, not a patch
to `train_step.py`:

```python
cfg.step.optimizer = PartialConfig(
    single_group,
    build=PartialConfig(my_optimizer, lr=1e-3),
)
```

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/cifar10
```

Runs on CPU in a few seconds. Alongside the unit tests, `goldens/` holds
bit-for-bit snapshots of both networks' forward passes and of three optimizer
steps under each optimizer stack. They pin the arithmetic: a change that moves
a single floating-point bit fails, which is what keeps `exp000` frozen in
practice rather than only by convention.

Regenerate them only when a numeric change is intended:

```bash
BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/baselines/cifar10
```

Doing so means `exp000` has changed, and its recorded result no longer
describes the code.
