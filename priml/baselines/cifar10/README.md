# CIFAR-10

Image classification on 50000 32x32 photographs in 10 classes. Small enough
to train end to end in minutes on one GPU, which makes it the right place to
read a complete `priml` pipeline before reaching for a larger one.

## Reproduce

Run every command from the root of a fresh clone. `uv` installs the locked
dependencies on first use.

### Prerequisites

- Hardware: one GPU. The runtime takes CUDA when present, else the CPU.
  `exp_smoke` finishes on either.
- Credentials: none; CIFAR-10 is a public download.
- Disk: about 1 GB under `<base_dir>/datasets/cifar10`.

### Data

```bash
uv --quiet run --frozen python -m priml.baselines.cifar10.scripts.prepare_data
```

Downloads the CIFAR-10 archive through `torchvision`, which pins its URL and
MD5, and writes normalized `train.pt` and `test.pt` to
`/opt/scratch/datasets/cifar10`. Idempotent.

### Experiments

```bash
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp_smoke
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp001
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp002
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp003
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp004
```

Every experiment reads only the prepared tensors, so each runs after the Data
step and in any order. Each writes checkpoints to
`<base_dir>/runs/cifar10/<experiment>/checkpoints` and logs its metrics.

| Experiment | Reads | Run after | Result | Hardware |
|---|---|---|---|---|
| `exp_smoke` | `datasets/cifar10` | Data | Installation check only | CPU or one GPU |
| `exp000` | `datasets/cifar10` | Data | TBD | -- |
| `exp001` | `datasets/cifar10` | Data | TBD | -- |
| `exp002` | `datasets/cifar10` | Data | TBD | -- |
| `exp003` | `datasets/cifar10` | Data | TBD | -- |
| `exp004` | `datasets/cifar10` | Data | TBD | -- |

`TBD` means not yet measured on reference hardware.

### The `base_dir` knob

`base_dir` defaults to `/opt/scratch`. Data and run outputs resolve beneath it,
so one override relocates a whole run; point the Data step at the same root:

```bash
uv --quiet run --frozen python -m priml.baselines.cifar10.scripts.prepare_data --directory /data/priml/datasets/cifar10
uv --quiet run --frozen python -m priml priml.baselines.cifar10.experiments.exp000 --override base_dir=/data/priml
```

Hyperparameters are deliberately not shown here. Changing one at launch
produces a result whose config exists nowhere in the code -- it cannot be
rerun, and it cannot be compared against `exp000`. Write a fork instead.

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

Runs on CPU in a few seconds. Alongside the unit tests, `testdata/` holds
bit-for-bit snapshots of both networks' forward passes and of three optimizer
steps under each optimizer stack. They pin the arithmetic: a change that moves
a single floating-point bit fails, which is what keeps `exp000` frozen in
practice rather than only by convention.

Regenerate them only when a numeric change is intended:

```bash
uv --quiet run --frozen pytest priml/baselines/cifar10 --regenerate-b4b
```

Doing so means `exp000` has changed, and its recorded result no longer
describes the code.
