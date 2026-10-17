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

## Packages

| Package | Task | Slot |
|---|---|---|
| [`pytorch_vision/`](pytorch_vision/README.md) | `cv-classification-loss` | `step.loss` |
| [`lr_schedule/`](lr_schedule/README.md) | `dl-lr-schedule` | `step.schedule` |
| [`pooling/`](pooling/README.md) | `cv-pooling-aggregation` | `step.model.pool` |
| [`normalization/`](normalization/README.md) | `dl-normalization` | `step.model.norm` |
| [`initialization/`](initialization/README.md) | `dl-weight-initialization` | `step.model.init_conv`, `init_head`, `init` |
| [`activation/`](activation/README.md) | `dl-activation-function` | `step.model.activation` |
| [`regularization/`](regularization/README.md) | `dl-regularization` | `step.regularization` |
| [`multitask/`](multitask/README.md) | `cv-multitask-loss` | `step.mtl_loss` |
| [`weighting/`](weighting/README.md) | `cv-sample-weighting` | `step.loss.class_weight` |
| [`residual/`](residual/README.md) | `dl-residual-connection` | `step.model.block` |
| [`augmentation/`](augmentation/README.md) | `cv-data-augmentation` | `dataset.augment` |

`pytorch_vision/` also holds the data, model and step the other vision tasks
reuse; its README lists the tasks still to port and the slot each becomes.

## Proof against upstream

```bash
git clone https://github.com/Imbernoulli/MLS-Bench.git /opt/scratch/mlsbench
git -C /opt/scratch/mlsbench checkout 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c
```

Each package's `scripts/capture_reference.py --root /opt/scratch/mlsbench`
reports every mismatch and writes the goldens its `reference_test.py` replays.
