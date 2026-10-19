# MLS-Bench DL scaffold

The data, model, training step and recipe the
[MLS-Bench](https://github.com/Imbernoulli/MLS-Bench) DL tasks share, pinned to
MLS-Bench commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. They port the
`pytorch-vision` training script as `cv-classification-loss` ships it,
unedited, at its `resnet56-cifar100` setting. Every DL task's script is a
variant of that script, so each task package in `mlsbench/dl/` forks `exp000`
here under its own study name and turns the task's editable function into a
**slot**: a new idea is an `expNNN` fork that changes one value, not an edit of
a copy of the training script. No task lives here;
[`classification_loss/`](../classification_loss/README.md) is the task whose
script this is.

Upstream has no license, so no upstream source is copied. The port reproduces
its arithmetic and its random draws; `scripts/capture_reference.py` proves it.

## Prepare the data

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
```

Every task reads it; each task's README launches its `exp000`. A run reports
`test_acc_best` and `test_acc_final` each epoch. The reference publishes the
best accuracy over all epochs, which selects the epoch with the test labels;
`test_acc_final` is the number without that selection.

`exp000` seeds the loop with the reference's 42. `TrainLoop` salts its seed, so
the stream is not the one `torch.manual_seed(42)` gives the reference, and
`deterministic` is off, so GPU kernels may still vary between runs. Parity is
established at the component level below.

## Layout

| File | Holds |
|---|---|
| `data.py` | Uint8 CIFAR and Fashion-MNIST, shuffle and augmentation with the reference loader's draws |
| `model.py` | `ResNet`, its `block`, `norm`, `pool`, `activation` and init slots, and `BasicBlock` |
| `vgg.py`, `mobilenet.py` | `VGG16BN` and `MobileNetV2`, with the same `norm`, `pool`, `activation` and init slots |
| `settings.py` | `vgg16bn_cifar100`, `resnet110_cifar100`, `resnet20_cifar10` and `mobilenetv2_fmnist`: switch a recipe's network and dataset |
| `train_step.py` | The step, its `loss` and `schedule` slots, and the pristine `cross_entropy` |
| `metrics.py` | Best and final test accuracy |
| `experiments.py` | `exp000`, the recipe every task forks |
| `reference.py`, `upstream.py`, `draws.py` | Shared inputs, runs of the reference, draw counting |
| `scripts/` | `prepare_data.py`, `capture_reference.py` |

## Proof against the reference

```bash
git clone https://github.com/Imbernoulli/MLS-Bench.git /opt/scratch/mlsbench
git -C /opt/scratch/mlsbench checkout 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c
uv --quiet run --frozen python priml/baselines/mlsbench/dl/vision/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/vision/testdata
```

It imports the pristine script from the clone, runs it and the port under the
same seed inside `host_agnostic_numerics()`, and compares with exact equality:
the initial state and forward of the full ResNet-56, what `main` tells the loss
in each epoch, the learning rate of all 200 epochs, the optimizer, the reported
accuracy, both image pipelines, and every pixel the loaders produce over two
passes. It counts the random draws on both sides and fails if any call site is
unreached. Nothing runs on a GPU. The values that fit in a golden (at most
32,768 bytes) go to `testdata/` and `reference_test.py` replays them in the
default tier. Each task's own capture proves its slot and every parameter after
its updates.

The capture also proves the script's `vgg16bn-cifar100` and
`mobilenetv2-fmnist` settings: each full network's initial state in
registration order and where it leaves the generator, the shallowest
network's forwards, every
pixel of each setting's loaders, both Fashion-MNIST pipelines, and the
Fashion-MNIST path's random call sites.

## Porting the other settings and tasks

A task's fork for another setting is one call, then its own slot:

```python
cfg = settings.mobilenetv2_fmnist(exp000())
assert isinstance(cfg.step.model, MobileNetV2.Config)
cfg.step.model.norm = GroupNorm
```

Prepare Fashion-MNIST with `prepare_data.py --dataset fmnist`. It is cached
resized to 32x32 with one channel; normalization repeats it to three.

The eleven DL tasks are ported in the sibling packages of `mlsbench/dl/`; the
`mlsbench/` README lists the slot each one became.

Five more tasks use the `pytorch-vision` package but have scripts that differ
and need their own recipe check before anything here is reused:
`optimization-gradient-compression` (warmup cosine and its own train loop),
`security-backdoor-defense`, `security-machine-unlearning`,
`security-membership-inference-defense` and `security-poison-robust-learning`.
