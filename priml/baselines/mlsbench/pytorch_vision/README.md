# MLS-Bench vision tasks, as priml experiments

The template for porting an [MLS-Bench](https://github.com/Imbernoulli/MLS-Bench)
task to priml. The editable function of the task becomes a **slot** of a
config, so a new idea is an `expNNN` fork that changes one value, not an edit
of a copy of the training script. This package ports `cv-classification-loss`
at its `resnet56-cifar100` setting, pinned to MLS-Bench commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  cross-entropy
  +-- exp001  label smoothing 0.1
  +-- exp002  focal loss, gamma 2
  +-- exp003  PolyLoss, epsilon 2
```

Upstream has no license, so no upstream source is copied. The port reproduces
its arithmetic and its random draws; `scripts/capture_reference.py` proves it.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.pytorch_vision.experiments.exp000
```

The run reports `test_acc_best` and `test_acc_final` each epoch. The reference
publishes the best accuracy over all epochs, which selects the epoch with the
test labels; `test_acc_final` is the number without that selection.

`exp000` seeds the loop with the reference's 42. `TrainLoop` salts its seed, so
the stream is not the one `torch.manual_seed(42)` gives the reference, and
`deterministic` is off, so GPU kernels may still vary between runs. Parity is
established at the component level below.

## Layout

| File | Holds |
|---|---|
| `data.py` | Uint8 CIFAR-100, shuffle and augmentation with the reference loader's draws |
| `model.py` | `ResNet`, its `block`, `norm`, `pool`, `activation` and init slots, and `BasicBlock` |
| `train_step.py` | The step, the `loss` and `schedule` slots, the four losses |
| `metrics.py` | Best and final test accuracy |
| `experiments.py` | `exp000` to `exp003` |
| `reference.py`, `upstream.py`, `draws.py` | Shared inputs, runs of the reference, draw counting |
| `scripts/` | `prepare_data.py`, `capture_reference.py` |

## Proof against the reference

```bash
git clone https://github.com/Imbernoulli/MLS-Bench.git /opt/scratch/mlsbench
git -C /opt/scratch/mlsbench checkout 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c
uv --quiet run --frozen python priml/baselines/mlsbench/pytorch_vision/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/pytorch_vision/testdata
```

It imports the pristine script from the clone, runs it and the port under the
same seed inside `host_agnostic_numerics()`, and compares with exact equality:
the initial state and forward of the full ResNet-56, the four losses and their
gradients, the learning rate of all 200 epochs, the optimizer, both image
pipelines, every pixel the loaders produce over two passes, and every
parameter after each update of a short run per loss. It counts the random
draws on both sides and fails if any call site is unreached. Nothing runs on a
GPU. The values that fit in a golden (at most 32,768 bytes) go to `testdata/`
and `reference_test.py` replays them in the default tier.

## Porting the other settings and tasks

Not yet ported, same task: `vgg16bn-cifar100`, `mobilenetv2-fmnist` (new
`model.py` classes behind the same `model` slot; the data path is shared, with
Fashion-MNIST repeating its channel).

The ten sibling tasks are ported in the sibling packages of `mlsbench/`; its
README lists the slot each one became.

Five more tasks use the `pytorch-vision` package but have scripts that differ
and need their own recipe check before anything here is reused:
`optimization-gradient-compression` (warmup cosine and its own train loop),
`security-backdoor-defense`, `security-machine-unlearning`,
`security-membership-inference-defense` and `security-poison-robust-learning`.
