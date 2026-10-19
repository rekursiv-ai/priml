# MLS-Bench `security-machine-unlearning`, as priml experiments

The harness pretrains a classifier on every class, unlearns one class with an
editable `UnlearningMethod`, and scores retained accuracy, forget accuracy and
a membership-inference AUC. The method becomes the slot `method` of
`MachineUnlearning.Config`; everything else is fixed by the harness and is a
field only so it prints. Pinned to MLS-Bench commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  retain-only finetuning, ResNet-20 on CIFAR-10   (pristine)
  +-- exp001  retain-only finetuning                    (retain_finetune)
  +-- exp002  negative gradient                         (negative_gradient)
  +-- exp003  bad teacher                               (bad_teacher)
  +-- exp004  SCRUB                                     (scrub)
  +-- exp005  VGG-16-BN on CIFAR-100                    (vgg16bn-cifar100-class0)
  +-- exp006  MobileNetV2 on Fashion-MNIST              (mobilenetv2-fmnist-class0)
```

The pristine method only adds an unused `forget_weight = 0.0`. It computes
the same thing as `retain_finetune`, so `exp001` changes only its name. The
capture proves both scripts against the one port method.

| File | Holds |
|---|---|
| `unlearning.py` | `MachineUnlearning`: seed, pretrain, unlearn, evaluate |
| `methods.py` | `RetainFinetune`, `NegativeGradient`, `BadTeacher`, `Scrub` |
| `data.py` | `UnlearningData`: the forget split and loaders with the harness's draws |
| `metrics.py` | `evaluate_accuracy`, `auc_from_scores`, `Report` |
| `experiments.py` | `exp000` to `exp006` |
| `reference.py`, `upstream.py` | Port-side and harness-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

How the harness differs from the DL scaffold's script:

- It pretrains with the vision recipe (SGD, per-epoch cosine, cross-entropy)
  and then runs a second phase: Adam at `1e-3`, one retained and one forget
  batch per step. Each loader cycles on its own. An epoch has as many steps
  as the shorter loader has batches.
- It seeds torch with the seed itself. `TrainLoop` would salt it, so this is
  a job, not a `TrainLoop`.
- The attack scores the forget class's training images through the
  augmenting training pipeline, as the harness does.
- Its MobileNetV2 uses ReLU, not ReLU6; `exp006` sets the activation slot.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data --dataset cifar10
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.machine_unlearning.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/tl/machine_unlearning/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/tl/machine_unlearning/testdata
```

The capture imports the harness, its pristine method and each baseline's
edit from the clone. It compares with exact equality:

- the launch script's flags, parsed by the harness, against `exp000`
- the forget split of fixed labels, held as a list and as a tensor
- three updates of every method on fixed batches: the network after each
  and every teacher
- accuracy, confidences and AUC on fixed inputs, ties included
- each setting's full network: its initial state and the generator
  after the build; the shallowest network's forwards
- per variant, the harness's whole `main` on a synthetic dataset with its
  four-worker loaders: the initial and pretrained network, each step's
  batches, loss, position, Adam settings and network, the evaluation, and
  the generator afterwards

It also counts main-process draws on both sides and checks that Python's and
NumPy's generators are never drawn from. `reference_test.py` replays the
stored values. The end-to-end runs at the VGG and MobileNetV2 settings use
the scaffold's networks, which the capture compares at full size.
