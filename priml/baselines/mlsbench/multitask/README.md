# MLS-Bench `cv-multitask-loss`, as priml experiments

The task trains a ResNet on CIFAR-100 with two heads: fine (100 classes) and
coarse (20 superclasses). Its editable `MultiTaskLoss` combines the two
cross-entropies and may hold parameters the optimizer also trains. Here it is
the slot `step.mtl_loss`. Ported at the `resnet56-cifar100mt` setting, pinned
to MLS-Bench commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  sum of the two losses
  +-- exp001  uncertainty weighting       (uncertainty)
  +-- exp002  dynamic weight average      (dwa)
  +-- exp003  PCGrad                      (pcgrad)
```

| File | Holds |
|---|---|
| `model.py` | `TwoHeadResNet`: the vision `ResNet` plus a second linear head |
| `data.py` | `MultiTaskCifar100Data`: the vision loader plus `label_coarse` |
| `losses.py` | `SumLoss`, `UncertaintyLoss`, `DynamicWeightAverage`, `PCGrad` |
| `train_step.py` | `MultiTaskStep`: two losses, `mtl_loss`, optimizer order |
| `metrics.py` | `HeadAccuracy`: best and final accuracy of one head |
| `experiments.py` | `exp000` to `exp003` |
| `reference.py`, `upstream.py` | Port-side and script-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

How the pristine script differs from `cv-classification-loss`:

- A second head `fc_coarse` (64 -> 20) is registered after `fc_fine`. Both
  heads are built before any initializer draws, and both take the head
  initializers.
- Superclass labels come from the fixed `CIFAR100_COARSE_MAP`.
- The loss is `MultiTaskLoss(CE(fine), CE(coarse), epoch, total_epochs)`.
  SGD runs over `model.parameters()` and then the combination's parameters.
- The reported metric is the fine head's best test accuracy. The port also
  reports the coarse head's accuracy as `test_acc_coarse`.

## PCGrad: upstream's parameter set is nondeterministic

Upstream's `pcgrad` edit finds its "shared" parameters by walking the fine
loss's autograd graph. It dedupes nodes by `id()` of short-lived Python
wrappers, and freed addresses get reused, so the walk skips a heap-dependent
subset of leaves. Measured on one graph: 22, 23, then 22 of ResNet-8's 29
leaves on three calls. On ResNet-56 it found 168 or 173 of 173, depending on
the process. The published `pcgrad` row therefore came from an unknown subset
of the parameters.

The port walks in the same order but keeps every node alive, so it always
takes the full set: 173 leaves for ResNet-56, every parameter except
`fc_coarse.*`. The capture pins each upstream `pcgrad` instance's
`_shared_params` to that set before its first call, as an instance attribute
with no file edit. It then compares bit for bit and asserts that upstream's
own walk is a subsequence of the port's.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.multitask.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/multitask/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/multitask/testdata
```

The capture imports the pristine script and each baseline's edit from the
clone. It compares with exact equality:

- each combination's value and gradients on fixed losses, PCGrad's projection
  and no-projection branches both included
- the superclass map, the 200 learning rates and the optimizer settings
- the size and order of every tensor SGD updates, per experiment
- the reported accuracy
- the two-head ResNet-56's initial state and both forwards
- the generator's position after each experiment's build
- every pixel and label of the loaders
- every parameter after two updates per combination

`reference_test.py` replays the stored values in the default tier.
