# MLS-Bench `security-poison-robust-learning`, as priml experiments

The task's editable `RobustLoss.compute_loss` -- the training loss -- becomes
the slot `step.loss`. A seeded fraction of the training labels is moved to the
next class; after 100 epochs the network is scored on the clean test split and
on how often it predicts the flipped labels of the flipped images, through the
training pipeline. Pinned to MLS-Bench commit
`b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  resnet20-cifar10-labelflip, cross-entropy
  +-- exp001  cross_entropy            (the same cross-entropy)
  +-- exp002  generalized CE, q 0.7    (generalized_ce)
  +-- exp003  symmetric CE, beta 0.5   (symmetric_ce)
  +-- exp004  soft bootstrap, beta 0.8 (bootstrap)
  +-- exp005  vgg16bn-cifar100-labelflip      (exp006-exp009: the four)
  +-- exp010  mobilenetv2-fmnist-labelflip    (exp011-exp014: the four)
```

| File | Holds |
|---|---|
| `data.py` | `PoisonedData`: the label flip, and evaluation over the test and poisoned splits |
| `losses.py` | `generalized_ce`, `symmetric_ce`, `bootstrap` |
| `metrics.py` | `PoisonRobustness`: test accuracy, poison fit, robust score |
| `experiments.py` | `exp000` to `exp014` |
| `reference.py`, `upstream.py` | Port-side and driver-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data --dataset cifar10
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.poison_robust_learning.experiments.exp000
```

## Proof against upstream

```bash
/opt/scratch/scripts/capture.sh priml/baselines/mlsbench/tl/poison_robust_learning
```

It imports the driver with its pristine loss and with each baseline's edit and
compares with exact equality: every loss and its gradient on fixed and on
saturated logits; the label flip, its indices, mask and items on a fixed split
and at full size for each setting; both evaluations on fixed logits; the 100
rates and the optimizer; per setting the full network's initial state,
generator position and forward; and per loss a whole run of the driver's
`main` on the shallowest ResNet with its four loader workers: every loss,
logit, label, epoch and rate, every parameter after each update, every pixel
and logit of both evaluations, the published metrics, and Python's and NumPy's
generators left unused. `reference_test.py` replays the stored values.
