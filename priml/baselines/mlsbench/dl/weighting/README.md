# MLS-Bench `cv-sample-weighting`, as priml experiments

ResNet-32 on long-tailed CIFAR-100 (`resnet32-cifar100lt`, imbalance ratio
100). The task's editable `compute_class_weights` is the slot
`step.loss.class_weight`: a function from per-class training counts to the
weights of a weighted cross-entropy. Pinned to MLS-Bench commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  uniform weights
  +-- exp001  sqrt inverse frequency           (inverse_freq)
  +-- exp002  sqrt inverse effective number    (effective_number)
  +-- exp003  log-frequency gap, rarest 5x     (balanced_softmax)
  +-- exp004  resnet32-cifar10lt: CIFAR-10, ratio 100, uniform
  |     +-- exp005..exp007  the three baselines
  +-- exp008  vgg16bn-cifar100lt (hidden): VGG-16-BN, ratio 50, uniform
        +-- exp009..exp011  the three baselines
```

`vgg16bn_cifar100lt.sh` passes `--imbalance-ratio 50`, so `exp008` changes
the network and the ratio together.

`LongTailCifar100Data` cuts the prepared training split at load: class `c`
keeps `max(int(500 * 100^(-c/99)), 1)` images, picked by a NumPy
`RandomState(42)` exactly as upstream picks them. It hands the counts to the
loss through the loop's `bind_step` hook, before the first update; the counts
are data, never config.

Differences from `cv-classification-loss`: ResNet-32 (`(5, 5, 5)`), the
long-tailed training split, the weighted loss, and upstream logging its test
loss with the weighted criterion (here the test loss stays unweighted; test
accuracy is unaffected).

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.weighting.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/weighting/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/weighting/testdata
```

Runs the pristine script and each baseline's edit from the clone and compares
with exact equality: each weighting's weights, loss and gradient; the counts,
weights and kept positions `main` builds over CIFAR-100's label layout; the
kept positions and counts of the real training split when `--cifar100` names
torchvision's CIFAR-100 root (`/opt/scratch/datasets/cifar100` once
`prepare_data` has run); ResNet-32's initial state and forward; the
200 rates; the optimizer; every pixel of the long-tailed loaders; and every
parameter after two updates per weighting. At the other two settings it
compares each weighting's full network state, the setting's counts, weights,
kept positions, rates, optimizer, draws and forwards, the long-tailed CIFAR-10
loader, and two updates of the shallowest VGG the script's own `VGG16_CFG`
builds (patched to the vision capture's `TINY_ARCHS`, on both sides).
`reference_test.py` replays `testdata/reference.pt` and
`testdata/reference_settings.pt`.
