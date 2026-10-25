# MLS-Bench `cv-pooling-aggregation`, as priml experiments

The task's editable `CustomPool` -- the global pool between the last stage and
the classifier -- becomes the slot `step.model.pool`. Ported at the
`resnet56-cifar100` setting, pinned to MLS-Bench commit
`b873944969c53b01e4ee6e843c77767b8c702485`. The pristine script is the
`cv-classification-loss` script with the pool made a module, so `exp000` is
[`vision`](../vision/README.md)'s `exp000` under a new name.

```
exp000  global average pool
  +-- exp001  global max pool               (global_max)
  +-- exp002  generalized mean, p from 3    (gem)
  +-- exp003  mean of average and max       (avg_max)
  +-- exp004  vgg16bn-cifar100 setting
  |     +-- exp005 / exp006 / exp007  global max / gem / avg_max
  +-- exp008  mobilenetv2-fmnist setting (hidden)
        +-- exp009 / exp010 / exp011  global max / gem / avg_max
```

| File | Holds |
|---|---|
| `pools.py` | `GlobalMaxPool`, `GeneralizedMeanPool`, `AvgMaxPool` |
| `experiments.py` | `exp000` to `exp011` |
| `reference.py`, `upstream.py` | Port-side and script-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.pooling.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/pooling/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/pooling/testdata
```

It imports the pristine script and each baseline's edit from the clone and
compares with exact equality: each pool's value and gradients on fixed
features (GeM's exponent included), the 200 learning rates, the optimizer, the
reported accuracy, ResNet-56's initial state and forward, the generator's
position after building the network with each pool, every pixel of the
loaders, and every parameter after two updates per pool. At the other two
settings it compares each pool's fork against the script at that setting: the
full network's initial state and draws, the shallowest network's forwards
through the pool,
and the Fashion-MNIST loaders' every pixel. `reference_test.py` replays the
stored values in the default tier.
