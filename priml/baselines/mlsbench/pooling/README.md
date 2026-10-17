# MLS-Bench `cv-pooling-aggregation`, as priml experiments

The task's editable `CustomPool` -- the global pool between the last stage and
the classifier -- becomes the slot `step.model.pool`. Ported at the
`resnet56-cifar100` setting, pinned to MLS-Bench commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. The pristine script is the
`cv-classification-loss` script with the pool made a module, so `exp000` is
[`pytorch_vision`](../pytorch_vision/README.md)'s `exp000` under a new name.

```
exp000  global average pool
  +-- exp001  global max pool               (global_max)
  +-- exp002  generalized mean, p from 3    (gem)
  +-- exp003  mean of average and max       (avg_max)
```

| File | Holds |
|---|---|
| `pools.py` | `GlobalMaxPool`, `GeneralizedMeanPool`, `AvgMaxPool` |
| `experiments.py` | `exp000` to `exp003` |
| `reference.py`, `upstream.py` | Port-side and script-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.pooling.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/pooling/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/pooling/testdata
```

It imports the pristine script and each baseline's edit from the clone and
compares with exact equality: each pool's value and gradients on fixed
features (GeM's exponent included), the 200 learning rates, the optimizer, the
reported accuracy, ResNet-56's initial state and forward, the generator's
position after building the network with each pool, every pixel of the
loaders, and every parameter after two updates per pool. `reference_test.py`
replays the stored values in the default tier.
