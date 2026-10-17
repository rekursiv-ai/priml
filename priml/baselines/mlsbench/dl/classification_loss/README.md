# MLS-Bench `cv-classification-loss`, as priml experiments

The task's editable `compute_loss` -- a function from the logits, the labels
and the epoch to a scalar -- becomes the slot `step.loss`. Ported at the
`resnet56-cifar100` setting, pinned to MLS-Bench commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. The task's pristine script is the
one [`vision/`](../vision/README.md) ports, so `exp000` is `vision`'s `exp000`
under the task's name. A loss receives what the script's `config` dict
carries, as a `LossContext`, so an epoch-dependent loss is expressible.

```
exp000  cross-entropy
  +-- exp001  label smoothing 0.1    (label_smoothing)
  +-- exp002  focal loss, gamma 2    (focal_loss)
  +-- exp003  PolyLoss, epsilon 2    (poly_loss)
```

| File | Holds |
|---|---|
| `losses.py` | `label_smoothing`, `focal`, `poly`; the pristine `cross_entropy` is `vision`'s |
| `experiments.py` | `exp000` to `exp003` |
| `reference.py`, `upstream.py` | Port-side and script-side runs of the capture |
| `scripts/capture_reference.py` | The bit-for-bit proof |

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.classification_loss.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/classification_loss/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/classification_loss/testdata
```

It imports the pristine script and each baseline's edit from the clone and
compares with exact equality: each loss's value and gradient on fixed logits,
and per loss two updates of the shallowest network -- the loss, its logits, a
probe's logits afterwards, what the loss was told, and every parameter after
each. `reference_test.py` replays the stored values in the default tier. The
network, data, schedule and optimizer are `vision`'s, proven by its capture.

Not yet ported: the `vgg16bn-cifar100` and `mobilenetv2-fmnist` settings.
