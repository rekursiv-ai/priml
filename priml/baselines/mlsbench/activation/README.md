# MLS-Bench `dl-activation-function`, as priml experiments

The task's editable `CustomActivation` is `step.model.activation`
(`ResNet.Config.activation`, pushed into every block). One activation is built
for the stem and one per block, which applies it twice, as the script does.

The task has no `resnet56-cifar100` setting, so `exp000` is its first ResNet
setting, `resnet20-cifar10`: the `cv-classification-loss` recipe at depth
`(3, 3, 3)`, 10 classes and CIFAR-10's channel statistics. Pinned to
MLS-Bench commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  ReLU
  +-- exp001  GELU
  +-- exp002  SiLU
  +-- exp003  Mish
```

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.activation.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.activation.experiments.exp000
```

## Proof against the reference

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/activation/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/activation/testdata
```

It imports the pristine script and the `gelu`, `silu` and `mish` edits from
the clone and compares with exact equality: each activation and its gradient
on fixed inputs, the 200 learning rates, the optimizer, ResNet-20's and
ResNet-56's initial state and forward, the generator after the build, every
pixel of the CIFAR-10 loaders, and every parameter after two updates per
activation. `reference_test.py` replays `testdata/reference.pt`.
