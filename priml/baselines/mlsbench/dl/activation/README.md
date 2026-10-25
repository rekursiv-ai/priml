# MLS-Bench `dl-activation-function`, as priml experiments

The task's editable `CustomActivation` is `step.model.activation`
(`ResNet.Config.activation`, pushed into every block). One activation is built
for the stem and one per block, which applies it twice, as the script does.

The task has no `resnet56-cifar100` setting, so `exp000` is its first ResNet
setting, `resnet20-cifar10`: the `cv-classification-loss` recipe at depth
`(3, 3, 3)`, 10 classes and CIFAR-10's channel statistics. Pinned to
MLS-Bench commit `b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  ReLU
  +-- exp001  GELU
  +-- exp002  SiLU
  +-- exp003  Mish
  +-- exp004  VGG-16-BN on CIFAR-100, ReLU
  |     +-- exp005..exp007  GELU, SiLU, Mish
  +-- exp008  MobileNetV2 on Fashion-MNIST, ReLU
        +-- exp009..exp011  GELU, SiLU, Mish
```

The script builds MobileNetV2 with `CustomActivation` (ReLU), not ReLU6, so
`exp008` sets `activation = relu` on the switched network.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.activation.experiments.exp000
```

## Proof against the reference

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/activation/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/activation/testdata
```

It imports the pristine script and the `gelu`, `silu` and `mish` edits from
the clone and compares with exact equality: each activation and its gradient
on fixed inputs, the 200 learning rates, the optimizer, ResNet-20's and
ResNet-56's initial state and forward, the generator after the build, every
pixel of the CIFAR-10 loaders, and every parameter after two updates per
activation. At `vgg16bn-cifar100` and `mobilenetv2-fmnist` it compares, per
activation, the full network's state and draws, the shallowest network's forwards,
and every pixel
of each setting's loaders. `reference_test.py` replays `testdata/reference.pt`
and `testdata/reference_settings.pt`.
