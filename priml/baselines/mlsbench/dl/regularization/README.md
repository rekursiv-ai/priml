# MLS-Bench `dl-regularization`, as priml experiments

The task's editable `compute_regularization(model, inputs, outputs, targets,
config)` returns a scalar added to cross-entropy every step. Here it is the
`step.regularization` slot of `RegularizedTrainStep`, a subclass of the
vision step; its default, `zero`, adds nothing. Setting `resnet56-cifar100`,
pinned to MLS-Bench commit `b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  no regularization (the pristine body raises; its docstring names zero)
  +-- exp001  dropblock: penalty on 3x3-block energy of conv kernels, delayed ramp
  +-- exp002  confidence_penalty: -0.1 * mean entropy
  +-- exp003  orthogonal_reg: 1e-4 * sum ||W W^T - I||^2 over conv weights
```

```
  +-- exp004  vgg16bn-cifar100: VGG-16-BN, 5-epoch warmup then cosine, zero
  |     +-- exp005..exp007  the three baselines
  +-- exp008  mobilenetv2-fmnist (hidden): MobileNetV2, zero
        +-- exp009..exp011  the three baselines
```

The script gives `--arch vgg*` its own `LambdaLR`: five epochs of linear
warmup, then a cosine over the other 195 (`schedules.warmup_cosine_factor`).
`exp004` carries it with the network.

`orthogonal_reg` selects 4-D weights whose name holds `conv`. The script's VGG
names its convolutions `features.<i>`, so on VGG the baseline penalizes
nothing and its term is zero (`exp007`, `select=complement(everything)`); on
MobileNetV2 every convolution is named `conv*` (`exp011`,
`select=with_ndim(everything, 4)`).

The `dropblock` baseline changes no model: it is a weight penalty, so all
three baselines fill the same slot.

The recipe is the launch script's, not the training script's defaults: the
script defaults `--weight-decay` to 0, and `resnet56_cifar100.sh` passes
`5e-4`. The capture traces `main` with the launch script's own arguments.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.regularization.experiments.exp000
```

## Proof against the reference

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/regularization/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/regularization/testdata
```

It checks that the pristine regularizer raises, then compares with exact
equality: each regularizer's value and gradients (logits and every parameter)
on a probe network at three epochs, the batch size, the rate of all 200
epochs, the optimizer, ResNet-56's initial state and forward, the generator
position after the full build, every pixel the loaders produce, and every
parameter after two updates per regularizer. At the other two settings it
compares each regularizer's full network state, draws and forwards; the
setting's 200 rates (VGG's warmup included), optimizer and batch size, traced
once with its own launch script; each regularizer's value and gradients on the
shallowest network of the setting's architecture; two updates of
the shallowest network the script's own `VGG16_CFG` / `MobileNetV2.CFG`
builds (patched to the vision capture's `TINY_ARCHS`, on both sides); and
every Fashion-MNIST loader pixel. `reference_test.py` replays
`testdata/reference.pt` and `testdata/reference_settings.pt`.
