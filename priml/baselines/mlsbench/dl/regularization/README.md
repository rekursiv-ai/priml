# MLS-Bench `dl-regularization`, as priml experiments

The task's editable `compute_regularization(model, inputs, outputs, targets,
config)` returns a scalar added to cross-entropy every step. Here it is the
`step.regularization` slot of `RegularizedTrainStep`, a subclass of the
vision step; its default, `zero`, adds nothing. Setting `resnet56-cifar100`,
pinned to MLS-Bench commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  no regularization (the pristine body raises; its docstring names zero)
  +-- exp001  dropblock: penalty on 3x3-block energy of conv kernels, delayed ramp
  +-- exp002  confidence_penalty: -0.1 * mean entropy
  +-- exp003  orthogonal_reg: 1e-4 * sum ||W W^T - I||^2 over conv weights
```

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
parameter after two updates per regularizer. `reference_test.py` replays
`testdata/reference.pt`.
