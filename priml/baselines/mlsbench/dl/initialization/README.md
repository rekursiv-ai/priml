# MLS-Bench `dl-weight-initialization`, as priml experiments

The task's editable `initialize_weights(model, config)` becomes slots of
`ResNet.Config`: the per-tensor `init_conv` and `init_head`, and `init`, which
runs on the built network afterwards for a rule that needs its structure
(Fixup). Setting `resnet56-cifar100`, MLS-Bench commit
`b873944969c53b01e4ee6e843c77767b8c702485`.

```
exp000  normal, std 0.01 (the pristine script)
  +-- exp001  Kaiming normal         (baseline kaiming_normal)
  +-- exp002  Kaiming, then Fixup    (baseline fixup)
  +-- exp003  orthogonal, gain sqrt(2) (baseline orthogonal)
  +-- exp004  VGG-16-BN on CIFAR-100, normal, std 0.01
  |     +-- exp005..exp007  Kaiming, Fixup, orthogonal
  +-- exp008  MobileNetV2 on Fashion-MNIST (hidden), normal, std 0.01
        +-- exp009..exp011  Kaiming, Fixup, orthogonal
```

Upstream's `fixup` edit scales residual branches only when `--arch` starts
with `resnet`, so on VGG-16-BN and MobileNetV2 it is its Kaiming phase alone:
exp006 and exp010 equal exp005 and exp009 apart from the name, and keep
`init` at `keep_initialization`. `initializers.fixup` still refuses a network
without `BasicBlock`s.

The pristine script is the `cv-classification-loss` script with the loss fixed
to `nn.CrossEntropyLoss` and the initialization editable, so `exp000` is that
task's `exp000` with `init_conv` and `init_head` set to `small_normal`. The
data, model, optimizer and cosine schedule are reused unchanged.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.initialization.experiments.exp000
```

## Proof against the script

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/initialization/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/initialization/testdata
```

It imports the pristine script and each baseline's edit from the clone and
compares with exact equality, per initialization: ResNet-56's state after
`main` seeds and initializes it, and its forward; where the generator stands
afterwards; the shallowest network's state and forward; every parameter after
each of two updates. It also compares the learning rates, the optimizer and
every pixel the loaders produce. At `vgg16bn-cifar100` and `mobilenetv2-fmnist`
it compares, per initialization, the full network's state after `main`
initializes it and its draws, the shallowest network's forwards, and every
pixel of the Fashion-MNIST
loaders. `reference_test.py` replays `testdata/reference.pt` and
`testdata/reference_settings.pt`.

Not compared here: the best-accuracy bookkeeping (`best/accuracy`), which the
script shares unchanged with `cv-classification-loss` and is proven by that
capture. The capture runs `main` at the script's argparse defaults, not the
launch `.sh`; `experiments_test.py` pins the `.sh` values (lr, momentum,
weight decay, batch size, epochs) against `exp000`.
