# MLS-Bench `dl-normalization`, as priml experiments

MLS-Bench `dl-normalization`: ResNet-56 on CIFAR-100 with the
normalization layer as a slot, `step.model.norm`. The model builds the stem's,
every block's and every projection's norm from it.

```
exp000  batch norm (pristine)
  +-- exp001  group norm
  +-- exp002  batch-instance norm
  +-- exp003  switchable norm
  +-- exp004  ResNet-110 on CIFAR-100 (hidden setting), batch norm
  |     +-- exp005..exp007  group, batch-instance, switchable norm
  +-- exp008  MobileNetV2 on Fashion-MNIST, batch norm
        +-- exp009..exp011  group, batch-instance, switchable norm
```

The script also has a `vgg16bn-cifar100` launch script, but the task's
`config.json` does not score it, so it is not ported.

`exp000` is `vision`'s `exp000`. The pristine `CustomNorm` wraps
`nn.BatchNorm2d`, which only renames the state's keys.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.normalization.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/normalization/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/normalization/testdata
```

It compares, all with exact equality: each norm's output and gradients,
group norm's group counts, ResNet-56's initial state and forward under every
norm, where the generator stands after building each (`draws/<norm>`), the
schedule, the optimizer, the reported best accuracy (`best/accuracy`), the
loaders, and a two-update lockstep per norm. At `resnet110-cifar100` and
`mobilenetv2-fmnist` it compares, per norm, the full network's state after
`main` seeds and initializes it and its draws, the shallowest network's
forwards, and every pixel of the
Fashion-MNIST loaders. `reference_test.py` replays `testdata/reference.pt` and
`testdata/reference_settings.pt`.
