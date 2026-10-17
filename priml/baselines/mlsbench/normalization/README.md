# `dl-normalization`

MLS-Bench `dl-normalization`: ResNet-56 on CIFAR-100 with the
normalization layer as a slot, `step.model.norm`. The model builds the stem's,
every block's and every projection's norm from it.

```
exp000  batch norm (pristine)
  +-- exp001  group norm
  +-- exp002  batch-instance norm
  +-- exp003  switchable norm
```

`exp000` is `pytorch_vision`'s `exp000`. The pristine `CustomNorm` wraps
`nn.BatchNorm2d`, which only renames the state's keys.

## Run

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.normalization.experiments.exp000
```

## Capture

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/normalization/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/normalization/testdata
```

It compares each norm's output and gradients, ResNet-56's initial state and
forward under every norm, the schedule, the optimizer, the loaders, and a
two-update lockstep per norm, all with exact equality.
`reference_test.py` replays `testdata/reference.pt`.
