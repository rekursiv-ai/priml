# MLS-Bench `dl-lr-schedule`

The learning-rate schedule as a config slot, `step.schedule`, on the
`resnet56-cifar100` setting. The task's script is the `cv-classification-loss`
script with its scheduler removed, so the network, data and loss are
[`vision/`](../vision/README.md)'s, already proven there.

```
exp000  constant (pristine)
  +-- exp001  cosine annealing
  +-- exp002  5-epoch linear warmup, then cosine
  +-- exp003  one cycle
```

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.lr_schedule.experiments.exp000
```

## Proof against the script

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/lr_schedule/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/lr_schedule/testdata
```

It runs the pristine script and each baseline's edit from the clone, and
compares with exact equality: every schedule's rate over all 200 epochs, the
optimizer, ResNet-56's initial state and forward, the generator after the
build, every loader pixel over two passes, and every parameter after each
update of a two-update run per schedule. `reference_test.py` replays the
stored values in the default tier.
