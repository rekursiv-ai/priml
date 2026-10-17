# MLS-Bench `dl-residual-connection`, as priml experiments

The task's editable `CustomBlock` becomes `step.model.block`, a slot of
`ResNet.Config` holding one block template or one config per position. The
setting is `resnet56-cifar100`, pinned to MLS-Bench commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. The recipe is the
`cv-classification-loss` recipe (see `../vision/`), so `exp000` is that
task's `exp000` under a new study name.

```
exp000  basic block (pristine)
  +-- exp001  pre-activation, branch scaled by a learned scalar from 0.1
  +-- exp002  basic block, branch gated by a learned scalar from 1
  +-- exp003  stochastic depth, deepest block survives with probability 0.5
```

Upstream numbers its stochastic-depth blocks with a class-wide counter that
restarts on every 16-to-16-channel block, so in ResNet-56 all nine
first-stage blocks sit at position 1 and the rest at 2 to 19.
`StochasticDepthBlock.Config.place` reproduces that numbering at any depth.

The unpublished `bottleneck` and `se_block` edits are not ported.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.residual.experiments.exp000
```

## Proof against upstream

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/residual/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/residual/testdata
```

The capture compares each block alone (identity and projected), including its
gradients and stochastic depth's `torch.rand` draws. It also compares ResNet-56
with every block, the generator position after each build, the schedule,
optimizer and loaders, and two lockstep updates per block. Equality is exact.
`reference_test.py` replays the stored values.
