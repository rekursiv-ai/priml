# MLS-Bench `cv-data-augmentation`, as priml experiments

The task's editable `build_train_transform` becomes the slot
`dataset.augment` (`Cifar100Data.Config.augment`), at the `resnet56-cifar100`
setting, pinned to MLS-Bench commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  pad, crop and mirror (the pristine pipeline)
  +-- exp001  + Cutout, one 16-pixel hole after the crop
  +-- exp002  RandAugment(2, 9), then crop and mirror
  +-- exp003  TrivialAugment Wide, then crop and mirror
```

The reference runs each pipeline per image, on PIL images, in a loader worker.
The port runs it per image on uint8 tensors from that batch's worker stream,
drawing in the reference's order. `ops.py` repeats Pillow 12.3's arithmetic for
every operation the policies reach, so the pixels are Pillow's bit for bit.

## Run it

```bash
uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.augmentation.experiments.exp000
```

## Proof against the reference

```bash
uv --quiet run --frozen python priml/baselines/mlsbench/dl/augmentation/scripts/capture_reference.py --root /opt/scratch/mlsbench --out priml/baselines/mlsbench/dl/augmentation/testdata
```

It imports the pristine script and each baseline's edit from the clone and
compares with exact equality: every pixel of each training pipeline over 160
images and of the test pipeline, every pixel and label of each variant's
loaders over two passes, ResNet-56's initial state and forward, the generator
after building it, the 200 rates, the optimizer, and every parameter after two
updates per variant. It fails if any policy operation, and each signed one with
both signs, is never reached on either side. `reference_test.py` replays the stored golden.
