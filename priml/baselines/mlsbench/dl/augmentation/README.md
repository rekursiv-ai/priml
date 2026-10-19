# MLS-Bench `cv-data-augmentation`, as priml experiments

The task's editable `build_train_transform` becomes the slot
`dataset.augment` (`Cifar100Data.Config.augment`). All three settings are
ported, pinned to MLS-Bench commit `80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`.

```
exp000  ResNet-56, CIFAR-100, pad, crop and mirror   resnet56-cifar100
  +-- exp001  + Cutout, one 16-pixel hole after the crop
  +-- exp002  RandAugment(2, 9), then crop and mirror
  +-- exp003  TrivialAugment Wide, then crop and mirror
  +-- exp004  ResNet-20, CIFAR-10                     resnet20-cifar10
  |     +-- exp005 / exp006 / exp007  the three augmentations above
  +-- exp008  MobileNetV2, Fashion-MNIST              mobilenetv2-fmnist (hidden)
        +-- exp009 / exp010 / exp011  the three augmentations above
```

The reference runs each pipeline per image, on PIL images, in a loader worker.
The port runs it per image on uint8 tensors from that batch's worker stream,
drawing in the reference's order. `ops.py` repeats Pillow 12.3's arithmetic for
every operation the policies reach, so the pixels are Pillow's bit for bit.

On Fashion-MNIST the reference feeds the policies grayscale `L` images after
its `Resize(32)` and repeats the channel after `ToTensor`; the slot likewise
receives the prepared `(B, 1, 32, 32)` images and `normalize` repeats the
channel. Every operation is tested against Pillow on `L` images (`ops_test.py`),
where `Color` is the identity because `convert("L")` is.

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
updates per variant. At `resnet20-cifar10` and `mobilenetv2-fmnist` it also
compares, per variant, the full network's state and the shallowest network's forwards, every pixel of
the training pipeline over 160 images and of the loaders, and the test
pipeline. It fails if any policy operation, and each signed one with both
signs, is never reached on either side at any setting. `reference_test.py`
replays the stored goldens.
