# ImageNet

Image classification on ILSVRC2012: 1,281,167 training photographs in 1,000
classes, scored on the 50,000-image validation split. `exp000` is the
ffcv-imagenet ResNet-50 recipe at its 16-epoch budget -- SGD with momentum,
label smoothing, a cyclic rate, progressive resizing 160->192, and testing at
256 -- on a single device.

## Reproduce

Run every command from the root of a fresh clone. `uv` installs the locked
dependencies on first use.

### Prerequisites

- Hardware: one CUDA GPU. ffcv-imagenet measured this recipe on eight A100s;
  `exp000` runs it on one device, so it takes correspondingly longer.
  `exp_smoke` takes four steps.
- Libraries: libjpeg-turbo 3.0 or later, which `pyturbojpeg` decodes with.
  The `install-libjpeg-turbo.sh` script shipped with this repository installs
  it on Ubuntu.
- Credentials: an image-net.org account that has accepted the ImageNet terms
  of access. ILSVRC2012 is licensed for non-commercial research, so no script
  downloads it.
- Disk: about 145 GB for the extracted images, plus as much again for the
  archives while extracting.

### Data

Sign in at https://image-net.org/download-images.php and download two files
from the ILSVRC2012 page: `ILSVRC2012_img_train.tar` (138 GB) and
`ILSVRC2012_img_val.tar` (6.3 GB). Extract them from the directory holding
them:

```bash
mkdir -p /opt/scratch/datasets/imagenet/train /opt/scratch/datasets/imagenet/val
tar -xf ILSVRC2012_img_val.tar -C /opt/scratch/datasets/imagenet/val
tar -xf ILSVRC2012_img_train.tar -C /opt/scratch/datasets/imagenet/train
for archive in /opt/scratch/datasets/imagenet/train/*.tar; do mkdir "${archive%.tar}" && tar -xf "$archive" -C "${archive%.tar}" && rm "$archive"; done
```

The training archive holds one inner archive per synset; the loop unpacks each
into its own directory. The result must be exactly:

```text
/opt/scratch/datasets/imagenet/
  train/n01440764/n01440764_10026.JPEG   1,000 synset directories, 1,281,167 images
  val/ILSVRC2012_val_00000001.JPEG       50,000 images, flat, as extracted
```

Keep `val/` flat. A tree sorted into per-synset folders, as some tools
produce, does not match.

Then verify the layout and write the validation labels:

```bash
uv --quiet run --frozen python -m priml.baselines.imagenet.scripts.prepare_data
```

It checks every synset directory, the training image count, and every
validation file name, and fails naming what differs. The archive carries no
validation labels, so it downloads the ILSVRC2012 devkit ground truth from a
pinned commit, checks its SHA-256, and writes `validation_labels.txt` beside
`train/` and `val/`.

### Experiments

```bash
uv --quiet run --frozen python -m priml priml.baselines.imagenet.experiments.exp_smoke
uv --quiet run --frozen python -m priml priml.baselines.imagenet.experiments.exp000
```

Both read the extracted images and the labels, so both run after the Data
step. `exp_smoke` writes nothing but its log. `exp000` writes checkpoints to
`<base_dir>/runs/imagenet/exp000/checkpoints` and logs its metrics.

| Experiment | Reads | Run after | Result | Hardware |
|---|---|---|---|---|
| `exp_smoke` | `datasets/imagenet` | Data | Installation check only | One GPU |
| `exp000` | `datasets/imagenet` | Data | TBD | -- |

`TBD` means not yet measured on reference hardware.

### The `base_dir` knob

`base_dir` defaults to `/opt/scratch`. Data and run outputs resolve beneath it,
so one override relocates a whole run. Extract to `<base_dir>/datasets/imagenet`
and point the Data step at it:

```bash
uv --quiet run --frozen python -m priml.baselines.imagenet.scripts.prepare_data --directory /data/priml/datasets/imagenet
uv --quiet run --frozen python -m priml priml.baselines.imagenet.experiments.exp000 --override base_dir=/data/priml
```

## Experiments

`exp000` is the baseline and is frozen: improvements land as forks, never as
edits, so a number measured against it stays comparable.

| Experiment | Parent | Change |
|---|---|---|
| `exp000` | -- | ffcv-imagenet ResNet-50, SGD, cyclic rate, progressive resizing, 16 epochs |

## Layout

| File | Contents |
|---|---|
| `data.py` | The ffcv-imagenet input pipelines over the extracted archive, and the DeiT ones |
| `model.py` | torchvision's ResNet-50 with ffcv-imagenet's blur pooling |
| `train_step.py` | Optimizer, schedule, label smoothing, progressive resizing |
| `experiments.py` | The configs above |
| `scripts/prepare_data.py` | Layout verification and validation labels |

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/imagenet
```

Runs on CPU. `testdata/` holds the `exp000` config snapshot and bit-for-bit
snapshots of the network and of three training steps. Regenerate the tensor
snapshots only when a numeric change is intended:

```bash
uv --quiet run --frozen pytest priml/baselines/imagenet --regenerate-b4b
```
