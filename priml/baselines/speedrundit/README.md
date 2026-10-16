# SpeedrunDiT

One baseline package implements the REG/SPRINT recipe from
[`SwayStar123/REG` at `3c51606`](https://github.com/SwayStar123/REG/tree/3c51606c801dd9e87ee9ef778782766ab7c379ca).

| Experiment | Model and loss | Numerical implementation |
| --- | --- | --- |
| `exp000` | SiT-B/1, RMSNorm, QK norm, first-layer value residual, SPRINT routing, layerwise MLP widths, REG projection, CLS flow, contrastive flow, AdamW + Muon | REG position-table, rotary backward, and Muon operation order |
| `exp001` | Same model and loss | Float32 position-table computation and the shared rotary and fused Muon arithmetic |
| `exp_smoke` | `exp000` mechanisms at a tiny width and five updates | Single-process CPU-friendly check |

`source_parity_test.py` checks a tiny `exp000`-mechanism model's forward
outputs, kept token ids, three losses, and final weights against
`reg_source.pt`. That golden was minted after the same run of the REG commit's
own code was shown bit-for-bit equal to it, so the test does not need the REG
repository or `timm` at runtime. The CPU check does not establish bitwise
identity across different GPU kernels or distributed launches.

## Reproduce

Training reads paired RGB images and sampled, unscaled 32-channel INVAE
latents from a processed corpus. DINOv2 targets are computed online from the
images; they are not stored as latent files.

### Prerequisites

- **ImageNet.** The ILSVRC2012 training set is licensed. Request access at
  <https://image-net.org/download.php>, accept its terms, and download
  `ILSVRC2012_img_train.tar` (about 138 GB). Extract it so each of the 1,000
  synsets is one directory of JPEGs:

  ```bash
  mkdir -p /data/imagenet/train && tar -xf ILSVRC2012_img_train.tar -C /data/imagenet/train
  cd /data/imagenet/train && for f in *.tar; do mkdir "${f%.tar}" && tar -xf "$f" -C "${f%.tar}" && rm "$f"; done
  ```

  The result is `/data/imagenet/train/<synset>/<synset>_<id>.JPEG`. The
  preparation reads only `train/` and refuses a tree without exactly the
  1,000 synset directories.
- **Weights.** The preparation downloads the `REPA-E/e2e-invae` checkpoint
  from Hugging Face at a pinned revision and checks its SHA-256. Training
  loads DINOv2 ViT-B/14 through `torch.hub` from `facebookresearch/dinov2`
  and its published weights. No credentials are needed.
- **GPU.** The preparation encodes on one CUDA GPU. `exp000` and `exp001` are
  data-parallel runs of 32 images per GPU; eight GPUs make REG's global batch
  of 256. `exp_smoke` runs on one device, CPU included.
- **Disk.** The processed corpus holds one 256-pixel PNG and one latent per
  training image, about 1.28 million of each.

### Data

```bash
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --source /data/imagenet --output /opt/scratch/datasets/speedrundit
```

It center-crops every training image to 256 pixels with REG's ADM recipe,
writes it to `images/`, encodes it with INVAE into `vae-in/`, and lists each
latent with its class ID in `vae-in/dataset.json`. `--checkpoint` names a
local copy of the INVAE checkpoint, checked against the same digest, and
`--limit` stops after that many images. Existing corpora in REG's `images/`
and `vae-in/` layout are accepted directly. The training step applies INVAE's
`0.3099` latent scale.

### Experiments

```bash
uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp_smoke
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.speedrundit.experiments.exp000
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.speedrundit.experiments.exp001
```

| Experiment | Reads | Run after | Writes | Recorded result |
|---|---|---|---|---|
| `exp_smoke` | `datasets/speedrundit/` | data step | `runs/speedrundit/exp_smoke` | Installation check: five updates |
| `exp000` | `datasets/speedrundit/` | data step | `runs/speedrundit/exp000`, a checkpoint every 10,000 steps | TBD |
| `exp001` | `datasets/speedrundit/` | data step | `runs/speedrundit/exp001` | TBD |

Paths sit beneath `/opt/scratch`. `exp000` and `exp001` train for 400,000
steps and evaluate nothing during training; REG scores generated images
separately.

### The base_dir knob

Every default sits beneath `/opt/scratch`. To move everything, pick a root:
pass `--output /data/root/datasets/speedrundit` to the data step and
`--override base_dir=/data/root` to each launch. The launch then reads from
`/data/root/datasets/speedrundit` and writes to `/data/root/runs/speedrundit`.
