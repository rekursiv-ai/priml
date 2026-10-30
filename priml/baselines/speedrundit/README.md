# SpeedrunDiT

One baseline package implements the REG/SPRINT recipe from
[`SwayStar123/REG` at `3c51606`](https://github.com/SwayStar123/REG/tree/3c51606c801dd9e87ee9ef778782766ab7c379ca),
trained on the latents of a pretrained vision autoencoder.

| Experiment | Parent | Change |
| --- | --- | --- |
| `exp000` | -- | SiT-B/1, RMSNorm, QK norm, first-layer value residual, SPRINT routing, layerwise MLP widths, REG projection, CLS flow, contrastive flow, AdamW + Muon, on INVAE latents |
| `exp001` | `exp000` | Float32 position table and the shared rotary and fused Muon arithmetic |
| `exp002` | `exp000` | VTP-Large latents (64 channels) |
| `exp003` | `exp000` | RAE DINOv2-B latents (768 channels), stored float16 |
| `exp004` | `exp003` | The same RAE latents stored as uint8 Lloyd-Max indices |
| `exp_smoke` | `exp000` | `exp000` mechanisms at a tiny width and five updates |

`source_parity_test.py` checks a tiny `exp000`-mechanism model's initial
weights, three training losses, and final weights against `reg_source.pt`.
That golden was minted after the same run of the REG commit's
own code was shown bit-for-bit equal to it, so the test does not need the REG
repository or `timm` at runtime. The CPU check does not establish bitwise
identity across different GPU kernels or distributed launches.

## Reproduce

Training reads paired RGB images and raw autoencoder latents from the
experiment's corpus. DINOv2 targets are computed online from the images; they
are not stored as latent files.

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
- **Weights.** The preparation downloads the experiment's autoencoder
  checkpoint from Hugging Face at a pinned revision and checks its SHA-256.
  Training loads DINOv2 ViT-B/14 through `torch.hub` from
  `facebookresearch/dinov2` and its published weights. No credentials are
  needed.
- **GPU.** The preparation encodes on one CUDA GPU. `exp000` through `exp004`
  are data-parallel runs of 32 images per GPU; eight GPUs make REG's global
  batch of 256. `exp_smoke` runs on one device, CPU included.
- **Disk.** The shared `images/` directory holds one 256-pixel PNG per
  training image, about 1.28 million; each corpus adds its latents (see
  [Latent storage](#latent-storage)).

### Data

An experiment owns its corpus. `dataset.source` names the autoencoder that
encodes it, the storage codec that writes it, and the subdirectory it lives in
beside one shared `images/` directory:

```
/opt/scratch/datasets/speedrundit/
  images/            ADM center crops, shared by every corpus
  vae-in/            INVAE latents (the REG layout), float32
  vtp-large/         VTP-Large latents, float32
  rae-dinov2-base-f16/   RAE latents, float16
  rae-dinov2-base-u8/    RAE latents, uint8 indices + codec.pt
```

Each latent directory holds one `.npy` per image, `dataset.json` (labels),
`corpus.json` (the receipt), and, for a fitted codec, `codec.pt`;
`images/source.json` binds the shared crops to their ImageNet source. The
loader compares the receipt's autoencoder encoding config (checkpoints by
digest), latent shape, codec, and table digest with the experiment's config,
and refuses a mismatch or a corpus whose preparation has not finished.

Prepare an experiment's corpus from extracted ImageNet:

```sh
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --experiment exp003 --source /data/imagenet --device cuda
```

The preparer builds exactly the autoencoder and codec the experiment declares,
fits a fitted codec first on a sample of every source image, and resumes an
interrupted run when the source, producers, seed, and (for a seeded corpus)
batch size are unchanged. Autoencoder weights download into the Hugging Face
cache. An existing REG corpus is admitted without re-encoding; `--source` also
binds crops that record no source, so other corpora can be prepared beside it:

```sh
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --experiment exp000 --receipt-only --source /data/imagenet
```

### Experiments

```bash
uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp_smoke
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.speedrundit.experiments.exp000
```

The other experiments launch as `exp000` does.

| Experiment | Reads | Writes | Recorded result |
|---|---|---|---|
| `exp_smoke` | `datasets/speedrundit/vae-in/` | `runs/speedrundit/exp_smoke` | Installation check: five updates |
| `exp000` | `datasets/speedrundit/vae-in/` | `runs/speedrundit/exp000` | TBD |
| `exp001` | `datasets/speedrundit/vae-in/` | `runs/speedrundit/exp001` | TBD |
| `exp002` | `datasets/speedrundit/vtp-large/` | `runs/speedrundit/exp002` | TBD |
| `exp003` | `datasets/speedrundit/rae-dinov2-base-f16/` | `runs/speedrundit/exp003` | TBD |
| `exp004` | `datasets/speedrundit/rae-dinov2-base-u8/` | `runs/speedrundit/exp004` | TBD |

Each reads its corpus beside the shared `images/`, prepared by the data step
for that experiment. Paths sit beneath `/opt/scratch`. Every experiment except
`exp_smoke` trains for 400,000 steps, writes a checkpoint every 10,000, and
evaluates nothing during training; REG scores generated images separately.

### The base_dir knob

Every default sits beneath `/opt/scratch`. To move everything, pick a root:
pass `--directory /data/root/datasets/speedrundit` to the data step and
`--override base_dir=/data/root` to each launch. The launch then reads from
`/data/root/datasets/speedrundit` and writes to `/data/root/runs/speedrundit`.

## Latent storage

Latents are stored raw; the train step's `latent_norm` (filled from the
autoencoder's published statistics) maps them into diffusion space, so a
corpus never has to be re-encoded to change normalization. The codec is the
separate question of how the raw latent is written:

| Autoencoder | Latent | float32 | float16 | uint8 |
| --- | --- | --- | --- | --- |
| INVAE | 32 x 16 x 16 | 42 GB | 21 GB | 10.5 GB |
| VTP | 64 x 16 x 16 | 84 GB | 42 GB | 21 GB |
| RAE DINOv2-B | 768 x 16 x 16 | 1.01 TB | 504 GB | 252 GB |

Sizes are for the 1,281,167 ImageNet-1k training images, latents only.

`uint8` stores one index per scalar against a per-channel table of 256
float32 levels. The default table is Lloyd-Max, started at the cube-root
density; per-channel tables matter because RAE's published per-element
variances span 1,120x across channels (and at most 3.2x across positions
within one). Before a uint8 corpus replaces the float16 one, measure it:

```sh
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.benchmark_codec --experiment exp003 --source /data/imagenet --decode --device cuda --output /opt/scratch/artifacts/speedrundit/codecs.json
```

It scores every candidate codec (float32, bfloat16, float16, and uint8 tables
fitted uniformly, clipped, by quantile, by Gaussian companding, and by
Lloyd-Max per channel, per scale group, and shared) in the normalized space the
model trains in and, with `--decode`, in decoded pixels (PSNR, LPIPS), and
reports how the held-out error falls as the fitting set grows. Whether uint8
storage changes the trained model is `exp004` against `exp003`.

## Autoencoders

The implementations live in [`priml/model/ae`](../../model/ae): INVAE
(`kl.invae()`, an `AutoencoderKL`), `VTP`, and `RAE`. Each `encode` takes
uint8 images and returns a distribution over raw latents: a Gaussian for
INVAE, a point mass for VTP and RAE. The
corpus stores a draw by default, as the REG corpora do; set
`dataset.source.sample = False` to store the mode. Each config states its
latent shape and published normalizer.
