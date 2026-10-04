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

`source_parity_test.py` checks a tiny `exp000`-mechanism model's forward
outputs, kept token ids, three losses, and final weights against
`reg_source.pt`. That golden was minted after the same run of the REG commit's
own code was shown bit-for-bit equal to it, so the test does not need the REG
repository or `timm` at runtime. The CPU check does not establish bitwise
identity across different GPU kernels or distributed launches.

## Data

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
`corpus.json` (the receipt), and, for a fitted codec, `codec.pt`. The loader
compares the receipt's autoencoder, checkpoint digests, latent shape, codec,
and table digest with the experiment's config and refuses a mismatch.

Prepare an experiment's corpus from extracted ImageNet:

```sh
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --experiment exp003 --source /datasets/imagenet --device cuda
```

The preparer builds exactly the autoencoder and codec the experiment declares,
fits a fitted codec first on a subset of the images, and resumes an
interrupted run. Autoencoder weights are pinned Hugging Face revisions with
SHA-256 digests and download into the Hugging Face cache. An existing REG
corpus is admitted without re-encoding:

```sh
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --experiment exp000 --receipt-only
```

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
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.benchmark_codec --experiment exp003 --source /datasets/imagenet --decode --device cuda --output codecs.json
```

It scores every candidate codec (float32, bfloat16, float16, and uint8 tables
fitted uniformly, clipped, by quantile, by Gaussian companding, and by
Lloyd-Max per channel, per scale group, and shared) in the normalized space the
model trains in and, with `--decode`, in decoded pixels (PSNR, LPIPS), and
reports how the held-out error falls as the fitting set grows. Whether uint8
storage changes the trained model is `exp004` against `exp003`.

## Autoencoders

The implementations live in [`priml/model/vision_ae`](../../model/vision_ae):
`INVAE` (variational; `encode` draws from the posterior as the REG corpora
do), `VTP`, and `RAE` (both deterministic). Each takes uint8 images and returns
raw latents; each config states its latent shape and published normalizer.
