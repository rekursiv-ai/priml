# SpeedrunDiT

One baseline package implements the REG/SPRINT recipe from
[`SwayStar123/REG` at `3c51606`](https://github.com/SwayStar123/REG/tree/3c51606c801dd9e87ee9ef778782766ab7c379ca).

| Experiment | Model and loss | Numerical implementation |
| --- | --- | --- |
| `exp000` | SiT-B/1, RMSNorm, QK norm, first-layer value residual, SPRINT routing, layerwise MLP widths, REG projection, CLS flow, contrastive flow, AdamW + Muon | REG position-table, rotary backward, and Muon operation order |
| `exp001` | Same model and loss | Float32 position-table computation and the shared rotary and fused Muon arithmetic |
| `exp_smoke` | `exp000` mechanisms at a tiny width and five updates | Single-process CPU-friendly check |

`source_parity_test.py` checks `exp000`'s initialization, model outputs,
backward gradients, losses, and five optimizer updates against a golden made
from that REG commit. `reg_source.pt` contains the source artifact; the test
does not need the REG repository or `timm` at runtime. The CPU parity test does
not establish bitwise identity across different GPU kernels or distributed
launches.

## Data

Training reads paired RGB images and sampled, unscaled 32-channel INVAE
latents from a processed corpus. DINOv2 targets are computed online from the
images; they are not stored as latent files. Existing corpora in the REG
`images/` and `vae-in/` layout are accepted directly. Each latent filename is
listed with its class ID in `vae-in/dataset.json`.

To prepare extracted ImageNet locally:

```sh
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data \
  --source /datasets/imagenet \
  --output /datasets/speedrundit \
  --device cuda
```

The local [INVAE implementation](../../model/invae.py) loads the published
`REPA-E/e2e-invae` checkpoint. Pass `--checkpoint` to use a local copy. The
training step applies INVAE's `0.3099` latent scale. Dataset preparation
reuses Priml's extracted ImageNet source and synset mapping.
