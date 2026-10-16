# Open-weights image AEs

Params counted from HF safetensors headers; factors/patching from repo `config.json`.

Sorted by gFID.

Each header shows its default in parentheses; a blank cell = that default (the column's most common value).

- **gFID**: lower = better. ImageNet-256, no CFG, from the Qwen-Image-VAE-2.0 report (Table 1). `†` = from a different harness (see model note); not directly comparable. Blank = not reported.
- **Compress**: bytes in / bytes out = HxW² · 3 / Cout, with uint8 RGB in and the latent Lloyd-Max quantized to uint8 per channel. Always 3 input channels, regardless of Cin. Images only: the temporal factor F is ignored.
- **F / HxW / Cout**: encoder output (latent); downsample factor vs pixels and latent channels.
- **Cin**: encoder input channels.
- **Ein Patch**: spatial rate of the encoder's first-layer patchify (pixel-unshuffle / ViT patch-embed / wavelet). Blank = none.
- **Eout Patch**: spatial patchify of the latent before it becomes a generator token, whether done in the VAE (FLUX.2) or the DiT.
- **Ee6 / De6**: encoder / decoder params, in millions.
- **E arch / D arch**: `Causal` prefix = temporally causal convs.
- **Shared with**: other models whose VAE is the same weights/arch.

| Model | gFID | Compress | F(1) | HxW(8) | Cout(16) | Cin(3) | Ein Patch | Eout Patch(2) | E arch(Conv2D) | D arch(Conv2D) | Ee6 | De6 | License(MIT) | Shared with |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [InVAE / E2E-InVAE](https://huggingface.co/REPA-E/e2e-invae-hf) | 4.09 † | 24 |  | 16 | 32 |  |  | 1 |  |  | 28 | 41 |  |  |
| [VTP-Large f16d64](https://huggingface.co/MiniMaxAI/VTP-Large-f16d64) | 5.25 | 12 |  | 16 | 64 |  | 16 | 1 | ViT-L | ViT-L | 304 | 304 | modified MIT |  |
| [VA-VAE f16d32](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2) | 6.03 | 24 |  | 16 | 32 |  |  | 1 |  |  | 28 | 41 |  |  |
| [RAE DINOv2-B](https://huggingface.co/nyu-visionx/RAE-collections) | 6.64 | 0.77 |  | 14 | 768 |  | 14 | 1 | ViT-B | ViT-XL | 87 | 415 |  |  |
| [SD1.x ft-mse-840000](https://huggingface.co/stabilityai/sd-vae-ft-mse-original) | 8.30 † | 48 |  |  | 4 |  |  | 1 |  |  | 34 | 49 |  |  |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 9.52 | 12 |  | 16 | 64 | 4 |  | 1 |  |  | 79 | 259 | Qwen Research | Qwen-Image-VAE-2.0 f16c64 |
| [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 10.61 | 6 |  | 16 | 128 |  |  |  |  |  | 34 | 50 | Apache-2.0 | FLUX.2-dev |
| [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | 15.65 | 16 | 4 | 16 | 48 |  | 2 |  | CausalConv3D | CausalConv3D | 150 | 555 | Apache-2.0 |  |
| [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | 16.25 | 12 | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 54 | 73 | Apache-2.0 | Qwen-Image, Wan2.2-A14B |
| [DC-AE f32c32](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) | 16.88 | 96 |  | 32 | 32 |  |  | 1 | Conv2D+EffViT | Conv2D+EffViT | 153 | 159 |  | Sana |
| [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-2512) | 17.68 | 12 | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 54 | 73 | Apache-2.0 | Wan2.1 |
| [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) | 19.08 | 24 | 4 | 16 | 32 |  |  | 1 | CausalConv3D | CausalConv3D | 389 | 871 | Tencent Hunyuan Community |  |
| [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | 21.29 | 12 | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 100 | 146 | Tencent Hunyuan Community |  |
| [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | 25.41 | 12 |  |  |  |  |  |  |  |  | 34 | 50 | Apache-2.0 | Z-Image, FLUX.1-dev |
| [HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1) | 33.32 | 48 |  | 32 | 64 |  |  | 1 |  |  | 145 | 261 | Tencent Hunyuan Community |  |
| [Step-Video](https://huggingface.co/stepfun-ai/stepvideo-t2v) | 33.53 | 12 | 8 | 16 | 64 |  |  | 1 | CausalConv3D+attn | CausalConv3D+attn | 110 | 389 |  |  |
| [LTX-2](https://huggingface.co/Lightricks/LTX-2) | 38.19 | 24 | 8 | 32 | 128 |  | 4 | 1 | CausalConv3D | Conv3D | 694 | 528 | LTX-2 Community |  |
| [LTX-Video 0.9](https://huggingface.co/Lightricks/LTX-Video) | 44.94 | 24 | 8 | 32 | 128 |  | 4 | 1 | CausalConv3D | Conv3D | 180 | 239 | LTXV Open Weights |  |
| [SDXL-VAE](https://huggingface.co/stabilityai/sdxl-vae) |  | 48 |  |  | 4 |  |  | 1 |  |  | 34 | 49 |  |  |
| [TAESD](https://huggingface.co/madebyollin/taesd) / [TAESDXL](https://huggingface.co/madebyollin/taesdxl) |  | 48 |  |  | 4 |  |  | 1 | TinyConv | TinyConv | 1 | 1 |  |  |
| [TAEF1](https://huggingface.co/madebyollin/taef1) / [TAESD3](https://huggingface.co/madebyollin/taesd3) |  | 12 |  |  |  |  |  |  | TinyConv | TinyConv | 1 | 1 |  |  |
| [TAEF2](https://huggingface.co/madebyollin/taef2) |  | 6 |  |  | 32 |  |  |  | TinyConv | TinyConv | 1 | 1 |  |  |
| [SD3 / 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |  | 12 |  |  |  |  |  |  |  |  | 34 | 50 | Stability Community |  |
| [FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) |  | 6 |  | 16 | 128 |  |  |  |  |  | 34 | 28 | Apache-2.0 | FLUX.2 |
| [GLM-Image](https://huggingface.co/zai-org/GLM-Image) |  | 12 |  |  |  |  |  |  |  |  | 172 | 234 |  |  |
| [CogVideoX](https://huggingface.co/zai-org/CogVideoX-5b) |  | 12 | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 92 | 123 | CogVideoX | CogVideoX-2b |
| [Mochi 1](https://huggingface.co/genmo/mochi-1-preview) |  | 16 | 6 |  | 12 | 15 |  |  | CausalConv3D+attn | CausalConv3D | 97 | 363 | Apache-2.0 |  |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |  | 24 | 8 | 32 | 128 |  | 4 | 1 | CausalConv3D | Diffusion | 319 | 417 | LTX-2.x Community |  |
| [Cosmos CV8x8x8](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CV8x8x8) |  | 12 | 8 |  |  |  | 4 |  | CausalConv3D+attn | CausalConv3D+attn | 43 | 63 | NVIDIA Open Model |  |

## Model notes

### FLUX.2
- Eout Patch 2x2 is done inside the VAE (space-to-depth, 32 ch → 128 ch); the DiT patch is 1.
- Batch-norm stats live on the 128-ch folded latent, so the fold is counted as part of the AE.

### FLUX.2 small decoder
- Reuses the full FLUX.2 encoder; only the decoder is narrowed (`decoder_block_out_channels: [96,192,384,384]`).

### Qwen-Image-2.1
- Input is RGBA (4 ch).
- gFID is the Qwen-Image-VAE-2.0 f16c64 score.
- Shared with Qwen-Image-VAE-2.0 f16c64 is inferred: E/D params match the report (76M / 248M); not stated by Qwen.
- Image-only: all 88 conv weights are 4-D (2D convs). The config's `scale_factor_temporal: 8` is vestigial.

### RAE DINOv2-B
- Encoder is a frozen DINOv2-B ViT (patch 14, 224 px input).

### VTP-Large
- Eout Patch 1: LightningDiT-XL/1 (`generation/configs/train_vtp_l_dit_xl.yaml`).
- Checkpoint also carries a 123M CLIP-style text tower (not counted in Ee6/De6).

### Mochi 1
- Encoder input is 3 RGB + 12 Fourier-feature channels.

### Cosmos CV8x8x8
- Ein Patch is a 2-level Haar wavelet transform, not a learned layer; also 4x in time.
- Official weights are TorchScript; params read from diffusers conversion in `Arsh9210/Cosmos-1.0-Diffusion-7B-Text2World` (JIT tensor bytes agree: 86 MB / 126 MB bf16).
- Eout Patch 2: Cosmos-1.0 DiT `patch_size: [1,2,2]`.

### TAEF2
- Eout Patch 2 is the FLUX.2 VAE's in-VAE 2x2 fold; TAEF2 emits the unfolded 32-ch latent.

### SD3 / 3.5
- Repo gated; params and DiT `patch_size: 2` read from ungated mirror `adamo1139/stable-diffusion-3.5-medium-ungated`.


### VA-VAE f16d32
- Official weights are `.pt`; params read from diffusers conversion `REPA-E/vavae-hf`.
- Eout Patch 1: LightningDiT-XL/1 (`lightningdit_xl_vavae_f16d32_800ep_cfg.yaml`).

### HunyuanImage-2.1
- Official weights are `.ckpt`; params and DiT `patch_size: [1,1]` read from `hunyuanvideo-community/HunyuanImage-2.1-Diffusers`.

### CogVideoX
- 5b license is the CogVideoX license; 2b is Apache-2.0.
- 2b VAE has identical param counts (92M / 123M).

### Step-Video
- Latent F8/HxW16/C64 from `FastVideo/stepvideo-t2v-diffusers` VAE config; DiT `patch_size: 1`.
- Encoder's last downsample is a 2x2x2 pixel-unshuffle (`conv_patchify`) before `conv_out`.
- All convs are `CausalConv`; mid-block attention uses `is_causal=True` (`stepvideo/vae/vae.py`).

### LTX-2.5
- Repo gated; params read from an ungated mirror (`vonkaiser/LTX-2.5-FP8-NVFP4`).
- Eout Patch 1: DiT `proj_in` is `[4096, 128]`, i.e. one 128-ch latent per token (read from `unsloth/LTX-2.5-FP8`).
- Decoder is a 1-step diffusion decoder (`NADiffusionDecoder`); its temporal causality was not determined.

### SD1.x ft-mse-840000
- gFID 8.30 †: SD-VAE + SiT-XL/2, 1400 epochs, SiT training (REPA-E README, Table B).

### InVAE / E2E-InVAE
- gFID 4.09 †: REPA-E end-to-end training from IN-VAE + SiT-XL/1, 80 epochs (REPA-E README, Table A); the VAE is tuned jointly, so this scores E2E-InVAE, not the frozen IN-VAE.
- Eout Patch 1: REPA-E pairs IN-VAE with SiT-XL/1 (REPA-E README).

## Sources

- [Qwen-Image-VAE-2.0 Technical Report](https://arxiv.org/html/2605.13565v1)
- [MiniMaxAI/VTP-Large-f16d64](https://huggingface.co/MiniMaxAI/VTP-Large-f16d64)
- [REPA-E/invae](https://huggingface.co/REPA-E/invae)
- [REPA-E README](https://github.com/End2End-Diffusion/REPA-E)
- [RAE](https://rae-dit.github.io/)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [LTX-Video](https://arxiv.org/html/2501.00103v1)
- [Cosmos Tokenizer](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CV8x8x8)
- [diffusers autoencoders](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/autoencoders)
