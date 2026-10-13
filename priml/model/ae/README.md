# Open-weights image AEs

Params counted from HF safetensors headers; factors/patching from repo `config.json`.

Each header shows its default in parentheses; a blank cell = that default (the column's most common value). `?` = unknown.

- **F / HxW / C**: encoder output (latent); downsample factor vs pixels and latent channels.
- **Cin**: encoder input channels.
- **Ein Patch**: spatial rate of the encoder's first-layer patchify (pixel-unshuffle / ViT patch-embed / wavelet). Blank = none.
- **Eout Patch**: spatial patchify of the latent before it becomes a generator token, whether done in the VAE (FLUX.2) or the DiT.
- **Ee6 / De6**: encoder / decoder params, in millions.
- **E arch / D arch**: `Causal` prefix = temporally causal convs.
- **Shared with**: other models whose VAE is the same weights/arch.

| Model | F(1) | HxW(8) | C(16) | Cin(3) | Ein Patch | Eout Patch(2) | E arch(Conv2D) | D arch(Conv2D) | Ee6 | De6 | License(MIT) | Shared with |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [SD1.x ft-mse-840000](https://huggingface.co/stabilityai/sd-vae-ft-mse-original) |  |  | 4 |  |  | 1 |  |  | 34 | 49 |  |  |
| [SDXL-VAE](https://huggingface.co/stabilityai/sdxl-vae) |  |  | 4 |  |  | 1 |  |  | 34 | 49 |  |  |
| [TAESD](https://huggingface.co/madebyollin/taesd) / [TAESDXL](https://huggingface.co/madebyollin/taesdxl) |  |  | 4 |  |  | 1 | TinyConv | TinyConv | 1 | 1 |  |  |
| [TAEF1](https://huggingface.co/madebyollin/taef1) / [TAESD3](https://huggingface.co/madebyollin/taesd3) |  |  |  |  |  |  | TinyConv | TinyConv | 1 | 1 |  |  |
| [TAEF2](https://huggingface.co/madebyollin/taef2) |  |  | 32 |  |  |  | TinyConv | TinyConv | 1 | 1 |  |  |
| [SD3 / 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |  |  |  |  |  |  |  |  | ? | ? | Stability Community |  |
| [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |  |  |  |  |  |  |  |  | 34 | 50 | Apache-2.0 | Z-Image, FLUX.1-dev |
| [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |  | 16 | 128 |  |  |  |  |  | 34 | 50 | Apache-2.0 | FLUX.2-dev |
| [FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) |  | 16 | 128 |  |  |  |  |  | 34 | 28 | Apache-2.0 | FLUX.2 |
| [GLM-Image](https://huggingface.co/zai-org/GLM-Image) |  |  |  |  |  |  |  |  | 172 | 234 |  |  |
| [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-2512) | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 54 | 73 | Apache-2.0 | Wan2.1 |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) |  | 16 | 64 | 4 |  | 1 |  |  | 79 | 259 | Qwen Research | Qwen-Image-VAE-2.0 f16c64 |
| [InVAE / E2E-InVAE](https://huggingface.co/REPA-E/e2e-invae-hf) |  | 16 | 32 |  |  | 1 |  |  | 28 | 41 |  |  |
| [VA-VAE f16d32](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2) |  | 16 | 32 |  |  | ? |  |  | ? | ? |  |  |
| [RAE DINOv2-B](https://huggingface.co/nyu-visionx/RAE-collections) |  | 14 | 768 |  | 14 | 1 | ViT-B | ViT-XL | 87 | 415 |  |  |
| [VTP-Large f16d64](https://huggingface.co/MiniMaxAI/VTP-Large-f16d64) |  | 16 | 64 |  | 16 | ? | ViT-L | ViT-L | 304 | 304 | modified MIT |  |
| [DC-AE f32c32](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) |  | 32 | 32 |  |  | 1 | Conv2D+EffViT | Conv2D+EffViT | 153 | 159 |  | Sana |
| [HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1) |  | 32 | 64 |  |  | 1 |  |  | ? | ? | Tencent Hunyuan Community |  |
| [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 54 | 73 | Apache-2.0 | Qwen-Image, Wan2.2-A14B |
| [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | 4 | 16 | 48 |  | 2 |  | CausalConv3D | CausalConv3D | 150 | 555 | Apache-2.0 |  |
| [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 100 | 146 | Tencent Hunyuan Community |  |
| [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) | 4 | 16 | 32 |  |  | 1 | CausalConv3D | CausalConv3D | 389 | 871 | Tencent Hunyuan Community |  |
| [CogVideoX](https://huggingface.co/zai-org/CogVideoX-5b) | 4 |  |  |  |  |  | CausalConv3D | CausalConv3D | 92 | 123 | CogVideoX | CogVideoX-2b |
| [Mochi 1](https://huggingface.co/genmo/mochi-1-preview) | 6 |  | 12 | 15 |  |  | CausalConv3D+attn | CausalConv3D | 97 | 363 | Apache-2.0 |  |
| [Step-Video](https://huggingface.co/stepfun-ai/stepvideo-t2v) | 8 | 16 | 64 |  | ? | ? | CausalConv3D | CausalConv3D | 110 | 389 |  |  |
| [LTX-Video 0.9](https://huggingface.co/Lightricks/LTX-Video) | 8 | 32 | 128 |  | 4 | 1 | CausalConv3D | Conv3D | 180 | 239 | LTXV Open Weights |  |
| [LTX-2](https://huggingface.co/Lightricks/LTX-2) | 8 | 32 | 128 |  | 4 | 1 | CausalConv3D | Conv3D | 694 | 528 | LTX-2 Community |  |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 8 | 32 | 128 |  | 4 | 1 | CausalConv3D | Diffusion | 319 | 417 | LTX-2.x Community |  |
| [Cosmos CV8x8x8](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CV8x8x8) | 8 |  |  |  | 4 | ? | CausalConv3D+attn | CausalConv3D+attn | ? | ? | NVIDIA Open Model |  |

## Model notes

### FLUX.2
- Eout Patch 2x2 is done inside the VAE (space-to-depth, 32 ch → 128 ch); the DiT patch is 1.
- Batch-norm stats live on the 128-ch folded latent, so the fold is counted as part of the AE.

### FLUX.2 small decoder
- Reuses the full FLUX.2 encoder; only the decoder is narrowed (`decoder_block_out_channels: [96,192,384,384]`).

### Qwen-Image-2.1
- Input is RGBA (4 ch).
- Shared with Qwen-Image-VAE-2.0 f16c64 is inferred: E/D params match the report (76M / 248M); not stated by Qwen.
- Image-only: all 88 conv weights are 4-D (2D convs). The config's `scale_factor_temporal: 8` is vestigial.

### RAE DINOv2-B
- Encoder is a frozen DINOv2-B ViT (patch 14, 224 px input).

### VTP-Large
- Checkpoint also carries a 123M CLIP-style text tower (not counted in Ee6/De6).

### Mochi 1
- Encoder input is 3 RGB + 12 Fourier-feature channels.

### Cosmos CV8x8x8
- Ein Patch is a 2-level Haar wavelet transform, not a learned layer; also 4x in time.

### TAEF2
- Eout Patch 2 assumes FLUX.2-style use.

### SD3 / 3.5
- Repo gated: Eout Patch 2 and params not read from checkpoint.

### InVAE / E2E-InVAE
- Eout Patch 1 assumes the REPA-E SiT-XL/1 setup; not read from config.

### VA-VAE f16d32
- `.pt` only; arch (Conv2D assumed) and params not inspected.

### HunyuanImage-2.1
- `.ckpt` only; params and Eout Patch not inspected.

### CogVideoX
- 5b license is the CogVideoX license; 2b is Apache-2.0.
- 2b VAE assumed identical to 5b; not diffed.

### Step-Video
- No diffusers config; latent C, arch, causality, and patching not read from source.

### LTX-2.5
- Repo gated; params read from an ungated mirror (`vonkaiser/LTX-2.5-FP8-NVFP4`).
- Eout Patch 1 assumed from LTX-2; transformer config not read.
- Decoder is a 1-step diffusion decoder (`NADiffusionDecoder`); its temporal causality is unknown.

## Sources

- [Qwen-Image-VAE-2.0 Technical Report](https://arxiv.org/html/2605.13565v1)
- [MiniMaxAI/VTP-Large-f16d64](https://huggingface.co/MiniMaxAI/VTP-Large-f16d64)
- [REPA-E/invae](https://huggingface.co/REPA-E/invae)
- [RAE](https://rae-dit.github.io/)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [LTX-Video](https://arxiv.org/html/2501.00103v1)
- [Cosmos Tokenizer](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CV8x8x8)
- [diffusers autoencoders](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/autoencoders)
