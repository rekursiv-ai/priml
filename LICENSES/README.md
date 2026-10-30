# Third-party licenses

priml is Apache-2.0 (`../LICENSE`). The code below ports or adapts these
projects; each file is that project's license, verbatim. A module citing one
names its file here.

| File | Project | License | Used by |
|---|---|---|---|
| `Craftax` | [MichaelTMatthews/Craftax](https://github.com/MichaelTMatthews/Craftax) | MIT | `baselines/craftax` |
| `Craftax_Baselines` | [MichaelTMatthews/Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) @ `7ce36fa` | MIT | `baselines/craftax` |
| `JAX` | [jax-ml/jax](https://github.com/jax-ml/jax) | Apache-2.0 | `model/init.py` |
| `LightningDiT` | [JingfengYao/LightningDiT](https://github.com/JingfengYao/LightningDiT) @ `9860985` | MIT | `model/ae/latent_norm.py` |
| `Muon` | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | MIT | `optimizers/muon.py` |
| `PSLP` | [dance858/PSLP](https://github.com/dance858/PSLP) | Apache-2.0 | `baselines/convextok/presolver` |
| `PufferLib` | [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) @ `6ffa5b10` | MIT | `baselines/craftax`, `loss/policy_gradient*.py`, `math/advantage.py`, `model/embedding.py`, `model/min_gru.py`, `optimizers/fused_muon.py` |
| `PyTorch` | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | `math/pooling.py` |
| `RAE` | [bytetriper/RAE](https://github.com/bytetriper/RAE) @ `a4d18c4` | MIT | `model/ae/rae.py` |
| `REG` | [SwayStar123/REG](https://github.com/SwayStar123/REG) @ `3c51606` | MIT | `model/ae/kl.py` |
| `TinyRecursiveModels` | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | MIT | `baselines/arcagi` |
| `VTP` | [MiniMax-AI/VTP](https://github.com/MiniMax-AI/VTP) @ `5ce1eb6` | MIT | `model/ae/vtp.py` |
| `Wan2.1` | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) @ `9737cba` | Apache-2.0 | `model/ae/wan.py`, `model/ae/video.py`, `model/ae/latent_norm.py` |
| `Wan2.2` | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) @ `1ea34ff` | Apache-2.0 | `model/ae/wan.py`, `model/ae/video.py` |
| `cifar10-airbench` | [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) | MIT | `baselines/cifar10` |
| `ffcv-imagenet` | [libffcv/ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet) | Apache-2.0 | `baselines/imagenet` |
| `flux` | [black-forest-labs/flux](https://github.com/black-forest-labs/flux) @ `802fb47` | Apache-2.0 | `model/ae/kl.py`, `model/ae/latent_norm.py` |
| `flux2` | [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) @ `50fe516` | Apache-2.0 | `model/ae/kl.py`, `model/ae/latent_norm.py` |
| `generative-inverse-heat-dissipation` | [AaltoML/generative-inverse-heat-dissipation](https://github.com/AaltoML/generative-inverse-heat-dissipation) | MIT | `math/frequency.py` |
| `mochi` | [genmoai/mochi](https://github.com/genmoai/mochi) @ `15c9bad` | Apache-2.0 | `model/ae/mochi.py`, `model/ae/video.py`, `model/ae/latent_norm.py` |
| `purejaxql` | [mttga/purejaxql](https://github.com/mttga/purejaxql) | Apache-2.0 | `baselines/craftax` (PQN) |
| `taesd` | [madebyollin/taesd](https://github.com/madebyollin/taesd) @ `fdbae2d` | MIT | `model/ae/taesd.py` |
| `torch-dct` | [zh217/torch-dct](https://github.com/zh217/torch-dct) | MIT | `math/frequency.py` |
| `transformerXL_PPO_JAX` | [Reytuag/transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX) | MIT | `baselines/craftax` (GTrXL) |
| `transformers` | [huggingface/transformers](https://github.com/huggingface/transformers) @ `v5.17.0` | Apache-2.0 | `math/gated_delta_rule.py` |

[karpathy/autoresearch](https://github.com/karpathy/autoresearch), which
`baselines/nanochat` and `optimizers/fused_adamw.py` port, states "MIT" in its
README and ships no license file, so there is no text to reproduce here.

[water-vapor/charm](https://github.com/water-vapor/charm),
whose model and training recipes `baselines/arcagi` re-implements at commit
`728bf26903684ef5e8bcb476cb8e55e42cf35b9d`, publishes no license, so there is
no license text to reproduce here.

[Imbernoulli/MLS-Bench](https://github.com/Imbernoulli/MLS-Bench), whose tasks
`baselines/mlsbench` re-implements at commit `80cf5c5`, publishes no license,
so there is no text to reproduce here either.
