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
| `REG` | [SwayStar123/REG](https://github.com/SwayStar123/REG) @ `3c51606` | MIT | `model/ae/invae.py` |
| `TinyRecursiveModels` | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | MIT | `baselines/arcagi` |
| `VTP` | [MiniMax-AI/VTP](https://github.com/MiniMax-AI/VTP) @ `5ce1eb6` | MIT | `model/ae/vtp.py` |
| `cifar10-airbench` | [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) | MIT | `baselines/cifar10` |
| `ffcv-imagenet` | [libffcv/ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet) | Apache-2.0 | `baselines/imagenet` |
| `generative-inverse-heat-dissipation` | [AaltoML/generative-inverse-heat-dissipation](https://github.com/AaltoML/generative-inverse-heat-dissipation) | MIT | `math/frequency.py` |
| `purejaxql` | [mttga/purejaxql](https://github.com/mttga/purejaxql) | Apache-2.0 | `baselines/craftax` (PQN) |
| `torch-dct` | [zh217/torch-dct](https://github.com/zh217/torch-dct) | MIT | `math/frequency.py` |
| `transformerXL_PPO_JAX` | [Reytuag/transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX) | MIT | `baselines/craftax` (GTrXL) |
| `transformers` | [huggingface/transformers](https://github.com/huggingface/transformers) @ `v5.17.0` | Apache-2.0 | `math/gated_delta_rule.py` |

[karpathy/autoresearch](https://github.com/karpathy/autoresearch), which
`baselines/nanochat` and `optimizers/fused_adamw.py` port, states "MIT" in its
README and ships no license file, so there is no text to reproduce here.

[Imbernoulli/MLS-Bench](https://github.com/Imbernoulli/MLS-Bench), whose tasks
`baselines/mlsbench` re-implements at commit `80cf5c5`, publishes no license,
so there is no text to reproduce here either.
