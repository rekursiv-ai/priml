# MLA (Multi-head Latent Attention)

DeepSeek-AI, "DeepSeek-V2", arXiv:2405.04434, §2.1.

## MLA

```
r                                              # latent dim; 512 in V2
Z                                  ∈ ℝʳˣᶜ      # down-projection
zₜ    = Z xₜ                       ∈ ℝʳ        # cached; replaces kₜ, vₜ of all heads
K⁽ⁱ⁾,V⁽ⁱ⁾                          ∈ ℝᵈˣʳ      # up-projections, head i
kₜ⁽ⁱ⁾ = K⁽ⁱ⁾ zₜ                    ∈ ℝᵈ
vₜ⁽ⁱ⁾ = V⁽ⁱ⁾ zₜ                    ∈ ℝᵈ
```

Multi-head attention ([attention.md](attention.md)), but:

1. cache `zₜ` (`r` numbers), not `kₜ, vₜ` for every head (`2c`).
2. absorb: `qₜ⁽ⁱ⁾ᵀ kₛ⁽ⁱ⁾ = xₜᵀ (Q⁽ⁱ⁾ᵀ K⁽ⁱ⁾) zₛ`; fold `K⁽ⁱ⁾` into `Q⁽ⁱ⁾`,
   `V⁽ⁱ⁾` into `Y`. Decoding attends over `zₛ` directly.
3. RoPE sits between `Q` and `K` and blocks the absorb, so position goes in
   a separate decoupled part:

```
Kᴿ                                 ∈ ℝᵈᴿˣᶜ     # shared across heads
kᴿₜ   = RoPE(Kᴿ xₜ)                ∈ ℝᵈᴿ       # cached too
qᴿₜ⁽ⁱ⁾ = RoPE(Qᴿ⁽ⁱ⁾ xₜ)            ∈ ℝᵈᴿ
scoreₜₛ⁽ⁱ⁾ = qₜ⁽ⁱ⁾ᵀ kₛ⁽ⁱ⁾ + qᴿₜ⁽ⁱ⁾ᵀ kᴿₛ ∈ ℝ
```

- cache per token per layer: `r + dᴿ ≈ 4.5 d`, vs `2 m d` for MHA.
- queries are also compressed (`r′ = 1536`) to save training memory.

V2 vs DeepSeek 67B: KV cache −93.3%, max generation throughput 5.76×.

Used in DeepSeek V2/V3, Kimi K2, and the attention layers of Kimi Linear ([kda.md](kda.md)).
