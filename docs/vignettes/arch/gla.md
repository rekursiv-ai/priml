# Gated linear attention

Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", arXiv:2312.06635.

## Notation (one head)

```
xₜ                                 ∈ ℝᶜ        # input
d                                              # head dim
Sₜ                                 ∈ ℝᵈˣᵈ      # state: key → value table
```

## Linear attention

```
Q,K,V                              ∈ ℝᵈˣᶜ      # Linear operators
qₜ    = Q xₜ                       ∈ ℝᵈ        # query
kₜ    = K xₜ                       ∈ ℝᵈ        # key
vₜ    = V xₜ                       ∈ ℝᵈ        # value
Sₜ    = Sₜ₋₁ + kₜ vₜᵀ              ∈ ℝᵈˣᵈ      # store vₜ under kₜ; never forget
yₜ    = Sₜᵀ qₜ = Σₛ₌₁ᵗ (qₜᵀ kₛ) vₛ ∈ ℝᵈ        # attention without softmax
```

## GLA

```
A₁                                 ∈ ℝ¹⁶ˣᶜ     # low-rank gate projection
A₂                                 ∈ ℝᵈˣ¹⁶
αₜ    = σ(A₂ A₁ xₜ)^(1 / τ)          ∈ (0,1)ᵈ    # τ = 16 slows forgetting
Sₜ    = Diag(αₜ) Sₜ₋₁ + kₜ vₜᵀ     ∈ ℝᵈˣᵈ      # fade each key channel, then add
yₜ    = Sₜᵀ qₜ                     ∈ ℝᵈ
```

Linear attention, but: add `Diag(αₜ)`, one input-chosen decay per key channel.
Same idea as minGRU's `1 − uₜ`, applied to a `d × d` table.

## Decay family

| model            | decay on `Sₜ₋₁`  | file                       |
|------------------|------------------|----------------------------|
| linear attention | `1`              |                            |
| RetNet           | fixed scalar `γ` |                            |
| Mamba-2          | scalar `αₜ`      | [mamba.md](mamba.md)       |
| GLA              | `Diag(αₜ)`       |                            |
| + erase at `kₜ`  | see DeltaNet     | [deltanet.md](deltanet.md) |

Paper: FlashLinearAttention is faster than FlashAttention-2 even at 1K;
a GLA layer has ≈ `4c²` params, like softmax attention.
