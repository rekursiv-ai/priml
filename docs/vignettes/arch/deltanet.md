# Gated DeltaNet

Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", arXiv:2412.06464.

## Problem

GLA writes by adding. Writing `v′` under a key that already holds `v`
reads back `v + v′`, not `v′` ([gla.md](gla.md)).

## DeltaNet

Notation (`xₜ`, `d`, `qₜ`, `kₜ`, `vₜ`, `Sₜ`) as in [gla.md](gla.md).

```
kₜ                                 ∈ ℝᵈ        # L2-normalized: ‖kₜ‖ = 1
B                                  ∈ ℝ¹ˣᶜ      # Linear operator
βₜ    = σ(B xₜ)                    ∈ (0,1)     # write strength
Sₜ    = (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ ∈ ℝᵈˣᵈ # erase at kₜ, then write
      = Sₜ₋₁ + βₜ kₜ (vₜ − Sₜ₋₁ᵀ kₜ)ᵀ           # move the stored value toward vₜ
yₜ    = Sₜᵀ qₜ                     ∈ ℝᵈ
```

- `βₜ = 1`: `Sₜᵀ kₜ = vₜ` exactly; the old value is gone.
- One SGD step on `½ ‖Sᵀ kₜ − vₜ‖²` with learning rate `βₜ`.

## Gated DeltaNet

```
αₜ                                 ∈ (0,1)     # decay: one scalar per head
Sₜ    = αₜ (I − βₜ kₜ kₜᵀ) Sₜ₋₁ + βₜ kₜ vₜᵀ ∈ ℝᵈˣᵈ
```

DeltaNet, but: add `αₜ`. `αₜ → 0` clears all memory; `αₜ → 1` is pure DeltaNet.
Mamba-2, but: add the erase `(I − βₜ kₜ kₜᵀ)`.

- q, k, v: Linear → short conv → SiLU; L2-normalize q, k.
- training: chunked, WY form for the product of `(I − β k kᵀ)` terms.
- throughput: same as DeltaNet; both slightly below Mamba-2 (§4).
- `βₜ ∈ (0,2)` gives negative eigenvalues, which helps state tracking.

## Used in

| model          | layers                                     |
|----------------|--------------------------------------------|
| Qwen3-Next 80B | 12 × [3 Gated DeltaNet, 1 gated attention] |
| Qwen3.5 397B   | 15 × [3 Gated DeltaNet, 1 gated attention] |

Per-channel `αₜ` instead of scalar: [kda.md](kda.md).
