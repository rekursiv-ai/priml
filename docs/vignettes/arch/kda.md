# KDA (Kimi Delta Attention)

Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv:2510.26692.

## KDA

```
xₜ                                 ∈ ℝᶜ        # input
d                                              # head dim
Q,K,V                              ∈ ℝᵈˣᶜ      # Linear operators
qₜ    = Q xₜ                       ∈ ℝᵈ        # query
kₜ    = K xₜ / ‖K xₜ‖₂             ∈ ℝᵈ        # key, L2-normalized: ‖kₜ‖₂ = 1
vₜ    = V xₜ                       ∈ ℝᵈ        # value
Sₜ                                 ∈ ℝᵈˣᵈ      # state: key → value table
B                                  ∈ ℝ¹ˣᶜ      # Linear operator
βₜ    = σ(B xₜ)                    ∈ (0,1)     # write strength
αₜ                                 ∈ (0,1)ᵈ    # decay: one per key channel
Sₜ    = (I − βₜ kₜ kₜᵀ) Diag(αₜ) Sₜ₋₁ + βₜ kₜ vₜᵀ ∈ ℝᵈˣᵈ # paper eq. 1
yₜ    = Sₜᵀ qₜ                     ∈ ℝᵈ
```

Gated DeltaNet ([deltanet.md](deltanet.md)), but:

1. `αₜ` scalar → vector, as in GLA ([gla.md](gla.md)).
2. order matters now: fade, then erase at `kₜ`. A scalar commutes; a vector does not.
3. chunked kernel uses a restricted Diagonal-Plus-Low-Rank form.
4. output: head-wise RMSNorm and an input-dependent gate before the output projection.

## Kimi Linear

- 48B total, 3B active MoE; 1.4T tokens.
- layers: 3 KDA : 1 MLA ([mla.md](mla.md)); MLA layers use no RoPE.
- ratio ablation: 3:1 best; 7:1 worse on validation.

vs full-MLA twin, same recipe (paper, vendor-measured):

| measure                  | Kimi Linear            |
|--------------------------|------------------------|
| KV cache                 | up to −75%             |
| decode at 1M context     | 6× faster              |
| prefill at 4k–16k        | ≈ MLA                  |
| prefill at ≥ 128k        | "significantly faster" |
| compute-optimal training | ≈ 1.16× efficiency     |

## Used in

- Kimi K3 (Moonshot, 2.8T, July 2026): KDA at 3:1.
- Ling-3.0-flash (124B / 5.1B active): 5:1 KDA : MLA, per its model card.
- PufferLib: recurrence-only toy vs minGRU, `tests/kda_toy.cu`.

## Gated DeltaNet-2

Hatamizadeh, Choi, Kautz, arXiv:2605.22791, eq. 10.

```
εₜ                                 ∈ [0,1]ᵈ    # erase gate, per key channel (paper: bₜ)
ωₜ                                 ∈ [0,1]ᵈ    # write gate, per value channel (paper: wₜ)
Sₜ    = (I − kₜ (εₜ ⊙ kₜ)ᵀ) Diag(αₜ) Sₜ₋₁ + kₜ (ωₜ ⊙ vₜ)ᵀ ∈ ℝᵈˣᵈ
```

KDA, but:

1. scalar `βₜ` → two vectors: `εₜ` (what to erase) and `ωₜ` (what to write).
2. `εₜ = ωₜ = βₜ 1` is KDA exactly; scalar `αₜ` too is Gated DeltaNet.

- 1.3B, 100B tokens: best overall vs Mamba-2, Gated DeltaNet, KDA, Mamba-3;
  largest gains on RULER multi-key retrieval.
- hybrid training throughput: 38.0 → 36.1 Kt/s as length grows; slightly below KDA.
- not yet in a released production model.

## Complex KDA

Siems et al., arXiv:2609.24797.

KDA, but: widen two ranges.

```
αₜ                                 ∈ [−1,1]ᵈ   # was (0,1)ᵈ: gate can flip sign
βₜ                                 ∈ [0,2]     # was (0,1): erase can reflect
```

- flip plus reflection is a rotation: the state can rotate, not just shrink.
- same cost and stability as KDA (transition stays diagonal plus rank one).
- tracks groups (e.g. `S₃`, `S₄`) that KDA can't; LM quality ≈ KDA.
