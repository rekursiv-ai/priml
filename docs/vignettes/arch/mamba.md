# Mamba

Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752.
Dao & Gu, "Transformers are SSMs" (Mamba-2), arXiv:2405.21060.

## Notation

```
xₜ                                 ∈ ℝᶜ        # input
p                                              # state size per channel
Sₜ                                             # state; capital but not a weight
softplus(s) = log(1 + eˢ)          ∈ (0,∞)     # elementwise
```

## Mamba (S6)

```
A                                  ∈ ℝᵖˣᶜ      # learned, A < 0; not input-dependent
Δ                                  ∈ ℝᶜˣᶜ      # step-size projection
K,Q                                ∈ ℝᵖˣᶜ      # paper: B, C
δₜ    = softplus(Δ xₜ)             ∈ (0,∞)ᶜ    # step size, per channel
kₜ    = K xₜ                       ∈ ℝᵖ        # paper: Bₜ
qₜ    = Q xₜ                       ∈ ℝᵖ        # paper: Cₜ
αₜ    = exp(A Diag(δₜ))            ∈ (0,1)ᵖˣᶜ  # decay
Sₜ    = αₜ ⊙ Sₜ₋₁ + kₜ (δₜ ⊙ xₜ)ᵀ  ∈ ℝᵖˣᶜ      # p numbers of memory per channel
yₜ    = Sₜᵀ qₜ                     ∈ ℝᶜ        # read
```

Per channel with `p = 1`: `sₜ = e^(−δₜ|a|) sₜ₋₁ + δₜ xₜ`. Small `δₜ`: keep ≈ 1,
write ≈ 0; large `δₜ`: forget and write. That is minGRU's input-chosen
rate ([mingru.md](mingru.md)), with keep and write not forced to sum to 1.

## Mamba-2 (SSD)

```
d                                              # head dim (64 or 128); c = m · d
vₜ                                 ∈ ℝᵈ        # one head's slice of xₜ
δₜ                                 ∈ (0,∞)     # step size: one per head
a                                  ∈ ℝ         # learned, a < 0; one per head
αₜ    = exp(δₜ a)                  ∈ (0,1)     # decay: one scalar per head
Sₜ    = αₜ Sₜ₋₁ + kₜ (δₜ vₜ)ᵀ      ∈ ℝᵖˣᵈ
yₜ    = Sₜᵀ qₜ                     ∈ ℝᵈ
```

Mamba, but:

1. decay is a scalar per head, not per channel × state.
2. head dim `d = 64–128`, not 1, so the chunked scan is matmuls.
3. this is linear attention with a scalar input-dependent decay ([gla.md](gla.md)).

Paper: SSD is 2–8× faster than Mamba's selective scan with 8× larger state;
vs FlashAttention-2 it crosses over at 2K and is 6× faster at 16K.

Blocks above omit the short conv, SiLU gate branch and `D` skip around the SSM.

Used in Nemotron 3: Nano 6 attention layers in 52; Super 8 in 88.
