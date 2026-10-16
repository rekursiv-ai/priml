# minGRU

Feng et al., "Were RNNs All We Needed?", arXiv:2410.01201v1, §3 and App. B.

min/GRU is an input-gated EMA.

```
EMA      hₜ = (1 − α)  ⊙ hₜ₋₁ + α  ⊙ xₜ   # α fixed
min/GRU  hₜ = (1 − uₜ) ⊙ hₜ₋₁ + uₜ ⊙ h̃ₜ   # uₜ, h̃ₜ depends on min/GRU.
```


## Notation

```
t                    ∈ 1..n                    # time step; n = sequence length
xₜ                   ∈ ℝᶜ                      # input
hₜ                   ∈ ℝᵇ                      # memory; h₀ given (usually 0)
σ(s)  = (1 + e⁻ˢ)⁻¹  ∈ (0,1)                   # elementwise
⊙                                              # elementwise product
[·,·]                                          # concat
```

## GRU (sequential)

Mathematically,

```
U,R,H                              ∈ ℝᵇˣ⁽ᶜ⁺ᵇ⁾  # Linear operators; biases omitted
uₜ    =    σ(U [xₜ, hₜ₋₁     ])    ∈ (0,1)ᵇ    # update gate, needs hₜ₋₁
rₜ    =    σ(R [xₜ, hₜ₋₁     ])    ∈ (0,1)ᵇ    # reset gate: share of hₜ₋₁ in h̃ₜ
h̃ₜ    = tanh(H [xₜ, hₜ₋₁ ⊙ rₜ])    ∈ (−1,1)ᵇ   # candidate, needs hₜ₋₁
hₜ    = (1 − uₜ) ⊙ hₜ₋₁ + uₜ ⊙ h̃ₜ  ∈ ℝᵇ        # n steps in order
```

## minGRU (parallel)


Its claim-to-fame is that the gate reads only xₜ, never hₜ₋₁, so all of
training runs in parallel.

Mathematically,

```
U,H                                ∈ ℝᵇˣᶜ      # GRU's U, H minus the hₜ₋₁ columns
uₜ    = σ(U xₜ)                    ∈ (0,1)ᵇ    # all t at once
h̃ₜ    = g(H xₜ)                    ∈ (0,∞)ᵇ    # all t at once
hₜ    = (1 − uₜ) ⊙ hₜ₋₁ + uₜ ⊙ h̃ₜ  ∈ ℝᵇ        # same update as GRU
```

Rewriting shows its a dynamically weighted projection residual,

```
hₜ    = hₜ₋₁ + σ(U xₜ) ⊙ (g(H xₜ) − hₜ₋₁)
             "dynamic"   "projection residual"
```

The activation `g(s)` is,

```
g(s)  = relu(s) + σ(−relu(−s))     ∈ (0,∞)     # keeps h̃ₜ > 0 for log-space scan
      = s + ½ if s ≥ 0 else σ(s)
```

Note that `g` is `softplus`-like but with a kink at 0 so its slope is exactly 1
when `s ≥ 0`. To our knowledge it is unknown if simply using softplus works as
well, but ostensibly not.

The advantage of minGRU is that `hₜ` only needs a single scan to accumulate
products and sums:

```
hₜ  =   ∏ₛ₌₁ᵗ (1 − uₛ) ⊙ h₀
      + Σₛ₌₁ᵗ ∏ᵣ₌ₛ₊₁ᵗ (1 − uᵣ) ⊙ uₛ ⊙ h̃ₛ  ∈ ℝᵇ     # cumprod + cumsum: log n steps
```

Computed in log space so products of many `1 − uᵣ` don't underflow
(App. B.1, B.2.1); this is why `h̃ₜ > 0`:

```
log(1 − uₜ)   = −softplus( U xₜ)                                   # since 1 − σ(s) = σ(−s)
log(uₜ ⊙ h̃ₜ)  = −softplus(−U xₜ) + log g(H xₜ)
log g(s)      = log(s + ½) if s ≥ 0 else −softplus(−s)
log γₜ        = Σₛ₌₁ᵗ log(1 − uₛ)                                  # cumsum; γ₀ = 1
log hₜ        = log γₜ + logcumsumexpₛ₌₀ᵗ (log(uₛ ⊙ h̃ₛ) − log γₛ)  # s = 0 term: log h₀
```

GRU, but:

1. `uₜ` and `h̃ₜ` read `xₜ` only, not `hₜ₋₁`.
2. del `rₜ` since there is no `hₜ₋₁` left in `h̃ₜ` to reset.
3. del `tanh`; `g` only keeps `h̃ₜ > 0` (main text: `h̃ₜ = H xₜ`, App. B.2.1: `g`).

## Stacking (restores dependence on the past)

One layer's `uₜ` sees only `xₜ`. Feeding layer `l`'s output to layer
`l+1` makes `uₜ` of layer 2+ see a summary of `x₁..xₜ`.

```
l                ∈ 1..ℓ      # layer; ℓ = depth
x⁽¹  ⁾ₜ = xₜ     ∈ ℝᶜ
x⁽ˡ⁺¹⁾ₜ = h⁽ˡ⁾ₜ  ∈ ℝᵇ        # layer l+1's input is layer l's output
```

Selective copy: `ℓ = 1` scores 37.0%, `ℓ = 3` scores 99.5% (Table 1).

## Training (why no BPTT)

Backward is the same scan reversed, since the per-step Jacobian is known
once forward is done.

```
∂hₜ₊₁/∂hₜ = diag(1 − uₜ₊₁)                     ∈ ℝᵇˣᵇ      # known after forward; reverse scan, log n steps
```

Sequential steps per training step:

| model  | steps       | why                                                      |
|--------|-------------|----------------------------------------------------------|
| GRU    | `ℓ · n`     | each `hₜ` waits for `hₜ₋₁` (BPTT), matmuls included      |
| minGRU | `ℓ · log n` | layers wait for each other; time does not                |
| Puffer | `ℓ · n`     | elementwise loop over `t`; matmuls hoisted out (below)   |

At `n = 512`: 175× faster per training step than GRU (§4.1).

## PufferLib's minGRU

PufferLib's minGRU: [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib)
`5.0` @ `ea7f989` (2026-10-14), `src/algo.cu:217-244`, `src/puffercpu.c:478-506`.
Blog ([4.0](https://puffer.ai/blog/4.0/)): "We use highway connections
instead of traditional normalization between layers."

```
U,H,K                              ∈ ℝᵇˣᵇ      # one bias-free matmul → 3b; Kaiming init
uₜ    = σ(U xₜ)                    ∈ (0,1)ᵇ    # [Same]
h̃ₜ    = g(H xₜ)                    ∈ (0,∞)ᵇ    # [Same]
hₜ    = (1 − uₜ) ⊙ hₜ₋₁ + uₜ ⊙ h̃ₜ  ∈ ℝᵇ        # [Same]
kₜ    = σ(K xₜ)                    ∈ (0,1)ᵇ    # highway gate
yₜ    = kₜ ⊙ hₜ + (1 − kₜ) ⊙ xₜ    ∈ ℝᵇ        # layer output; next layer's input
```

minGRU, but:

1. add highway: layer emits `yₜ`, not `hₜ`; `kₜ → 0` passes `xₜ` through.
   Replaces norm between layers; stacking becomes `x⁽ˡ⁺¹⁾ₜ = y⁽ˡ⁾ₜ`.
2. square layers (`c = b`): a Linear encoder maps obs to `ℝᵇ` first.
   Net: `Linear → ℓ × minGRU → Linear` (value head fused in decoder).
3. episode reset: `hₜ₋₁ ← 0` if episode ended before step `t`.
4. training loops over `t` (`ℓ · n` steps), not the log-space scan;
   `g` is kept anyway. Fast because `[U;H;K] xₜ` runs once for all `t`
   before the loop (`algo.cu:857`); only elementwise updates are serial
   (`algo.cu:296`, `:364`). GRU can't hoist: its matmul reads `hₜ₋₁`.
