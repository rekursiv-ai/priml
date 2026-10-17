# Test-time training memory

The memory is a small network `Mₜ`, trained online: one gradient step per
token on how badly it maps `kₜ` to `vₜ`. GLA's `Sₜ` is the case where the network
is linear and the step is plain addition ([memory.md](memory.md)).

## Notation (one head)

```
xₜ                                 ∈ ℝᶜ        # input
d                                              # head dim
Q,K,V                              ∈ ℝᵈˣᶜ      # Linear operators
qₜ    = Q xₜ                       ∈ ℝᵈ        # query
kₜ    = K xₜ                       ∈ ℝᵈ        # key
vₜ    = V xₜ                       ∈ ℝᵈ        # value
Mₜ    : ℝᵈ → ℝᵈ                                # memory: a network with weights Wₜ
ℓₜ(W) = ‖M(kₜ; W) − vₜ‖₂²          ∈ ℝ≥0       # associative loss at step t
```

## TTT

Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", arXiv:2407.04620.

```
θₜ                                 ∈ ℝ>0       # inner learning rate, from xₜ
Wₜ    = Wₜ₋₁ − θₜ ∇ℓₜ(Wₜ₋₁)                     # one SGD step per token
yₜ    = M(qₜ; Wₜ)                  ∈ ℝᵈ        # read: run the memory on qₜ
```

- TTT-Linear: `M(k; W) = W k`. The step is the delta rule ([deltanet.md](deltanet.md)).
- TTT-MLP: `M` is a two-layer MLP; holds more than `d` pairs, costs more I/O.
- Trained in chunks: one gradient per chunk, taken at the chunk's start weights.

## Titans

Behrouz, Zhong, Mirrokni, "Titans: Learning to Memorize at Test Time", arXiv:2501.00663, §3.1.

```
Sₜ    = ηₜ Sₜ₋₁ − θₜ ∇ℓₜ(Mₜ₋₁)                  # surprise with momentum
Mₜ    = (1 − αₜ) Mₜ₋₁ + Sₜ                      # fade, then write
ηₜ, θₜ, αₜ                         ∈ (0,1)     # from xₜ: carry, step, forget
yₜ    = Mₜ(qₜ)                     ∈ ℝᵈ
```

TTT, but:

1. add momentum `ηₜ Sₜ₋₁`: a surprising event keeps writing for a while ("past surprise").
2. add forget `αₜ`: weight decay on the memory, as GLA's `αₜ` fades `Sₜ`.
3. used beside attention: as context (MAC), as a gate (MAG), or as a layer (MAL).

## Against the operator memories

| memory | `M` | write | file |
|--------|-----|-------|------|
| linear attention | `S k`, linear | add `k vᵀ` | [gla.md](gla.md) |
| DeltaNet | `S k`, linear | one SGD step | [deltanet.md](deltanet.md) |
| TTT-Linear | `W k`, linear | one SGD step | |
| TTT-MLP | 2-layer MLP | one SGD step | |
| Titans | MLP | SGD + momentum + decay | |

Same family as [memory.md](memory.md), with the operator replaced by a network.
Capacity is no longer bounded by `d` pairs; what it holds depends on the network.
