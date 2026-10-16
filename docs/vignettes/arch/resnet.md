# Highway connections

Srivastava, Greff, Schmidhuber, "Highway Networks", arXiv:1505.00387.
He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385.

## Notation

```
x                    ∈ ℝᵇ                      # layer input
f(x)                 ∈ ℝᵇ                      # layer transform, e.g. Linear → nonlinearity
σ(s)  = (1 + e⁻ˢ)⁻¹  ∈ (0,1)                   # elementwise
⊙                                              # elementwise product
```

## Highway

```
K                                  ∈ ℝᵇˣᵇ      # Linear operator
k     = σ(K x)                     ∈ (0,1)ᵇ    # transform gate
y     = k ⊙ f(x) + (1 − k) ⊙ x     ∈ ℝᵇ        # per channel: transform or carry
```

- `k → 0` copies `x` through, so gradients reach early layers.
- Bias `K` toward carry at init (paper: gate bias −1 to −3).

## ResNet

```
y     = f(x) + x                   ∈ ℝᵇ        # no gate
```

Highway, but:

1. del `K`; transform and carry weights both fixed at 1, not `k` and `1 − k`.
2. so `y` is not a convex mix: `‖y‖` can grow with depth.

He et al., arXiv:1603.05027 §3.1: gating the shortcut trained worse
than plain identity.

## Related

| model                        | gate `k`        | transform `f(x)` | carry `x` |
|------------------------------|-----------------|------------------|-----------|
| ResNet                       | none (`1`, `1`) | `f(x)`           | `x`       |
| Highway                      | `σ(K x)`        | `f(x)`           | `x`       |
| GRU / minGRU, over time      | `uₜ`            | `h̃ₜ`             | `hₜ₋₁`    |
| Puffer minGRU, across layers | `kₜ = σ(K xₜ)`  | `hₜ`             | `xₜ`      |

GRU's update is a highway along time; Puffer adds one along depth
([mingru.md](mingru.md), Puffer).
