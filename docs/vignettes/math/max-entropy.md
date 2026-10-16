# Maximum Entropy

The maximum-entropy distribution follows from one fact: the entropy gap
between the exponential family and any feasible `p` is a KL, which is
nonnegative (gibbs-inequality.md).

Let `Y` be discrete with finite support `𝒴`, and,

  ```
  T : 𝒴 → ℝᵈ                         sufficient statistic
  τ ∈ ℝᵈ                              target moment
  H(p)   ≝ −Σ_y p(y) ⋅ log p(y)
  p_η(y) ≝ exp( ηᵀT(y) − A(η) )
  A(η)   ≝ log Σ_y exp( ηᵀT(y) )
  ```

and suppose some `η` satisfies `𝔼_{p_η}[T] = τ`. The problem is
`max_p H(p)` subject to `Σ_y p(y) = 1`, `p ≥ 0`, and `𝔼_p[T] = τ`.

First, stationarity identifies the candidate. The Lagrangian with
multipliers `η` (moments) and `λ` (normalization) is
`H(p) + ηᵀ(𝔼_p[T] − τ) + λ(Σ_y p(y) − 1)`. Setting its partial in `p(y)` to
zero,

  ```
  0    = −log p(y) − 1 + ηᵀT(y) + λ
  p(y) = exp( ηᵀT(y) + λ − 1 )
       ∝ exp( ηᵀT(y) )
  ```

so normalization forces `p = p_η`. Stationarity alone does not prove a
maximum; the next two steps do.

Second, the entropy of `p_η` is affine in the moment,

  ```
  H(p_η) = −Σ_y p_η(y) ⋅ ( ηᵀT(y) − A(η) )
         = −ηᵀ 𝔼_{p_η}[T] + A(η)
         = −ηᵀ τ + A(η)
  ```

Third, for any feasible `p`, the cross-entropy against `p_η` depends on `p`
only through `𝔼_p[T] = τ`,

  ```
  H(p_η) − H(p) = −ηᵀτ + A(η) + Σ_y p(y) ⋅ log p(y)
                = −Σ_y p(y) ⋅ ( ηᵀT(y) − A(η) ) + Σ_y p(y) ⋅ log p(y)
                = −Σ_y p(y) ⋅ log p_η(y) + Σ_y p(y) ⋅ log p(y)
                = Σ_y p(y) ⋅ log( p(y) / p_η(y) )
                = 𝕂(p‖p_η)
                ≥ 0
  ```

where the second step uses `Σ_y p(y) ⋅ ηᵀT(y) = ηᵀτ` and `Σ_y p(y) = 1`,
and the last step is Gibbs.

Hence,

  ```
  H(p) ≤ H(p_η) = A(η) − ηᵀτ    for every p with 𝔼_p[T] = τ
  ```

Equality holds iff `𝕂(p‖p_η) = 0`, i.e., `p = p_η`. So the maximizer is
unique.

Softmax is the special case `T(y) = f(y) ∈ ℝ` a score per class with one
moment constraint. With `η` the inverse temperature,
`p_η(y) = exp( η f(y) ) / Σ_k exp( η f(k) )`, so `softmax(η f)` is the
maximum-entropy distribution with expected score `τ`. With `T(y) = e_y`
(one-hot), `p_η = softmax(η)` and `𝔼[T] = τ` pins `p = τ` directly.

Regularity: finite `𝒴` (for countable or continuous `Y`, replace sums by
integrals against a base measure and require `A(η) < ∞`); `τ` in the range
of `η ↦ 𝔼_{p_η}[T]`, i.e., in the relative interior of the convex hull of
`T(𝒴)`. If `τ` is on the boundary, the maximizer exists but lies at a limit
`η → ∞` and has smaller support. Since `p_η > 0` everywhere,
`𝕂(p‖p_η)` is finite for every feasible `p`.
