# I-Projection

The I-projection follows from one fact: for the tilted reference
`p_η ∝ q ⋅ exp(ηᵀT)` matching the moments, the KL from any feasible `p` to `q`
splits as `𝕂(p‖p_η) + 𝕂(p_η‖q)`, a Pythagorean identity. Since the first term
is nonnegative (gibbs-inequality.md), `p_η` is the closest feasible
distribution to `q`.

Let `q` be a reference density with respect to a base measure `μ`, and,

  ```
  T : 𝒴 → ℝᵈ                         constraint statistic
  τ ∈ ℝᵈ                              target moment
  𝒞 ≝ { p : ∫ p dμ = 1, 𝔼_p[T] = τ }  the linear family
  p_η(y) ≝ q(y) ⋅ exp( ηᵀT(y) − A(η) )
  A(η)   ≝ log ∫ q ⋅ exp( ηᵀT ) dμ
  ```

and suppose some `η` with `A(η) < ∞` has `p_η ∈ 𝒞`. The I-projection of `q`
onto `𝒞` is `argmin_{p∈𝒞} 𝕂(p‖q)`: the feasible distribution least
distinguishable from the reference.

First, the KL to the reference is affine in the moment on `p_η`,

  ```
  𝕂(p_η‖q) = ∫ p_η ⋅ ( ηᵀT − A(η) ) dμ
           = ηᵀτ − A(η)
  ```

Second, for any `p ∈ 𝒞`, write `log(p/q) = log(p/p_η) + log(p_η/q)` and take
`𝔼_p`,

  ```
  𝕂(p‖q) = 𝔼_p[ log(p/p_η) ] + 𝔼_p[ ηᵀT − A(η) ]
         = 𝕂(p‖p_η) + ηᵀτ − A(η)
         = 𝕂(p‖p_η) + 𝕂(p_η‖q)
  ```

where the second step uses `𝔼_p[T] = τ`, the same as under `p_η`.

Hence,

  ```
  𝕂(p‖q) = 𝕂(p‖p_η) + 𝕂(p_η‖q) ≥ 𝕂(p_η‖q)    for every p ∈ 𝒞
  ```

with equality iff `p = p_η`. The I-projection is the exponential tilt of the
reference, and the identity is a Pythagorean theorem: `q`, its projection
`p_η`, and any other feasible `p` form a right triangle under KL.

Examples.

- Maximum entropy. `q` constant on a finite `𝒴` (or `q = 1` against `μ`). Then
  `𝕂(p‖q) = −H(p) + const`, so minimizing KL to the uniform reference is
  maximizing entropy, and `p_η ∝ exp(ηᵀT)` (max-entropy.md) is the special
  case.
- Exponential tilting. `q = 𝒩(0, 1)`, `T(y) = y`, `τ = m`. Then
  `p_η ∝ φ(y) e^{ηy}` completes the square to `𝒩(m, 1)`, with `η = m` and
  `𝕂 = m²/2`. Conditioning a random walk on an atypical mean, large
  deviations (Sanov), and importance-sampling proposals all use this tilt.
- Calibration and reweighting. `q` a model's predictive, `T` the indicators
  of a few events, `τ` their observed frequencies. The I-projection is the
  smallest change to the model's predictive that matches them, a multiplicative
  reweighting `q ⋅ e^{ηᵀT}`: iterative proportional fitting, raking, and
  entropic optimal transport solve this problem.

The dual, the M-projection `argmin_{p∈ℰ} 𝕂(r‖p)` over an exponential family
`ℰ` for data `r`, is maximum likelihood; it matches moments
(`𝔼_p[T] = 𝔼_r[T]`) and satisfies the mirror-image identity
`𝕂(r‖p) = 𝕂(r‖p⋆) + 𝕂(p⋆‖p)` for `p ∈ ℰ`. Alternating the two projections is
EM (em-monotonicity.md).

Regularity: `A(η) < ∞` at the solving `η`; `τ` in the relative interior of
the convex hull of the support of `T` under `q`, so that a solving `η` exists
(on the boundary the projection exists as a limit `η → ∞` with smaller
support); `𝕂(p‖q) < ∞` for the comparison to be informative, which needs
`p ≪ q`. If no `p ∈ 𝒞` has `p ≪ q`, the projection does not exist.
