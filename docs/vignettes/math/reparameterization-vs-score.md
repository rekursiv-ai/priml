# Reparameterization vs Score Function

Both gradient estimators follow from moving `∇_θ` inside an integral:

1. the score function moves it onto the density, and,
2. reparameterization moves the `θ`-dependence into `f` first, then onto `f`.

Let,

  ```
  F(θ)       ≝ 𝔼_{q_θ}[ f(X) ] = ∫ f(x) q_θ(x) dx
  X          ≝ T(ε, θ),   ε ~ p(ε),   so X ~ q_θ            (a reparameterization)
  ĝ_score(X) ≝ f(X) ⋅ ∇_θ log q_θ(X),     X ~ q_θ
  ĝ_rep(ε)   ≝ ∂_θT(ε, θ)ᵀ ⋅ ∇f(T(ε, θ)), ε ~ p
  ```

where `∂_θT` is the `dim(X) × dim(θ)` Jacobian of `T` in `θ`.

First, the score function estimator is unbiased,

  ```
  ∇_θ F(θ) = ∇_θ ∫ f(x) q_θ(x) dx
           = ∫ f(x) ∇_θ q_θ(x) dx
           = ∫ f(x) q_θ(x) ∇_θ log q_θ(x) dx
           = 𝔼_{q_θ}[ ĝ_score(X) ]
  ```

where the second step uses that `f` does not depend on `θ`, and the third
uses `∇q = q ∇log q` where `q > 0`. The score has zero mean, so
`(f(X) − c) ∇log q_θ(X)` is also unbiased for any constant baseline `c`.

Second, the reparameterization estimator is unbiased,

  ```
  ∇_θ F(θ) = ∇_θ 𝔼_{p}[ f(T(ε, θ)) ]
           = 𝔼_{p}[ ∇_θ f(T(ε, θ)) ]
           = 𝔼_{p}[ ∂_θT(ε, θ)ᵀ ∇f(T(ε, θ)) ]
           = 𝔼_{p}[ ĝ_rep(ε) ]
  ```

where the first step uses `X = T(ε, θ) ~ q_θ`, the second swaps `∇` and `𝔼`
(the law `p` does not depend on `θ`), and the third is the chain rule.

Third, the Gaussian example. Let `q_θ = N(θ, σ²)` with `σ` fixed,
`T(ε, θ) = θ + σε`, `ε ~ N(0, 1)`, and `f(x) = x²`. Then `F(θ) = θ² + σ²`
and `∇F = 2θ`. The two estimators, written in `ε`,

  ```
  ∇_θ log q_θ(x) = (x − θ) / σ²
                 = ε / σ
  ĝ_score        = (θ + σε)² ⋅ ε / σ
                 = θ²ε/σ + 2θε² + σε³
  ĝ_rep          = 1 ⋅ 2(θ + σε)
                 = 2θ + 2σε
  ```

Both have mean `2θ`, using `𝔼ε = 𝔼ε³ = 0` and `𝔼ε² = 1`. The
reparameterization variance is `𝕍(2σε) = 4σ²`. For the score,
using `𝔼ε⁴ = 3`, `𝔼ε⁶ = 15`, and that odd moments vanish,

  ```
  𝔼[ ĝ_score² ] = (θ⁴/σ²) 𝔼ε² + 4θ² 𝔼ε⁴ + σ² 𝔼ε⁶ + 2θ² 𝔼ε⁴
                  + (4θ³/σ) 𝔼ε³ + 4θσ 𝔼ε⁵
                = θ⁴/σ² + 12θ² + 15σ² + 6θ² + 0 + 0
                = θ⁴/σ² + 18θ² + 15σ²
  𝕍(ĝ_score)    = θ⁴/σ² + 18θ² + 15σ² − 4θ²
                = θ⁴/σ² + 14θ² + 15σ²
  ```

where the first step expands the square of three terms: three squares, and
cross terms `2 ⋅ (θ²/σ) ⋅ σ ⋅ ε⁴`, `2 ⋅ (θ²/σ) ⋅ 2θ ⋅ ε³`, `2 ⋅ 2θ ⋅ σ ⋅ ε⁵`.

Hence,

  ```
  𝔼[ĝ_score] = 𝔼[ĝ_rep] = ∇_θ 𝔼_{q_θ}[ f(X) ]
  𝕍(ĝ_rep)   = 4σ²
  𝕍(ĝ_score) = θ⁴/σ² + 14θ² + 15σ²   ≥ (15/4) ⋅ 𝕍(ĝ_rep)
  ```

In the example the score estimator is always worse, and its variance blows
up as `σ → 0` or `|θ| → ∞`. The baseline `c = 𝔼f = θ² + σ²` removes the
`θ⁴/σ²` term but not the gap. The ranking is not universal: for `f` with a
large or rough gradient (e.g., near-discontinuous), the reparameterization
variance can exceed the score's, and for discrete `X` no reparameterization
`T` differentiable in `θ` exists.

Regularity: the score estimator needs `q_θ` differentiable in `θ`, support
not depending on `θ`, and `∇` to pass through `∫` (e.g., dominated
`f ∇q_θ`). The reparameterization estimator needs a `T` with `p` free of `θ`,
`f` differentiable (a.e., with `T(ε, θ)` hitting the nondifferentiable set
with probability zero), and `∇` to pass through `𝔼_p`. Both need finite
second moments for the variances to exist.
