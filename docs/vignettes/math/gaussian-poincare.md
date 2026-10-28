# Gaussian Poincaré Inequality

The Gaussian Poincaré inequality follows from two facts:

1. rotating two independent Gaussians traces a path from one to the other
   whose velocity is an independent Gaussian, and,
2. Stein's lemma (`stein-lemma.md`) on that velocity turns `f ⋅ velocity`
   into `∇f`.

Stein gives the gradient of a Gaussian expectation; Poincaré bounds the
variance by the gradient.

Let,

  ```
  X, Z ~ N(0, Iₙ)   independent
  f : ℝⁿ → ℝ        differentiable, 𝔼‖∇f(X)‖² < ∞
  W_θ  ≝ X sin θ + Z cos θ         θ ∈ [0, π/2]
  W′_θ ≝ X cos θ − Z sin θ
  ```

For each `θ`, `(W_θ, W′_θ)` is a rotation of `(X, Z)`, so `W_θ` and `W′_θ`
are independent `N(0, Iₙ)`, and `X = W_θ sin θ + W′_θ cos θ`.

First, variance as a product with an independent copy. Since
`𝔼[ f(X) f(Z) ] = (𝔼f)²`,

  ```
  𝕍( f(X) ) = 𝔼[ f(X) ⋅ (f(X) − f(Z)) ]
  ```

Second, the path from `Z` to `X`. `W₀ = Z`, `W_{π/2} = X`, and
`∂_θ W_θ = W′_θ`, so

  ```
  f(X) − f(Z) = ∫₀^{π/2} ∇f(W_θ)ᵀ W′_θ dθ
  ```

Third, Stein in `W′_θ` with `W_θ` fixed. `∇f(W_θ)` depends on `W_θ` only, and
`f(X) = f(W_θ sin θ + W′_θ cos θ)` has gradient `cos θ ⋅ ∇f(X)` in `W′_θ`,

  ```
  𝔼[ f(X) ∇f(W_θ)ᵀ W′_θ ] = 𝔼[ ∇f(W_θ)ᵀ 𝔼[ f(X) W′_θ | W_θ ] ]
                          = cos θ ⋅ 𝔼[ ∇f(W_θ)ᵀ ∇f(X) ]
  ```

Fourth, combine and bound,

  ```
  𝕍( f(X) ) = ∫₀^{π/2} cos θ ⋅ 𝔼[ ∇f(W_θ)ᵀ ∇f(X) ] dθ
            ≤ ∫₀^{π/2} cos θ ⋅ ( 𝔼‖∇f(W_θ)‖² )^{½} ( 𝔼‖∇f(X)‖² )^{½} dθ
            = 𝔼‖∇f(X)‖² ⋅ ∫₀^{π/2} cos θ dθ
            = 𝔼‖∇f(X)‖²
  ```

where the first step swaps `𝔼` and `∫ dθ` (Fubini), the second is
Cauchy–Schwarz (`cauchy-schwarz-holder-young.md`), and the third uses
`W_θ ~ X`.

Hence,

  ```
  𝕍( f(X) ) ≤ 𝔼[ ‖∇f(X)‖² ]                X ~ N(0, Iₙ)
  𝕍( f(X) ) ≤ 𝔼[ ∇f(X)ᵀ Σ ∇f(X) ]          X ~ N(μ, Σ)
  ```

where the second follows from the first applied to `w ↦ f(μ + Aw)`,
`AAᵀ = Σ`.

Equality holds for linear `f(x) = aᵀx`: `𝕍(aᵀX) = ‖a‖² = 𝔼‖∇f‖²`. The
constant 1 does not depend on `n`, unlike bounds that sum per-coordinate
ranges.

Fine print: the same proof with `f(X)` times `g(X) − g(Z)` gives the
covariance identity `Cov(f(X), g(X)) = ∫₀^{π/2} cos θ ⋅ 𝔼[ ∇g(W_θ)ᵀ ∇f(X) ] dθ`.
The one-dimensional inequality is Chernoff (Ann. Probab. 1981, cited).
Efron–Stein (`efron-stein.md`) bounds variance by per-coordinate
fluctuations for any independent coordinates; Poincaré uses Gaussianity to
bound it by the gradient. Gaussian concentration (`gaussian-lipschitz.md`)
is the tail-bound counterpart for Lipschitz `f`.

Regularity: the proof needs `f` differentiable (locally Lipschitz suffices,
a.e.), `𝔼‖∇f(X)‖² < ∞` (so Cauchy–Schwarz and Fubini apply, and
`𝔼 f(X)² < ∞` follows), and Stein's conditions for `w′ ↦ f(w sin θ + w′ cos θ)`
for almost every `w`.
