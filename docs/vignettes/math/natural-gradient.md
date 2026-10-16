# Natural Gradient

The natural gradient follows from two facts:

1. to second order, `𝕂(p_θ ‖ p_{θ+δ}) = ½ δᵀFδ`, because the score has
   zero mean and the Fisher is the expected negative Hessian, and,
2. a linear objective over an ellipsoid is minimized along `−F⁻¹∇L`.

Let,

  ```
  p_θ(y) a density, θ ∈ ℝᵈ,   L(θ) a differentiable loss,   g ≝ ∇L(θ) ≠ 0
  s(Y) ≝ ∇_θ log p_θ(Y)
  F    ≝ 𝔼_{p_θ}[ s(Y) s(Y)ᵀ ] ≻ 0
  K(δ) ≝ 𝕂(p_θ ‖ p_{θ+δ}) = 𝔼_{p_θ}[ log p_θ(Y) − log p_{θ+δ}(Y) ]
  ```

First, the score is zero mean, since densities integrate to one,

  ```
  𝔼_{p_θ}[ s(Y) ] = ∫ p_θ(y) ⋅ ∇ log p_θ(y) dy
                  = ∫ ∇ p_θ(y) dy
                  = ∇ ∫ p_θ(y) dy
                  = ∇ 1
                  = 0
  ```

Second, the Fisher is the expected negative Hessian of the log-likelihood.
Differentiate `∇ log p = ∇p / p` once more,

  ```
  −𝔼_{p_θ}[ ∇² log p_θ(Y) ] = −𝔼_{p_θ}[ ∇²p_θ(Y)/p_θ(Y) − ∇p_θ(Y) ∇p_θ(Y)ᵀ/p_θ(Y)² ]
                            = −∫ ∇²p_θ(y) dy + 𝔼_{p_θ}[ s(Y) s(Y)ᵀ ]
                            = −∇² ∫ p_θ(y) dy + F
                            = F
  ```

Third, Taylor-expand `K` at `δ = 0`. Only the second term of `K` depends on
`δ`, so

  ```
  K(0)       = 0
  ∇K(0)      = −𝔼_{p_θ}[ s(Y) ] = 0
  ∇²K(0)     = −𝔼_{p_θ}[ ∇² log p_θ(Y) ] = F
  K(δ)       = ½ δᵀFδ + o(‖δ‖²)
  ```

using the first step for the gradient and the second for the Hessian.

Fourth, solve the problem with the quadratic constraint. Substitute
`w ≝ F^{1/2} δ` and `h ≝ F^{−1/2} g`, so `⟨g, δ⟩ = ⟨h, w⟩` and the constraint
`½ δᵀFδ ≤ ε` becomes `‖w‖ ≤ √(2ε)`. By Cauchy–Schwarz,

  ```
  ⟨g, δ⟩ = ⟨h, w⟩
         ≥ −‖h‖ ⋅ ‖w‖
         ≥ −√(2ε) ⋅ ‖h‖
         = −√(2ε) ⋅ √(gᵀF⁻¹g)
  ```

with equality iff `w = −√(2ε) ⋅ h/‖h‖`, i.e.,

  ```
  δ* = −√(2ε) ⋅ F⁻¹g / √(gᵀF⁻¹g)
  ```

Fifth, the true KL constraint gives the same direction as `ε → 0`. Restrict
to a ball `‖δ‖ ≤ r`. Fix `τ ∈ (0, 1)`. Because `F ⪰ λ_min I` with
`λ_min > 0`, the `o(‖δ‖²)` remainder is at most `τ ⋅ ½ δᵀFδ` once `r` is
small. So inside the ball,

  ```
  (1 − τ) ⋅ ½ δᵀFδ ≤ K(δ) ≤ (1 + τ) ⋅ ½ δᵀFδ
  ```

and, for `ε` small enough that both ellipsoids below lie in the ball,

  ```
  { ½ δᵀFδ ≤ ε/(1+τ) } ⊆ { K(δ) ≤ ε } ⊆ { ½ δᵀFδ ≤ ε/(1−τ) }
  ```

Let `δ_ε` be any feasible point at least as good as the inner ellipsoid's
optimum, e.g., the true minimizer. Normalize by the outer radius,
`w ≝ F^{1/2} δ_ε / √(2ε/(1−τ))`, so `‖w‖ ≤ 1`, and let `ĥ ≝ h/‖h‖`. The
fourth step on the inner ellipsoid gives
`⟨g, δ_ε⟩ ≤ −√(2ε/(1+τ)) ⋅ ‖h‖`. Dividing by the outer radius,
`⟨ĥ, w⟩ ≤ −c` with `c ≝ √((1−τ)/(1+τ))`. Then,

  ```
  ‖w + ĥ‖² = ‖w‖² + 2⟨ĥ, w⟩ + 1
           ≤ 1 − 2c + 1
           = 2 ⋅ (1 − c)
  ```

which tends to 0 as `τ → 0`. So the direction of `F^{1/2} δ_ε` tends to
`−ĥ`, i.e., the direction of `δ_ε` tends to that of `−F⁻¹g`.

Hence,

  ```
  argmin_δ { ⟨∇L, δ⟩ : 𝕂(p_θ ‖ p_{θ+δ}) ≤ ε }  ∝  −F⁻¹∇L     as ε → 0
  ```

Sixth, the step is invariant to reparameterization. Let `θ ≝ φ(ϑ)` with `φ`
a diffeomorphism and Jacobian `J ≝ ∂θ/∂ϑ`. By the chain rule the gradient
and the score both pick up `Jᵀ`, so `∇_ϑ L = Jᵀ g` and `F_ϑ = Jᵀ F J`. The
natural step in `ϑ`, mapped back to `θ` to first order, is

  ```
  J ⋅ Δϑ = −α ⋅ J ⋅ F_ϑ⁻¹ ⋅ ∇_ϑ L
         = −α ⋅ J ⋅ J⁻¹ F⁻¹ J⁻ᵀ ⋅ Jᵀ g
         = −α ⋅ F⁻¹ g
  ```

which is the natural step in `θ`. The plain gradient step instead maps to
`J ⋅ Δϑ = −α ⋅ J Jᵀ g`, which depends on `φ` unless `J Jᵀ = I`.

The invariance is exact to first order in `α`, and exact for the
continuous-time flow `θ̇ = −F⁻¹∇L`. It holds because the constraint
`𝕂(p_θ ‖ p_{θ+δ})` compares distributions, not parameters.

Regularity: the proof needs `p_θ` twice differentiable in `θ` with `∇` and
`∇²` passing through `∫`, `F ≻ 0`, `∇L ≠ 0`, the search restricted to a
neighborhood of `θ` (KL may be small far away for non-identifiable models),
and `φ` a diffeomorphism so `J` is invertible.
