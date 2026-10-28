# Bonnet and Price

The Gaussian gradient identities follow from two facts:

1. the score-function gradient moves `∇` onto `log p`, and,
2. a Gaussian's score is linear in `X − μ` for `μ` and quadratic for `Σ`,
   so Stein (first and second order) turns it into `∇f` and `∇²f`.

Let,

  ```
  X ~ N(μ, Σ),  Σ ≻ 0,  with density p
  Y ≝ X − μ
  F(μ, Σ) ≝ 𝔼[ f(X) ]
  Δ        a symmetric direction, ⟨A, B⟩ ≝ tr(AᵀB)
  ```

The scores, using `d log det Σ = tr(Σ⁻¹ dΣ)` and `dΣ⁻¹ = −Σ⁻¹ dΣ Σ⁻¹`,

  ```
  ∇_μ log p                    = Σ⁻¹ Y
  ∂ₜ log p(x; μ, Σ + tΔ)|ₜ₌₀   = −½ tr(Σ⁻¹Δ) + ½ YᵀΣ⁻¹ΔΣ⁻¹Y
                               = ½ ⟨Δ, Σ⁻¹ (YYᵀ − Σ) Σ⁻¹⟩
  ```

First, Bonnet,

  ```
  ∇_μ F = 𝔼[ f(X) Σ⁻¹ Y ]
        = Σ⁻¹ Σ 𝔼[ ∇f(X) ]
        = 𝔼[ ∇f(X) ]
  ```

where the first step is the score-function gradient
(`reparameterization-vs-score.md`) and the second is Stein's lemma
(`stein-lemma.md`).

Second, Price,

  ```
  ∂ₜ F(μ, Σ + tΔ)|ₜ₌₀ = 𝔼[ f(X) ⋅ ½ ⟨Δ, Σ⁻¹ (YYᵀ − Σ) Σ⁻¹⟩ ]
                      = ½ ⟨Δ, Σ⁻¹ 𝔼[ f(X) (YYᵀ − Σ) ] Σ⁻¹⟩
                      = ½ ⟨Δ, Σ⁻¹ Σ 𝔼[ ∇²f(X) ] Σ Σ⁻¹⟩
                      = ½ ⟨Δ, 𝔼[ ∇²f(X) ]⟩
  ```

where the third step is second-order Stein (`stein-second-order.md`).

Hence,

  ```
  ∇_μ 𝔼[ f(X) ] = 𝔼[ ∇f(X) ]          # Bonnet
  ∇_Σ 𝔼[ f(X) ] = ½ 𝔼[ ∇²f(X) ]       # Price
  ```

where `∇_Σ` is the gradient over symmetric matrices with the Frobenius inner
product. Treating `Σᵢⱼ` and `Σⱼᵢ` as one variable doubles the off-diagonal
entries.

Example: `d = 1`, `f(x) = x²`, so `F = μ² + σ²`. Then `∂_μ F = 2μ = 𝔼[2X]`
and `∂_{σ²} F = 1 = ½ 𝔼[2]`.

Fine print: Price needs `∇²f`. Step two of second-order Stein gives
`𝔼[ ∇²f(X) ] = 𝔼[ ∇f(X) Yᵀ ] Σ⁻¹`, a gradient-only form; symmetrize it to
use as a `∇_Σ` estimate. Bonnet also follows by shifting `X = μ + Z` and
differentiating under `𝔼`; the score route shows both identities are Stein.
The names follow Rezende, Mohamed, Wierstra (arXiv:1401.4082, §3.1), who
attribute them to Bonnet (1964) and Price (IEEE Trans. Inf. Theory 1958); both
are used to learn Gaussian variational posteriors (Opper & Archambeau, Neural
Computation 2009).

Regularity: the proof needs `Σ ≻ 0` (for `Σ⁻¹` in the scores), `∇` to pass
through `∫` in `μ` and `Σ` (e.g., dominated `f ∇p`), and Stein's conditions:
`f` and `∇f` absolutely continuous with `𝔼‖∇f‖`, `𝔼‖∇²f‖`, and
`𝔼‖∇f‖ ‖Y‖` finite.
