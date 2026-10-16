# Doob's Maximal Inequality

Doob's inequalities follow from two facts:

1. stopping a submartingale at the first crossing of `λ` cannot raise its
   mean above `𝔼[Mₙ]`, and,
2. integrating the resulting tail bound over `λ` and applying
   Cauchy–Schwarz gives the `L²` bound.

Let,

  ```
  F₀ ⊆ F₁ ⊆ … ⊆ Fₙ            a filtration
  M₀, …, Mₙ                    a submartingale: 𝔼[Mₖ | Fₖ₋₁] ≥ Mₖ₋₁
  Dₖ ≝ Mₖ − Mₖ₋₁
  M̄  ≝ max_{k ≤ n} Mₖ
  λ  > 0
  A  ≝ { M̄ ≥ λ }
  τ  ≝ min{ k ≤ n : Mₖ ≥ λ },  or τ ≝ n if there is none
  ```

First, `𝔼[M_τ] ≤ 𝔼[Mₙ]`. The time `τ ≤ n` is a stopping time, so
`{τ < k} = {τ ≤ k − 1} ∈ Fₖ₋₁`, and, as in `optional-stopping.md`,

  ```
  𝔼[Mₙ] − 𝔼[M_τ] = 𝔼[ Σₖ₌₁ⁿ 1{τ < k} ⋅ Dₖ ]
                 = Σₖ₌₁ⁿ 𝔼[ 1{τ < k} ⋅ 𝔼[ Dₖ | Fₖ₋₁ ] ]
                 ≥ 0
  ```

where the last step uses `𝔼[Dₖ | Fₖ₋₁] ≥ 0`.

Second, split on `A`. On `A`, `M_τ ≥ λ`; off `A`, `τ = n`,

  ```
  𝔼[Mₙ] ≥ 𝔼[M_τ]
        = 𝔼[ M_τ 1_A ] + 𝔼[ Mₙ 1_{Aᶜ} ]
        ≥ λ P(A) + 𝔼[ Mₙ 1_{Aᶜ} ]
  ```

Subtracting `𝔼[Mₙ 1_{Aᶜ}]`,

  ```
  λ P(M̄ ≥ λ) ≤ 𝔼[ Mₙ 1{M̄ ≥ λ} ]
             ≤ 𝔼[ Mₙ⁺ ]
  ```

Third, the `L²` bound. Let `M` be a martingale with `𝔼[Mₙ²] < ∞`, and let
`M* ≝ max_{k ≤ n} |Mₖ|`. By conditional Jensen (`jensen.md`), `|Mₖ|` is a
nonnegative submartingale, so the second step gives
`λ P(M* ≥ λ) ≤ 𝔼[|Mₙ| 1{M* ≥ λ}]`. Then,

  ```
  𝔼[M*²] = ∫₀^∞ 2λ P(M* ≥ λ) dλ
         ≤ ∫₀^∞ 2 𝔼[ |Mₙ| 1{M* ≥ λ} ] dλ
         = 2 𝔼[ |Mₙ| ∫₀^{M*} dλ ]
         = 2 𝔼[ |Mₙ| M* ]
         ≤ 2 √𝔼[Mₙ²] ⋅ √𝔼[M*²]
  ```

where the third step is Tonelli, and the last is Cauchy–Schwarz. Since
`M*² ≤ Σₖ Mₖ²`, `𝔼[M*²] < ∞`; divide by `√𝔼[M*²]` and square.

Hence,

  ```
  P( max_{k ≤ n} Mₖ ≥ λ ) ≤ 𝔼[Mₙ⁺] / λ          (submartingale)
  𝔼[ max_{k ≤ n} Mₖ² ]    ≤ 4 𝔼[Mₙ²]            (martingale)
  ```

The first bound is tight for the constant `Mₖ = λ`. Compared with Markov
on `Mₙ` alone, it controls the whole path for the price of the endpoint.

Fine print: the constant `4` is sharp (cited, not proved). The same proof
gives `𝔼[M*ᵖ] ≤ (p/(p − 1))ᵖ 𝔼|Mₙ|ᵖ` for `p > 1` (Hölder replaces
Cauchy–Schwarz); for `p = 1` no such bound holds. The `L²` bound also holds
for nonnegative submartingales. Applied to `Mₖ = Sₖ`, a sum of independent
mean-zero terms, the first inequality on `Sₖ²` gives Kolmogorov's maximal
inequality `P(max |Sₖ| ≥ λ) ≤ 𝕍(Sₙ)/λ²`. Continuous-time versions follow
for right-continuous paths by taking limits over finite grids.

Regularity: the first inequality needs `M` adapted and integrable with the
submartingale property, and `λ > 0`. The `L²` inequality needs a martingale
(or nonnegative submartingale) with `𝔼[Mₙ²] < ∞`, which gives
`𝔼[Mₖ²] ≤ 𝔼[Mₙ²]` for all `k ≤ n` and so finiteness of `𝔼[M*²]`.
