# Second-Order Stein and Hermite Polynomials

Second-order Stein follows from applying Stein's lemma (`stein-lemma.md`)
twice: once to `f` times a coordinate, then to each `∂ᵢf`. In one dimension,
repeating it gives every derivative: the `k`-th Hermite polynomial picks out
the `k`-th derivative.

Let,

  ```
  X ~ N(μ, Σ)  in ℝᵈ
  Y ≝ X − μ
  f : ℝᵈ → ℝ   twice differentiable
  ```

with `𝔼|f(X)|`, `𝔼‖∇f(X)‖ ‖Y‖`, `𝔼‖∇²f(X)‖` finite.

First, apply Stein to `gⱼ(x) ≝ f(x)(x − μ)ⱼ`, whose gradient is
`∇f(x)(x − μ)ⱼ + f(x) eⱼ`,

  ```
  𝔼[ f(X) Y Yⱼ ] = Σ 𝔼[ ∇gⱼ(X) ]
                 = Σ 𝔼[ ∇f(X) Yⱼ ] + 𝔼[ f(X) ] Σ eⱼ
  ```

and stack the columns `j`,

  ```
  𝔼[ f(X) Y Yᵀ ] = Σ 𝔼[ ∇f(X) Yᵀ ] + 𝔼[ f(X) ] Σ
  ```

Second, apply Stein to each `∂ᵢf`: `𝔼[ ∂ᵢf(X) Y ] = Σ 𝔼[ ∇∂ᵢf(X) ]`, so row
`i` of `𝔼[ ∇f(X) Yᵀ ]` is `𝔼[ ∇∂ᵢf(X) ]ᵀ Σ`, i.e.,

  ```
  𝔼[ ∇f(X) Yᵀ ] = 𝔼[ ∇²f(X) ] Σ
  ```

Hence,

  ```
  𝔼[ f(X) (Y Yᵀ − Σ) ] = Σ 𝔼[ ∇²f(X) ] Σ
  𝔼[ f(X) (Y Yᵀ − σ²I) ] = σ⁴ 𝔼[ ∇²f(X) ]       Σ = σ² I
  ```

Example: `μ = 0` and `f(x) = ½ xᵀAx` with `A` symmetric, so `∇²f = A`.
Isserlis gives `𝔼[ (YᵀAY) YYᵀ ] = tr(AΣ) Σ + 2ΣAΣ`, so the left side is
`½ (tr(AΣ)Σ + 2ΣAΣ − tr(AΣ)Σ) = ΣAΣ`, matching the right.

Third, every order in one dimension. Let `Z ~ N(0, 1)` and the probabilists'
Hermite polynomials `He₀ = 1`, `He_{k+1}(z) = z He_k(z) − He_k′(z)`, so
`He₁ = z`, `He₂ = z² − 1`. For `k ≥ 0`,

  ```
  𝔼[ f(Z) He_{k+1}(Z) ] = 𝔼[ Z ⋅ f(Z) He_k(Z) ] − 𝔼[ f(Z) He_k′(Z) ]
                        = 𝔼[ (f He_k)′(Z) ] − 𝔼[ f(Z) He_k′(Z) ]
                        = 𝔼[ f′(Z) He_k(Z) ]
  ```

where the second step is Stein applied to `f He_k`. By induction on `k`,

  ```
  𝔼[ f(Z) He_k(Z) ] = 𝔼[ f⁽ᵏ⁾(Z) ]
  ```

`k = 1` is Stein's lemma and `k = 2` is the hence above with `d = 1`,
`σ = 1`. Taking `f = He_j` gives orthogonality, `𝔼[ He_j He_k ] = k! δⱼₖ`,
using `He_j′ = j He_{j−1}` (cited) and `𝔼[ He_m(Z) ] = 0` for `m ≥ 1`. So the
Hermite coefficients of `f` are its expected derivatives.

Fine print: with `ξ ~ N(0, σ²I)` and any constant `b`,
`(f(ξ) − b)(ξξᵀ − σ²I)/σ⁴` is unbiased for `𝔼[ ∇²f(ξ) ]`, since
`𝔼[ ξξᵀ − σ²I ] = 0`. This is a zeroth-order Hessian estimator from function
values alone, the second-order analogue of the score-function gradient. The
multivariate Hermite form replaces `He_k` with tensor Hermite polynomials
and `f⁽ᵏ⁾` with `∇ᵏf` (cited, not proved).

Regularity: the proof needs Stein's lemma to apply to `f ⋅ (x − μ)ⱼ` and to
each `∂ᵢf`, i.e., `f` and `∇f` absolutely continuous with the moments above
finite. The Hermite induction needs `𝔼|f⁽ʲ⁾(Z)| ⋅ |He_k(Z)|` finite for the
orders used.
