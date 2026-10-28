# Stein Identity

The Stein identity follows from one fact: `f ⋅ p` vanishes at infinity, so
the integral of its gradient is zero. It holds for any smooth density, and
turns an expectation involving `∇log p` into one involving only `∇f`.

Let,

  ```
  p : ℝᵈ → (0, ∞)       a differentiable density
  s_p(x) ≝ ∇ₓ log p(x)  ∈ ℝᵈ          # score in x, not in a parameter
  f : ℝᵈ → ℝ            differentiable, with f(x) p(x) → 0 as ‖x‖ → ∞
  ```

and assume `𝔼_p‖∇f‖ < ∞` and `𝔼_p|f| ‖s_p‖ < ∞`.

First, the product rule writes the integrand as a derivative,

  ```
  ∇(f p) = p ∇f + f ∇p
         = p ⋅ ( ∇f + f s_p )
  ```

where the second step uses `∇p = p ∇log p` where `p > 0`.

Second, integrate coordinate `i`,

  ```
  𝔼_p[ ∂ᵢf(X) + f(X) ∂ᵢ log p(X) ] = ∫ ∂ᵢ(f p)(x) dx
                                    = ∫ [ f p ]_{xᵢ = −∞}^{xᵢ = ∞} dx₋ᵢ
                                    = 0
  ```

where the second step is Fubini (`fubini-tonelli.md`) then the fundamental
theorem of calculus along `xᵢ`, and the third uses `f p → 0`.

Hence,

  ```
  𝔼_p[ ∇f(X) + f(X) ∇log p(X) ] = 0
  ```

and, for vector `f : ℝᵈ → ℝᵈ`, applying it to each `fᵢ` along coordinate `i`
and summing,

  ```
  𝔼_p[ ∇·f(X) + f(X)ᵀ ∇log p(X) ] = 0
  ```

Three specializations:

1. Gaussian. With `p = N(μ, Σ)`, `∇log p(x) = −Σ⁻¹(x − μ)`, and the identity
   is `𝔼[ f(X)(X − μ) ] = Σ 𝔼[ ∇f(X) ]`: Stein's lemma (`stein-lemma.md`).
2. Constants. With `f ≡ 1`, `𝔼_p[ ∇ₓ log p(X) ] = 0`. For a location family
   `p(x − θ)`, `∇_θ log p = −∇ₓ log p`, so this is the zero-mean score of
   Cramér–Rao (`cramer-rao.md`).
3. Score matching, below.

Score matching fits an unnormalized model `p_θ` to data from an unknown `q`
by matching scores. Let,

  ```
  s_θ ≝ ∇ₓ log p_θ,   s_q ≝ ∇ₓ log q
  J(θ) ≝ ½ 𝔼_q[ ‖s_θ(X) − s_q(X)‖² ]
  ```

`J` needs `s_q`, which is unknown. Expand, then apply the identity under `q`
to `f = s_θ,ᵢ` for each `i`, so `𝔼_q[ s_θ,ᵢ ∂ᵢ log q ] = −𝔼_q[ ∂ᵢ s_θ,ᵢ ]`,

  ```
  J(θ) = ½ 𝔼_q‖s_θ‖² − 𝔼_q[ s_θᵀ s_q ] + ½ 𝔼_q‖s_q‖²
       = ½ 𝔼_q‖s_θ‖² + 𝔼_q[ ∇·s_θ ] + ½ 𝔼_q‖s_q‖²
       = 𝔼_q[ ½ ‖∇log p_θ‖² + Δ log p_θ ] + const
  ```

where `Δ` is the Laplacian and `const` does not depend on `θ`. The last line
needs only samples from `q`, and `∇ₓ log p_θ` does not see the normalizer of
`p_θ` (Hyvärinen, JMLR 2005, cited).

Fine print: the Stein discrepancy
`S(q; p) ≝ sup_{f ∈ F} 𝔼_q[ ∇·f + fᵀ∇log p ]` is 0 when `q = p` by the
identity; for a rich enough `F`, e.g. the unit ball of a suitable RKHS, it is
0 only when `q = p` (Liu, Lee, Jordan, ICML 2016, cited, not proved). Like
score matching, it needs `p` only up to normalization. Stein variational
gradient descent moves particles along the maximizing `f` in that RKHS (Liu &
Wang, arXiv:1608.04471, cited).

Regularity: the proof needs `p > 0` and differentiable, `f` differentiable,
`f p → 0` at infinity along each coordinate (for a bounded support, `f p = 0`
on the boundary), and `𝔼_p‖∇f‖`, `𝔼_p|f| ‖∇log p‖` finite so Fubini applies.
Score matching additionally needs `s_θ` differentiable and the identity's
conditions for each `f = s_θ,ᵢ` under `q`.
