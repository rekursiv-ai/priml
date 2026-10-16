# Fisher Information as Expected Hessian

The Hessian identity follows from two facts:

1. the score has zero mean for every `θ`, and,
2. `∇ p = p ⋅ s`, so differentiating that zero produces `𝔼[s sᵀ]`.

Let,

  ```
  s(Y) ≝ ∇ log p(Y|x,θ)
  I    ≝ 𝔼_{p(Y|x,θ)}[ s(Y) s(Y)ᵀ ]
  ```

First, the score is zero mean for every `θ` (cramer-rao.md, first step),

  ```
  𝔼_{p(Y|x,θ)}[ s(Y) ] = Σₖ p(Y=k|x,θ) ⋅ s(k)
                       = 0
  ```

Second, differentiate this identity in `θ`. Write `pₖ ≝ p(Y=k|x,θ)` and
`sₖ ≝ s(k)`. The `(i,j)` entry of `∇ (sₖᵀ)` is `∂ᵢ ∂ⱼ log pₖ`, so
`∇ (sₖᵀ) = ∇² log pₖ`,

  ```
  0 = ∇ Σₖ pₖ ⋅ sₖᵀ
    = Σₖ (∇ pₖ) ⋅ sₖᵀ + Σₖ pₖ ⋅ ∇ (sₖᵀ)
    = Σₖ pₖ ⋅ sₖ sₖᵀ + Σₖ pₖ ⋅ ∇² log pₖ
    = 𝔼_{p(Y|x,θ)}[ s(Y) s(Y)ᵀ ] + 𝔼_{p(Y|x,θ)}[ ∇² log p(Y|x,θ) ]
  ```

where the second step is the product rule, and the third uses
`∇ pₖ = pₖ ⋅ ∇ log pₖ = pₖ ⋅ sₖ`.

Hence,

  ```
  I = 𝔼_{p(Y|x,θ)}[ s(Y) s(Y)ᵀ ] = −𝔼_{p(Y|x,θ)}[ ∇² log p(Y|x,θ) ]
  ```

The identity holds in expectation only. The observed `−∇² log p(Y|x,θ)` need
not equal `s(Y) s(Y)ᵀ`, nor be positive semidefinite, for a given `Y`.

For softmax, let `f(k) ≝ f_θ(x)ₖ` be the logits and `A ≝ log Σⱼ exp f(j)`, so

  ```
  log p(Y=k|x,θ) = f(k) − A
  ```

Third, the score centers the logit gradient,

  ```
  s(k) = ∇ f(k) − ∇ A
       = ∇ f(k) − Σⱼ (exp f(j) / Σₘ exp f(m)) ⋅ ∇ f(j)
       = ∇ f(k) − 𝔼_p[ ∇ f(Y) ]
  ```

Fourth, the outer-product form is a covariance,

  ```
  I = 𝔼_p[ s(Y) s(Y)ᵀ ]
    = 𝔼_p[ (∇f(Y) − 𝔼_p[∇f(Y)]) (∇f(Y) − 𝔼_p[∇f(Y)])ᵀ ]
    = 𝕍_p[ ∇f(Y) ]
  ```

Fifth, the Hessian form agrees. Differentiate `∇ A = Σⱼ pⱼ ∇f(j)`,

  ```
  ∇² A = Σⱼ pⱼ ⋅ ∇² f(j) + Σⱼ (∇ pⱼ) ⋅ ∇f(j)ᵀ
       = 𝔼_p[ ∇² f(Y) ] + Σⱼ pⱼ ⋅ sⱼ ∇f(j)ᵀ
       = 𝔼_p[ ∇² f(Y) ] + 𝔼_p[ (∇f(Y) − 𝔼_p[∇f(Y)]) ∇f(Y)ᵀ ]
       = 𝔼_p[ ∇² f(Y) ] + 𝕍_p[ ∇f(Y) ]
  ```

so

  ```
  −𝔼_p[ ∇² log p(Y|x,θ) ] = −𝔼_p[ ∇² f(Y) ] + ∇² A
                          = −𝔼_p[ ∇² f(Y) ] + 𝔼_p[ ∇² f(Y) ] + 𝕍_p[ ∇f(Y) ]
                          = 𝕍_p[ ∇f(Y) ]
  ```

If the logits are linear in `θ` (multinomial logistic regression),
`∇² f = 0`. Then `−∇² log p(Y=k|x,θ) = 𝕍_p[∇f(Y)]` for every `k`: the
observed and expected information coincide.

Regularity: the proof needs `log p(Y|x,θ)` twice differentiable in `θ`, `∇`
to pass through `Σₖ` twice, and `𝔼[s sᵀ]` finite. All hold for softmax over
finite `Y` with `f_θ(x)` twice differentiable in `θ`.
