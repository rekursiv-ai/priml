# Cramér–Rao

Cramér–Rao follows from two facts:

1. the score has zero mean, and,
2. the score "explains" any unbiased estimator.

Let,

  ```
  s(Y) ≝ ∇ log p(Y|x,θ)
  I    ≝ 𝔼_{p(Y|x,θ)}[ s(Y) s(Y)ᵀ ]
  ```

And let `T(Y)` be unbiased for `ψ(θ)`, i.e., `𝔼_{p(Y|x,θ)}[T(Y)] = ψ(θ)`.

First, the score is zero mean, since probabilities sum to one,

  ```
  𝔼_{p(Y|x,θ)}[ s(Y) ] = Σₖ p(Y=k|x,θ) ⋅ ∇ log p(Y=k|x,θ)
                       = Σₖ ∇ p(Y=k|x,θ)
                       = ∇ Σₖ p(Y=k|x,θ)
                       = ∇ 1
                       = 0
  ```

Second, differentiating unbiasedness shows `T` covaries with the score exactly
by `∇ψ`,

  ```
  ∇ψ(θ) = ∇ Σₖ T(k) ⋅ p(Y=k|x,θ)
        = Σₖ T(k) ⋅ p(Y=k|x,θ) ⋅ ∇ log p(Y=k|x,θ)
        = 𝔼_{p(Y|x,θ)}[ T(Y) ⋅ s(Y) ]
        = 𝔼_{p(Y|x,θ)}[ (T(Y) − ψ(θ)) ⋅ s(Y) ]
  ```

where the second step uses that `T` does not depend on `θ`, and the last step
uses the zero-mean score.

Third, regress `T` on the score. Choose `L(Y) ≝ ∇ψᵀ I⁻¹ s(Y)`; the residual's
second moment is nonnegative,

  ```
  0 ≤ 𝔼_{p(Y|x,θ)}[ (T(Y) − ψ(θ) − L(Y))² ]
    = 𝕍(T) − 2 ⋅ 𝔼[ (T(Y) − ψ(θ)) ⋅ L(Y) ] + 𝔼[ L(Y)² ]
    = 𝕍(T) − 2 ⋅ ∇ψᵀ I⁻¹ 𝔼[ (T(Y) − ψ(θ)) ⋅ s(Y) ] + ∇ψᵀ I⁻¹ 𝔼[ s(Y) s(Y)ᵀ ] I⁻¹ ∇ψ
    = 𝕍(T) − 2 ⋅ ∇ψᵀ I⁻¹ ∇ψ + ∇ψᵀ I⁻¹ I I⁻¹ ∇ψ
    = 𝕍(T) − ∇ψᵀ I⁻¹ ∇ψ
  ```

Hence,

  ```
  𝕍(T) ≥ ∇ψᵀ I⁻¹ ∇ψ
  ```

Equality holds iff the residual is zero `p`-almost surely, i.e.,
`T(Y) − ψ(θ) = ∇ψᵀ I⁻¹ s(Y)`. In words, the estimator is affine in the score.

For the matrix form, let `T(Y)` be vector-valued and unbiased for `θ`. For any
fixed `u`, the scalar `uᵀT(Y)` is unbiased for `ψ(θ) = uᵀθ`, and `∇ψ = u`. The
scalar bound gives `uᵀ 𝕍(T) u = 𝕍(uᵀT) ≥ uᵀ I⁻¹ u` for all `u`, i.e.,

  ```
  𝕍(T) ⪰ I⁻¹
  ```

With `n` iid samples the scores add, so `Iₙ = n ⋅ I` and the bound becomes
`I⁻¹ / n`. MLE attains this asymptotically because
`θ̂ − θ ≈ I⁻¹ ⋅ (1/n) Σᵢ s(Yᵢ)`, which is the equality case.

Regularity: the proof needs `∇` to pass through `Σₖ`, `I` invertible, `T` not
a function of `θ`, and `𝕍(T) < ∞`. All hold for softmax over finite `Y` with
full-rank `𝕍_p[∇f(Y)]`.
