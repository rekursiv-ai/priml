# EM Monotonicity

EM never decreases the likelihood because of three facts:

1. the ELBO lower-bounds `log p(x|θ)` for every `q` (elbo.md),
2. the E-step makes the bound tight at the current `θ`, and,
3. the M-step increases the bound in `θ`.

Let,

  ```
  p(x, z|θ) a model with latent z,   p(x|θ) ≝ Σ_z p(x, z|θ)
  F(q, θ) ≝ 𝔼_q[ log p(x, z|θ) ] + H(q)
          = log p(x|θ) − 𝕂( q(z) ‖ p(z|x, θ) )
  E-step:  q_{t+1} ≝ p(z|x, θ_t)
  M-step:  θ_{t+1} ≝ argmax_θ 𝔼_{q_{t+1}}[ log p(x, z|θ) ]
  ```

The second form of `F` is the ELBO decomposition (elbo.md). `H(q)` does not
depend on `θ`, so the M-step maximizes `F(q_{t+1}, ·)`.

First, the E-step makes the bound tight. With `q_{t+1} = p(z|x, θ_t)` the KL
term is zero,

  ```
  F(q_{t+1}, θ_t) = log p(x|θ_t) − 𝕂( p(z|x, θ_t) ‖ p(z|x, θ_t) )
                  = log p(x|θ_t)
  ```

Second, chain the bound, the M-step, and the first step,

  ```
  log p(x|θ_{t+1}) = F(q_{t+1}, θ_{t+1}) + 𝕂( q_{t+1} ‖ p(z|x, θ_{t+1}) )
                   ≥ F(q_{t+1}, θ_{t+1})
                   ≥ F(q_{t+1}, θ_t)
                   = log p(x|θ_t)
  ```

where the second step is `𝕂 ≥ 0` (Gibbs), and the third holds because
`θ_{t+1}` maximizes `F(q_{t+1}, ·)`.

Hence,

  ```
  log p(x|θ_{t+1}) ≥ log p(x|θ_t)
  ```

Equality forces both inequalities to be tight: `θ_t` already maximizes
`F(q_{t+1}, ·)`, and the posterior did not move,
`p(z|x, θ_{t+1}) = p(z|x, θ_t)`.

Third, the Gaussian mixture M-step. Take `z_i ∈ {1, …, K}`,
`p(x_i, z_i = k|θ) = π_k ⋅ 𝒩(x_i | μ_k, Σ_k)`, iid over `i = 1, …, N`. The
E-step gives responsibilities, and the M-step objective separates,

  ```
  γ_ik ≝ π_k 𝒩(x_i|μ_k, Σ_k) / Σ_j π_j 𝒩(x_i|μ_j, Σ_j)
  N_k  ≝ Σᵢ γ_ik
  Q(θ) ≝ Σᵢ Σₖ γ_ik ⋅ [ log π_k + log 𝒩(x_i|μ_k, Σ_k) ]
  ```

For `π`, maximize `Σₖ N_k log π_k` subject to `Σₖ π_k = 1`. The Lagrange
condition `N_k/π_k = λ` and `Σₖ N_k = N` give,

  ```
  π_k = N_k / λ
      = N_k / N
  ```

For `μ_k`, set the gradient to zero; `Q` is concave in `μ_k`,

  ```
  0   = Σᵢ γ_ik ⋅ Σ_k⁻¹ (x_i − μ_k)
  μ_k = Σᵢ γ_ik x_i / N_k
  ```

For `Σ_k`, write `Λ ≝ Σ_k⁻¹`. The `Λ`-dependent part of `Q` is
`Σᵢ γ_ik [½ log det Λ − ½ (x_i − μ_k)ᵀΛ(x_i − μ_k)]`, concave in `Λ`. Using
`∇_Λ log det Λ = Λ⁻¹` and `∇_Λ vᵀΛv = vvᵀ`,

  ```
  0   = ½ N_k Λ⁻¹ − ½ Σᵢ γ_ik (x_i − μ_k)(x_i − μ_k)ᵀ
  Σ_k = Σᵢ γ_ik (x_i − μ_k)(x_i − μ_k)ᵀ / N_k
  ```

These are weighted counts, means, and covariances. By the second step, each
round of responsibilities then these updates cannot lower
`Σᵢ log Σₖ π_k 𝒩(x_i|μ_k, Σ_k)`.

Fine print: monotonicity does not imply convergence to a maximizer; EM can
stall at saddle points or local maxima, and in mixtures the likelihood is
unbounded when a `Σ_k` collapses onto one point. A generalized EM that only
increases `F(q_{t+1}, ·)`, without maximizing, keeps the same chain. For
continuous `z`, replace `Σ_z` with `∫ dz`. Viewed as coordinate ascent on
`F`, the E-step maximizes over `q` and the M-step over `θ`.

Regularity: the proof needs `p(x|θ) > 0`, the posterior `p(z|x, θ_t)`
computable (exact E-step), the M-step maximum attained, and `𝕂 ≥ 0`. The
GMM updates need `N_k > 0` and the weighted scatter matrix positive definite.
