# Polyak–Tsypkin

The best nonlinearity for Robbins–Monro follows from two facts:

1. at the optimal gain, a nonlinear stochastic approximation has the
   asymptotic variance of the matching M-estimator, `𝔼[ψ²] / 𝔼[ψ′]²`, and,
2. integration by parts makes `𝔼[ψ′] = 𝔼[ψ ⋅ s]` with `s` the score, so
   Cauchy–Schwarz bounds that variance below by `1/I`, with equality at
   `ψ ∝ s`.

So the optimal update applies the noise density's own score, and over a
contamination neighborhood the minimax update is Huber's clip.

Let `Y = θ⋆ + Z` with `Z ~ p`, symmetric about `0`, and,

  ```
  ψ : ℝ → ℝ                  odd, nondecreasing, the nonlinearity
  θ_{t+1} ≝ θ_t + (a/t) ⋅ ψ(Y_t − θ_t)
  s(z)    ≝ −p′(z)/p(z)       the location score
  I       ≝ 𝔼_p[s²]            Fisher information for location
  H       ≝ 𝔼_p[ψ′(Z)],   σ² ≝ 𝔼_p[ψ(Z)²]
  ```

This is Robbins–Monro (robbins-monro.md) for the root of
`h(θ) ≝ −𝔼_p[ψ(θ⋆ + Z − θ)]`, which vanishes at `θ⋆` by symmetry and has
slope `h′(θ⋆) = H > 0`; the noise at `θ⋆` has variance `σ²`.

First, the asymptotic variance at gain `a`. For a scalar root with slope
`H` and noise variance `σ²`, steps `a/t` with `2aH > 1` give
(Chung 1954; Fabian 1968, cited)

  ```
  √t (θ_t − θ⋆) → N( 0, a² σ² / (2aH − 1) )
  ```

Second, the best gain. Differentiate in `a`,

  ```
  d/da [ a² / (2aH − 1) ] = 2a (aH − 1) / (2aH − 1)²
  ```

which vanishes at `a = 1/H`, giving

  ```
  V(ψ) ≝ σ² / H² = 𝔼_p[ψ²] / 𝔼_p[ψ′]²
  ```

the same as the M-estimator `Σ ψ(Yᵢ − θ̂) = 0`. Averaging the iterates
(Polyak–Ruppert) reaches `V(ψ)` without knowing `H`.

Third, the bound. Integrate by parts, with `ψ p → 0` at `±∞`,

  ```
  H = ∫ ψ′(z) p(z) dz
    = −∫ ψ(z) p′(z) dz
    = 𝔼_p[ ψ(Z) s(Z) ]
  ```

and apply Cauchy–Schwarz,

  ```
  H² = 𝔼_p[ψ s]² ≤ 𝔼_p[ψ²] ⋅ 𝔼_p[s²] = σ² I
  ```

Hence,

  ```
  V(ψ) = σ² / H² ≥ 1 / I
  ```

with equality iff `ψ ∝ s`: the optimal nonlinearity is the noise score, and
the resulting recursion attains the Cramér–Rao bound (cramer-rao.md), as the
MLE does.

Fourth, unknown noise. If `p` is known only to lie in
`𝒬_ε(f) = { (1 − ε) f + ε h }`, the score of the least favorable member
minimizes `sup_p V(ψ, p)`, and for log-concave `f` it is the clipped score
`ψ_c = clip(s_f, −k, k)` (least-favorable-density.md).

Examples.

- Gaussian noise, `s(z) = z/σ²`. `ψ = s` with `a = 1/H = σ²` gives
  `θ_{t+1} = θ_t + (1/t)(Y_t − θ_t)`, the running mean, with `V = σ²`.
- Laplace noise, `s(z) = sign(z)/b`. `ψ = sign` gives
  `θ_{t+1} = θ_t + (a/t) sign(Y_t − θ_t)`, stochastic median tracking, with
  `H = 2p(0) = 1/b` and `V = 1/(4p(0)²) = b² = 1/I`.
- Contaminated Gaussian. `ψ = clip(z, −c, c)` is Huberized Robbins–Monro:
  each draw moves the iterate by at most `ac/t`, so one outlier cannot throw
  it, and the variance is minimax over `𝒬_ε`. Clipping the averaged step
  instead (a batch-norm clip) does not bound any one draw's influence.

Regularity: `p` symmetric (so the root is `θ⋆`), absolutely continuous with
`I < ∞`; `ψ` odd, nondecreasing, with `H > 0` and `σ² < ∞`; `ψ p → 0` at
`±∞` for the integration by parts; the step `a/t` with `2aH > 1` and the
Robbins–Monro conditions for the cited CLT. The vector case replaces `V` by
`H⁻¹ Σ H⁻ᵀ` and the bound by `I⁻¹` in the Loewner order.
