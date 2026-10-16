# Slutsky and the Delta Method

The delta method follows from two facts:

1. Slutsky: a sequence converging in distribution, paired with one
   converging in probability to a constant, converges jointly, and,
2. a differentiable `g` is linear plus a remainder that vanishes faster than
   `‖x − μ‖`.

Let,

  ```
  Xₙ → X  in distribution       (random vectors in ℝᵈ)
  Yₙ → c  in probability        (c a constant)
  h       bounded by B, L-Lipschitz, on ℝᵈ × ℝᵏ
  ```

Two results are cited, not proved. Portmanteau: `Uₙ → U` in distribution iff
`𝔼[h(Uₙ)] → 𝔼[h(U)]` for all bounded Lipschitz `h`. Continuous mapping:
`Uₙ → U` in distribution implies `ψ(Uₙ) → ψ(U)` for continuous `ψ`.

First, the key step of Slutsky: `(Xₙ, Yₙ) → (X, c)` jointly. Fix `ε > 0` and
split,

  ```
  |𝔼 h(Xₙ, Yₙ) − 𝔼 h(X, c)| ≤ 𝔼|h(Xₙ, Yₙ) − h(Xₙ, c)| + |𝔼 h(Xₙ, c) − 𝔼 h(X, c)|
                            ≤ 𝔼[ min(2B, L‖Yₙ − c‖) ] + |𝔼 h(Xₙ, c) − 𝔼 h(X, c)|
                            ≤ Lε + 2B ⋅ P(‖Yₙ − c‖ > ε) + |𝔼 h(Xₙ, c) − 𝔼 h(X, c)|
                            → Lε + 0 + 0
  ```

where the third step splits on `‖Yₙ − c‖ ≤ ε`, and the limit uses
`Yₙ → c` in probability and that `x ↦ h(x, c)` is bounded continuous. Since
`ε` is arbitrary, portmanteau gives joint convergence. Continuous mapping
then gives `Xₙ + Yₙ → X + c` and `Yₙ Xₙ → cX` (for scalar `Yₙ`).

Second, set up the delta method. Let `X̄ₙ` be the mean of `n` iid vectors with
mean `μ` and covariance `Σ`, so the multivariate CLT (cited) gives
`√n (X̄ₙ − μ) → Z ~ N(0, Σ)`. Let `g : ℝᵈ → ℝ` be differentiable at `μ`, write
`∇g = ∇g(μ)`, and define the remainder,

  ```
  r(x) ≝ ( g(x) − g(μ) − ∇gᵀ(x − μ) ) / ‖x − μ‖   for x ≠ μ
  r(μ) ≝ 0
  ```

Differentiability at `μ` is exactly continuity of `r` at `μ`.

Third, decompose,

  ```
  √n ( g(X̄ₙ) − g(μ) ) = ∇gᵀ √n (X̄ₙ − μ) + r(X̄ₙ) ⋅ √n ‖X̄ₙ − μ‖
                      = Aₙ + Bₙ
  ```

Fourth, each piece converges. By continuous mapping,
`Aₙ → ∇gᵀ Z ~ N(0, ∇gᵀ Σ ∇g)` and `√n ‖X̄ₙ − μ‖ → ‖Z‖` in distribution.
Next, `X̄ₙ → μ` in probability (weak law), and `r` is continuous at `μ`
with `r(μ) = 0`: for each `ε`, pick `δ` with `|r(x)| ≤ ε` on `‖x − μ‖ < δ`,
so,

  ```
  P( |r(X̄ₙ)| > ε ) ≤ P( ‖X̄ₙ − μ‖ ≥ δ )
                   → 0
  ```

Slutsky (product, `c = 0`) gives `Bₙ → 0 ⋅ ‖Z‖ = 0` in distribution, which
is convergence in probability since the limit is constant. Slutsky (sum,
`c = 0`) then gives `Aₙ + Bₙ → ∇gᵀ Z`.

Hence,

  ```
  √n ( g(X̄ₙ) − g(μ) ) → N( 0, ∇g(μ)ᵀ Σ ∇g(μ) )   in distribution
  ```

For vector `g : ℝᵈ → ℝᵐ` with Jacobian `J` at `μ`, the same proof applied
coordinatewise and jointly gives `N(0, J Σ Jᵀ)`. If `∇g(μ) = 0`, the limit is
the point mass at `0`, and a second-order expansion at rate `n` is needed.

Regularity: Slutsky needs the second sequence to converge to a constant (to
a random limit it fails in general). The delta method needs a CLT for `X̄ₙ`
(finite `Σ`), `g` differentiable at `μ` (not necessarily elsewhere), and the
cited portmanteau, continuous-mapping, and multivariate CLT results.
