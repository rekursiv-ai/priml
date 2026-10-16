# Kernel Ridge Regression Is the GP Posterior Mean

The equivalence follows from two facts:

1. the representer theorem (see representer.md) reduces kernel ridge
   regression to a quadratic in `α ∈ ℝⁿ`, whose solution is
   `(K + λI)⁻¹ y`, and,
2. Gaussian conditioning gives the posterior mean through the same matrix
   `(K + λI)⁻¹`.

Let,

  ```
  X ≝ (x₁..xₙ),  y ∈ ℝⁿ
  K ≝ k(X, X) ∈ ℝ^{n×n}            Kᵢⱼ ≝ k(xᵢ, xⱼ), symmetric PSD
  k(x, X) ∈ ℝ^{1×n},  k(X, x) ≝ k(x, X)ᵀ
  λ > 0
  ```

First, solve kernel ridge regression. Minimize over `f ∈ H` the objective
`J(f) ≝ Σᵢ (yᵢ − f(xᵢ))² + λ ‖f‖²_H`. The representer theorem gives
`f = k(·, X) α`, so `f(X) = Kα` and `‖f‖²_H = αᵀKα`,

  ```
  J(α)  = ‖y − Kα‖² + λ αᵀKα
  ∇J(α) = −2K (y − Kα) + 2λ Kα
        = 2K ( (K + λI) α − y )
  ```

`J` is convex in `α`, so any zero of the gradient is a minimizer.
`K + λI` is invertible since `K ⪰ 0` and `λ > 0`, so
`α = (K + λI)⁻¹ y` is one,

  ```
  f̂(x) ≝ k(x, X) α
       = k(x, X) (K + λI)⁻¹ y
  ```

Second, condition a Gaussian. For jointly Gaussian zero-mean `(a, b)` with
`Σ_bb` invertible, set `C ≝ Σ_ab Σ_bb⁻¹` and `r ≝ a − C b`,

  ```
  𝕍(r, b) = Σ_ab − C Σ_bb
          = 0
  ```

`r` and `b` are jointly Gaussian and uncorrelated, hence independent. So,

  ```
  𝔼[a | b] = C b + 𝔼[r]
           = Σ_ab Σ_bb⁻¹ b
  𝕍(a | b) = 𝕍(r)
           = Σ_aa − 2 C Σ_ba + C Σ_bb Cᵀ
           = Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba
  ```

Third, apply it to the GP. Let `f ~ GP(0, k)` and `yᵢ = f(xᵢ) + eᵢ` with
`eᵢ ~ N(0, λ)` iid, independent of `f`. Take `a = f(x)` and `b = y`,

  ```
  Σ_bb = 𝕍(f(X) + e) = K + λI
  Σ_ab = 𝕍(f(x), f(X) + e) = k(x, X)
  Σ_aa = k(x, x)
  ```

So,

  ```
  𝔼[f(x) | y] = k(x, X) (K + λI)⁻¹ y
  𝕍(f(x) | y) = k(x, x) − k(x, X) (K + λI)⁻¹ k(X, x)
  ```

Hence, with ridge penalty `λ` equal to the noise variance,

  ```
  f̂_KRR(x) = 𝔼_GP[f(x) | y] = k(x, X) (K + λI)⁻¹ y
  ```

The posterior variance has no KRR counterpart as a point estimate. It is a
worst-case error: over a unit ball of functions and noise, the squared KRR
error at `x` is at most `𝕍(f(x) | y)`, and some element attains it. Let
`w = (K + λI)⁻¹ k(X, x)`, so `f̂(x) = wᵀ y`. Give the pairs `(f, e)` the inner
product `⟨f, f′⟩_H + eᵀe′/λ`. The error `f(x) − wᵀ(f(X) + e)` is a linear
functional of `(f, e)`, with representer `(k(x, ·) − Σᵢ wᵢ k(xᵢ, ·), −λw)`,
so its worst case over the unit ball is that representer's norm,

  ```
  sup_{‖f‖²_H + ‖e‖²/λ ≤ 1} (f(x) − wᵀ(f(X) + e))² = k(x, x) − 2 wᵀk(X, x) + wᵀKw + λ wᵀw
                                                   = k(x, x) − 2 wᵀk(X, x) + wᵀ(K + λI) w
                                                   = k(x, x) − k(x, X) (K + λI)⁻¹ k(X, x)
                                                   = 𝕍(f(x) | y)
  ```

The identity of means is exact; there is no inequality to be tight.

Fine print: with the averaged objective `(1/n) Σᵢ (yᵢ − f(xᵢ))² + λ‖f‖²_H`,
the noise variance is `nλ`. If `K` is singular, other zeros `α′` of `∇J`
exist. They satisfy `K(K + λI)(α′ − α) = 0`, hence `K(α′ − α) = 0` (the
factors commute and `K + λI` is invertible). Then
`‖k(·, X)(α′ − α)‖²_H = (α′ − α)ᵀK(α′ − α) = 0`, so `f̂` is unique. The GP sample paths are
typically not in `H`, so the match is between the KRR estimate and the
posterior mean, not a claim that `f ∈ H`. A nonzero prior mean `m` gives
`m(x) + k(x, X)(K + λI)⁻¹(y − m(X))`.

Regularity: the proof needs `k` a PSD kernel with RKHS `H`, `λ > 0`, squared
loss, and for the GP, Gaussian noise of variance `λ` independent of `f`, and a
zero prior mean.
