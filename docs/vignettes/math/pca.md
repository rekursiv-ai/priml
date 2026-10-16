# Principal Component Analysis

Top-`k` PCA projects onto the top `k` eigenvectors of the covariance. The
proof rests on two facts:

1. for an orthogonal projector, retained plus lost energy is total energy,
   so maximizing projected variance minimizes reconstruction error, and,
2. Ky Fan: `tr(WᵀΣW)` over orthonormal `W` is at most the sum of the top `k`
   eigenvalues.

Let,

  ```
  x₁, …, x_N ∈ ℝᵈ,   Σᵢ xᵢ = 0
  Σ    ≝ (1/N) Σᵢ xᵢxᵢᵀ = Σⱼ λⱼ qⱼqⱼᵀ,   qᵢᵀqⱼ = δᵢⱼ,   λ₁ ≥ … ≥ λ_d
  W ∈ ℝᵈˣᵏ,   WᵀW = I
  P    ≝ WWᵀ
  E(W) ≝ (1/N) Σᵢ ‖xᵢ − P xᵢ‖²
  ```

where the eigendecomposition exists by spectral-theorem.md.

First, `P` is a symmetric idempotent,

  ```
  P² = W WᵀW Wᵀ
     = W Wᵀ
     = P
  ```

so for any `x` the reconstruction error is

  ```
  ‖x − Px‖² = xᵀ (I − P)ᵀ (I − P) x
            = xᵀ (I − 2P + P²) x
            = xᵀ (I − P) x
            = ‖x‖² − ‖Wᵀx‖²
  ```

Second, averaging over the data, using `aᵀa = tr(aaᵀ)` and linearity of `tr`,

  ```
  E(W) = (1/N) Σᵢ ‖xᵢ‖² − (1/N) Σᵢ ‖Wᵀxᵢ‖²
       = tr Σ − (1/N) Σᵢ tr(Wᵀ xᵢxᵢᵀ W)
       = tr Σ − tr(WᵀΣW)
  ```

Since `tr Σ` does not depend on `W`, minimizing `E` and maximizing
`tr(WᵀΣW)` have the same solutions. The latter is the total variance of the
centered projection `zᵢ ≝ Wᵀxᵢ`.

Third, reduce to weights. Let `hⱼ ≝ ‖Wᵀqⱼ‖² = qⱼᵀPqⱼ`. Then

  ```
  tr(WᵀΣW) = Σⱼ λⱼ tr(Wᵀ qⱼqⱼᵀ W)
           = Σⱼ λⱼ hⱼ
  ```

The weights satisfy `0 ≤ hⱼ ≤ 1`, since `qⱼᵀPqⱼ = ‖Pqⱼ‖² ≤ ‖qⱼ‖² = 1` for
an orthogonal projector, and

  ```
  Σⱼ hⱼ = tr(Qᵀ P Q)
        = tr(P)
        = tr(WᵀW)
        = k
  ```

where `Q ≝ [q₁, …, q_d]` is orthogonal.

Fourth, Ky Fan. Using `λⱼ ≥ λₖ` with `hⱼ − 1 ≤ 0` for `j ≤ k`, and `λⱼ ≤ λₖ`
with `hⱼ ≥ 0` for `j > k`,

  ```
  Σⱼ λⱼhⱼ − Σ_{j≤k} λⱼ = Σ_{j≤k} λⱼ (hⱼ − 1) + Σ_{j>k} λⱼ hⱼ
                       ≤ λₖ Σ_{j≤k} (hⱼ − 1) + λₖ Σ_{j>k} hⱼ
                       = λₖ (Σⱼ hⱼ − k)
                       = 0
  ```

Fifth, `W = Qₖ = [q₁, …, qₖ]` attains it, since `hⱼ = 1` for `j ≤ k` and
`hⱼ = 0` otherwise.

Hence,

  ```
  max_{WᵀW=I} tr(WᵀΣW) = Σ_{j≤k} λⱼ
  min_{WᵀW=I} E(W)     = Σ_{j>k} λⱼ
  ```

both attained at `W = Qₖ`. Any `W = QₖR` with `R` orthogonal also attains them,
since `P` is unchanged. If `λₖ > λₖ₊₁`, equality in the fourth step forces
`hⱼ = 0` for `j > k`, so `range(W) = span(q₁, …, qₖ)`: the optimal subspace is
unique. If `λₖ = λₖ₊₁`, it is not.

Regularity: the proof needs `Σ` symmetric (for the orthonormal eigenbasis)
and centered data so `Σ` is a covariance; without centering the same result
holds with `Σ` the second-moment matrix. `λⱼ ≥ 0` is not used.
