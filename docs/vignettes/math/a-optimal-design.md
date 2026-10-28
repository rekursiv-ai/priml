# A-Optimal Design

The A-optimal design bound follows from two facts:

1. Cramér–Rao (`cramer-rao.md`) bounds any unbiased estimator of a linear
   model's coefficients by the inverse Fisher information `τ² (XᵀX)⁻¹`, and,
2. for a fixed energy budget `tr(XᵀX)`, the trace of an inverse is smallest
   when all eigenvalues are equal.

Let,

  ```
  g ∈ ℝⁿ                         unknown coefficients
  X ∈ ℝ^{K×n},  rows xₖ          design, chosen in advance, rank n (so K ≥ n)
  Yₖ = xₖᵀ g + εₖ,  εₖ ~ N(0, τ²) iid
  budget:  Σₖ ‖xₖ‖² ≤ K r²
  ```

First, the Fisher information. The log-likelihood is
`−‖Y − Xg‖² / (2τ²) + const`, so the score and information are,

  ```
  s(Y) = Xᵀ (Y − X g) / τ²
  I    = 𝔼[ s sᵀ ]
       = Xᵀ 𝔼[ εεᵀ ] X / τ⁴
       = XᵀX / τ²
  ```

Second, Cramér–Rao in matrix form gives, for any unbiased `ĝ`,

  ```
  𝕍(ĝ)    ⪰ I⁻¹ = τ² (XᵀX)⁻¹
  tr 𝕍(ĝ) ≥ τ² tr( (XᵀX)⁻¹ )
  ```

Third, bound the trace of the inverse by the trace. With eigenvalues
`λ₁..λₙ > 0` of `XᵀX`, Cauchy–Schwarz (`cauchy-schwarz-holder-young.md`)
gives,

  ```
  n² = ( Σᵢ √λᵢ ⋅ 1/√λᵢ )²
     ≤ ( Σᵢ λᵢ ) ( Σᵢ 1/λᵢ )
     = tr(XᵀX) ⋅ tr( (XᵀX)⁻¹ )
  ```

and `tr(XᵀX) = Σₖ ‖xₖ‖² ≤ K r²`.

Hence,

  ```
  tr 𝕍(ĝ) ≥ τ² tr( (XᵀX)⁻¹ ) ≥ τ² n² / (K r²)
  ```

Equality holds iff `XᵀX = (K r²/n) I` (all `λᵢ` equal, budget spent) and
`ĝ` is least squares, `ĝ = (XᵀX)⁻¹ XᵀY`, whose covariance is exactly
`τ² (XᵀX)⁻¹`. So orthogonal designs with equal row norms are A-optimal; e.g.,
`K = n` rows `r ⋅ qₖ` for an orthonormal basis `q₁..qₙ`.

Random designs pay a price. For iid rows `xₖ ~ N(0, σ²Iₙ)`, `XᵀX` is Wishart
and, for `K > n + 1`, `𝔼[ (XᵀX)⁻¹ ] = I / (σ² (K − n − 1))` (inverse-Wishart
mean, cited). With the matched budget `r² = n σ²`,

  ```
  iid Gaussian:  τ² ⋅ n / (σ² (K − n − 1))
  orthogonal:    τ² ⋅ n / (σ² K)
  ```

so the random design needs about `n + 1` extra observations, and diverges as
`K ↓ n + 1`.

Fine print: for `K < n`, no unbiased estimator exists, since `g` and
`g + v` with `Xv = 0` give the same law of `Y`. Other criteria minimize other
functions of `(XᵀX)⁻¹`: D-optimality its determinant, E-optimality its
largest eigenvalue; the orthogonal design is optimal for all three under
this budget. A-, D-, E-optimality are from Kiefer (JRSS-B 1959, cited).

Regularity: the proof needs Gaussian noise for the stated Fisher information
(for other noise, least squares still attains `τ² (XᵀX)⁻¹` among linear
unbiased estimators by Gauss–Markov), `X` fixed and of full column rank, and
Cramér–Rao's conditions, which hold for the Gaussian linear model.
