# Antithetic and Orthogonal Perturbations

Zeroth-order gradient estimates improve by two facts:

1. antithetic pairs `±ξ` cancel the even-order terms of `R`, so curvature
   adds no variance, and,
2. orthogonal directions remove the cross terms between directions, so `n`
   antithetic pairs recover `∇R` exactly for quadratic `R`.

Let,

  ```
  R(ξ) = R₀ + gᵀξ + ½ ξᵀHξ           ξ ∈ ℝⁿ, H symmetric
  ξ ~ N(0, σ²Iₙ),   z ≝ ξ/σ
  Q = [q₁ … qₙ]                      Haar-random orthogonal
  P_K ≝ Σ_{k≤K} qₖqₖᵀ                random rank-K projection

  ĝ_b   ≝ (R(ξ) − R₀) ξ / σ²                               # baseline, 1 eval + R₀
  ĝ_a   ≝ (R(ξ) − R(−ξ)) ξ / (2σ²)                         # antithetic, 2 evals
  ĝ_o,K ≝ (1/K) Σ_{k≤K} (R(ξₖ) − R(−ξₖ)) ξₖ / (2σ²),
          ξₖ ≝ σ√n ⋅ qₖ                                    # orthogonal, 2K evals
  ```

`ξₖ` has norm `σ√n` so `𝔼[ ξₖξₖᵀ ] = σ²I`, matching the Gaussian.

First, all three are unbiased for `g`. For `ĝ_b`, the curvature term
`½ (ξᵀHξ) ξ` has odd degree, so its mean is zero,

  ```
  𝔼[ ĝ_b ] = 𝔼[ (gᵀξ) ξ ]/σ² + ½ 𝔼[ (ξᵀHξ) ξ ]/σ²
           = g + 0
  ```

For `ĝ_a` and `ĝ_o,K`, `R(ξ) − R(−ξ) = 2gᵀξ` exactly, so `ĝ_a = (gᵀξ) ξ/σ²`,
and `ĝ_o,K = (n/K) P_K g` with `𝔼[P_K] = (K/n) I`, since `P_K` has trace `K`
and a rotation-invariant law.

Second, the baseline estimator pays for curvature. The cross term has odd
degree and vanishes, so

  ```
  𝔼‖ĝ_b‖² = 𝔼[ (gᵀz)² ‖z‖² ] + ¼ σ² 𝔼[ (zᵀHz)² ‖z‖² ]
          = (n + 2) ‖g‖² + ¼ σ² (n + 4) ((tr H)² + 2‖H‖_F²)
  ```

where the first term sums `𝔼[(gᵀz)² zᵢ²] = ‖g‖² + 2gᵢ²` over `i`, and the
second writes `z = r u` with `r² ~ χ²ₙ` independent of `u`, so
`𝔼[(zᵀHz)²‖z‖²] = (𝔼r⁶/𝔼r⁴) ⋅ 𝔼[(zᵀHz)²]`, with
`𝔼r⁶/𝔼r⁴ = n(n+2)(n+4)/(n(n+2)) = n + 4` and
`𝔼[(zᵀHz)²] = (tr H)² + 2‖H‖_F²` (diagonalize `H`; use `𝔼zᵢ⁴ = 3`).

Third, antithetic pays only for the linear term,

  ```
  tr 𝕍(ĝ_a) = 𝔼[ (gᵀz)² ‖z‖² ] − ‖g‖²
            = (n + 1) ‖g‖²
  ```

Fourth, orthogonal directions remove the cross terms. Since `P_K² = P_K`,

  ```
  tr 𝕍(ĝ_o,K) = (n/K)² 𝔼[ gᵀP_K g ] − ‖g‖²
              = (n/K)² (K/n) ‖g‖² − ‖g‖²
              = (n/K − 1) ‖g‖²
  ```

Hence, for quadratic `R`,

  ```
  tr 𝕍(ĝ_b)    = (n + 1) ‖g‖² + ¼ σ² (n + 4) ((tr H)² + 2‖H‖_F²)
  tr 𝕍(ĝ_a)    = (n + 1) ‖g‖²
  K iid ĝ_a    : (n + 1) ‖g‖² / K
  tr 𝕍(ĝ_o,K)  = (n/K − 1) ‖g‖²
  ```

At `K = n`, `P_n = I` and `ĝ_o,n = g` exactly: finite differences along a
random orthonormal basis. At equal cost, orthogonal beats iid antithetic for
every `K`, since `n − K < n + 1`.

Fine print: at equal evaluations, antithetic is not free. `2K` baseline
samples give `(n + 1)‖g‖²/(2K)` plus the curvature term, half the linear
variance of `K` antithetic pairs; antithetic wins only when the `H` term
dominates. For non-quadratic `R`, antithetic cancels every even-order Taylor
term, and `ĝ_o,K` estimates the gradient of `R` smoothed over the sphere of
radius `σ√n` rather than the Gaussian. Structured random orthogonal
perturbations for evolution strategies: Choromański et al.
(arXiv:1804.02395, cited).

Regularity: the exact formulas assume quadratic `R` (else they hold to the
order of the Taylor remainder), `ξ` Gaussian for `ĝ_b` and `ĝ_a`, and `Q`
Haar-distributed for `𝔼[P_K] = (K/n) I`.
