# Control Variates

Control variates follow from one fact: subtracting a zero-mean quantity keeps
an estimator unbiased, and the best multiple to subtract is the regression
coefficient, which removes the explained part of the variance.

Let,

  ```
  Y ∈ ℝ            an unbiased estimator of μ_Y,  𝕍(Y) < ∞
  C ∈ ℝᵐ           a control with known mean μ_C,  𝕍(C) ≻ 0
  c ∈ ℝᵐ           a fixed coefficient
  Y_c ≝ Y − cᵀ (C − μ_C)
  Σ_CY ≝ Cov(C, Y) ∈ ℝᵐ
  ```

First, `Y_c` is unbiased for every `c`, since `𝔼[C − μ_C] = 0`.

Second, its variance is a quadratic in `c`,

  ```
  𝕍(Y_c) = 𝕍(Y) − 2 cᵀ Σ_CY + cᵀ 𝕍(C) c
  ```

Third, minimize. The gradient `−2Σ_CY + 2𝕍(C)c` vanishes at
`c* = 𝕍(C)⁻¹ Σ_CY`, and `𝕍(C) ≻ 0` makes it the minimum,

  ```
  𝕍(Y_{c*}) = 𝕍(Y) − 2 Σ_CYᵀ 𝕍(C)⁻¹ Σ_CY + Σ_CYᵀ 𝕍(C)⁻¹ 𝕍(C) 𝕍(C)⁻¹ Σ_CY
            = 𝕍(Y) − Σ_CYᵀ 𝕍(C)⁻¹ Σ_CY
  ```

Hence,

  ```
  c*        = 𝕍(C)⁻¹ Cov(C, Y)
  𝕍(Y_{c*}) = 𝕍(Y) ⋅ (1 − ρ²),   ρ² ≝ Σ_CYᵀ 𝕍(C)⁻¹ Σ_CY / 𝕍(Y)
  ```

`ρ²` is the squared multiple correlation of `Y` on `C`. Equality
`𝕍(Y_{c*}) = 0` holds iff `Y` is affine in `C` almost surely. This is the same
projection as the third step of Cramér–Rao (`cramer-rao.md`), with the score
as the control.

Example, the REINFORCE baseline. For the score estimator `f(X) s(X)`, with
`s ≝ ∇_θ log q_θ` (`reparameterization-vs-score.md`), the control `C = s` has
mean 0, and subtracting `b ⋅ s` gives `(f − b) s`. For a vector estimator and
scalar `b`, minimizing `𝔼‖(f − b) s‖²`,

  ```
  𝔼‖(f − b) s‖² = 𝔼[ f² ‖s‖² ] − 2b 𝔼[ f ‖s‖² ] + b² 𝔼‖s‖²
  b*            = 𝔼[ f ‖s‖² ] / 𝔼‖s‖²
  ```

so the optimal baseline is the mean of `f` weighted by `‖s‖²`, not the plain
mean (Greensmith, Bartlett, Baxter, JMLR 2004, cited).

Example, a gradient guess as a control. For a vector estimator `ĝ` of `g`
and any guess `h` with known expectation, `ĝ − (ĥ − 𝔼ĥ)` stays unbiased.
MuProp (Gu et al., arXiv:1511.05176) uses a first-order Taylor expansion as
the control for the score estimator.

Fine print: in practice `c*` is estimated from the same samples, which adds
an `O(1/K)` bias over `K` samples; splitting samples removes it. Multiple
controls are multiple regression; adding controls never increases the
variance at the optimum, since `ρ²` is nondecreasing in the set of controls.

Regularity: the proof needs `𝕍(Y) < ∞`, `𝕍(C)` finite and invertible, `μ_C`
known exactly, and `c` fixed (not estimated from the same draw).
