# Eckart–Young

Truncated SVD is the best rank-`k` approximation in spectral and Frobenius
norm. The proof rests on two facts:

1. a rank-`k` matrix kills a subspace of dimension `n − k`, which must meet
   the top `k + 1` right singular vectors, and,
2. singular values obey Weyl's inequality (see courant-fischer.md).

Let,

  ```
  A ∈ ℝᵐˣⁿ,   A = Σᵢ σᵢ uᵢ vᵢᵀ,   σ₁ ≥ σ₂ ≥ … ≥ σₙ ≥ 0
  Aₖ ≝ Σ_{i≤k} σᵢ uᵢ vᵢᵀ
  B ∈ ℝᵐˣⁿ,   rank B ≤ k < n
  ```

where the SVD exists by svd.md, `uᵢ` and `vᵢ` are orthonormal, and `σᵢ = 0`
pads indices beyond `min(m, n)`. Write `σᵢ(M)` for the singular values of `M`,
i.e., `σᵢ(M)² = λᵢ(MᵀM)`.

First, both norms are read off the singular values. For `M = UΣVᵀ`,

  ```
  ‖M‖_F² = tr(MᵀM)
         = tr(V ΣᵀΣ Vᵀ)
         = tr(ΣᵀΣ)
         = Σᵢ σᵢ(M)²
  ```

and `‖M‖₂² = max_{‖x‖=1} xᵀMᵀMx = λ₁(MᵀM) = σ₁(M)²` by Courant–Fischer with
`k = 1`. Since `A − Aₖ = Σ_{i>k} σᵢuᵢvᵢᵀ` is itself an SVD,

  ```
  ‖A − Aₖ‖₂ = σₖ₊₁,     ‖A − Aₖ‖_F² = Σ_{i>k} σᵢ²
  ```

Second, the spectral-norm bound. `null(B)` has dimension at least `n − k` and
`span(v₁, …, vₖ₊₁)` has dimension `k + 1`. Their dimensions sum to at least
`n + 1`, so some unit `x` lies in both. Then `Bx = 0` and
`x = Σ_{i≤k+1} (vᵢᵀx) vᵢ`, so

  ```
  ‖A − B‖₂² ≥ ‖(A − B)x‖²
            = ‖Ax‖²
            = ‖Σ_{i≤k+1} σᵢ (vᵢᵀx) uᵢ‖²
            = Σ_{i≤k+1} σᵢ² (vᵢᵀx)²
            ≥ σₖ₊₁² Σ_{i≤k+1} (vᵢᵀx)²
            = σₖ₊₁²
  ```

where the fourth line uses orthonormal `uᵢ` and the last uses `‖x‖ = 1`. Terms
with `σᵢ = 0` vanish, so `uᵢ` is needed only for `i ≤ min(m, n)`.

Third, Weyl's inequality for singular values. For `X, Y ∈ ℝᵐˣⁿ` and
`i + j − 1 ≤ n`,

  ```
  σ_{i+j−1}(X + Y) ≤ σᵢ(X) + σⱼ(Y)
  ```

Let `Tᵢ(X) ≝ span(vᵢ(X), …, vₙ(X))`, the span of the trailing eigenvectors of
`XᵀX`. On it `‖Xx‖² = xᵀXᵀXx ≤ σᵢ(X)²‖x‖²` by Courant–Fischer's first
step. Define `Tⱼ(Y)` likewise. Their intersection has dimension at least
`(n − i + 1) + (n − j + 1) − n = n − (i + j − 1) + 1`, so it contains a
subspace `W` of exactly that dimension. For unit `x ∈ W`,

  ```
  ‖(X + Y)x‖ ≤ ‖Xx‖ + ‖Yx‖
             ≤ σᵢ(X) + σⱼ(Y)
  ```

by the triangle inequality. Courant–Fischer on `(X + Y)ᵀ(X + Y)` in min-max
form gives `σ_{i+j−1}(X + Y)² ≤ max_{x∈W, ‖x‖=1} ‖(X + Y)x‖²`, which proves
it.

Fourth, the Frobenius bound. Apply Weyl with `X = A − B`, `Y = B`, `j = k + 1`,
and use `σₖ₊₁(B) = 0` since `rank B ≤ k`. For `1 ≤ i ≤ n − k`,

  ```
  σᵢ₊ₖ(A) ≤ σᵢ(A − B) + σₖ₊₁(B)
          = σᵢ(A − B)
  ```

Summing squares,

  ```
  ‖A − B‖_F² = Σ_{i=1}^{n} σᵢ(A − B)²
             ≥ Σ_{i=1}^{n−k} σᵢ(A − B)²
             ≥ Σ_{i=1}^{n−k} σᵢ₊ₖ(A)²
             = Σ_{i>k} σᵢ²
  ```

Hence, for every `B` with `rank B ≤ k`,

  ```
  ‖A − B‖₂  ≥ σₖ₊₁ = ‖A − Aₖ‖₂
  ‖A − B‖_F ≥ (Σ_{i>k} σᵢ²)^{1/2} = ‖A − Aₖ‖_F
  ```

`Aₖ` attains both bounds. If `σₖ = σₖ₊₁ > 0`, swapping `uₖvₖᵀ` for
`uₖ₊₁vₖ₊₁ᵀ` in `Aₖ` gives a second minimizer of both. The spectral minimizer
is not unique even with distinct `σᵢ`: for `A = diag(2, 1)` and `k = 1`,
every `B = diag(2 + t, 0)` with `|t| ≤ 1` has `‖A − B‖₂ = 1`.

Regularity: the proof needs only the SVD and Courant–Fischer; `A` is any real
`m × n` matrix. If `k ≥ rank A`, `Aₖ = A` and both bounds are zero.
