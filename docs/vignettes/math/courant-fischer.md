# Courant–Fischer

The `k`-th eigenvalue of a symmetric matrix is a max-min of the Rayleigh
quotient. The proof rests on two facts:

1. on a span of eigenvectors, the Rayleigh quotient is a weighted average of
   their eigenvalues, and,
2. two subspaces of `ℝⁿ` whose dimensions sum past `n` share a nonzero vector.

Let,

  ```
  A ∈ ℝⁿˣⁿ,  Aᵀ = A
  A    = Σᵢ λᵢ qᵢqᵢᵀ,   qᵢᵀqⱼ = δᵢⱼ,   λ₁ ≥ λ₂ ≥ … ≥ λₙ
  R(x) ≝ xᵀAx / xᵀx,   x ≠ 0
  Sₖ   ≝ span(q₁, …, qₖ)          dim Sₖ = k
  Tₖ   ≝ span(qₖ, …, qₙ)          dim Tₖ = n − k + 1
  ```

where the eigendecomposition exists by spectral-theorem.md. The claim is

  ```
  λₖ = max_{dim V = k} min_{x∈V, x≠0} R(x) = min_{dim V = n−k+1} max_{x∈V, x≠0} R(x)
  ```

First, bound `R` on `Sₖ` and `Tₖ`. Write `x = Σᵢ cᵢqᵢ`,

  ```
  R(x) = Σᵢ λᵢ cᵢ² / Σᵢ cᵢ²
  ```

On `Sₖ` only `i ≤ k` appear and each `λᵢ ≥ λₖ`, so `R(x) ≥ λₖ`. On `Tₖ` only
`i ≥ k` appear and each `λᵢ ≤ λₖ`, so `R(x) ≤ λₖ`. Both bounds are attained at
`x = qₖ`.

Second, any two subspaces with `dim V + dim T ≥ n + 1` intersect nontrivially,

  ```
  dim(V ∩ T) = dim V + dim T − dim(V + T)
             ≥ n + 1 − n
             = 1
  ```

Third, the max-min is at least `λₖ`. Take `V = Sₖ`,

  ```
  max_{dim V = k} min_{x∈V} R(x) ≥ min_{x∈Sₖ} R(x)
                                 = λₖ
  ```

Fourth, the max-min is at most `λₖ`. For any `V` with `dim V = k`,
`dim V + dim Tₖ = n + 1`, so some `x ≠ 0` lies in `V ∩ Tₖ`,

  ```
  min_{y∈V} R(y) ≤ R(x)
                 ≤ λₖ
  ```

Fifth, the min-max follows the same way with roles swapped. Taking `V = Tₖ`
gives `max_{x∈Tₖ} R(x) = λₖ`, so the min-max is at most `λₖ`. For any `V`
with `dim V = n − k + 1`, some `x ≠ 0` lies in `V ∩ Sₖ`,

  ```
  max_{y∈V} R(y) ≥ R(x)
                 ≥ λₖ
  ```

Hence,

  ```
  λₖ = max_{dim V = k} min_{x∈V, ‖x‖=1} xᵀAx = min_{dim V = n−k+1} max_{x∈V, ‖x‖=1} xᵀAx
  ```

The outer max is attained at `V = Sₖ` and the outer min at `V = Tₖ`. The
`k = 1` and `k = n` cases are `λ₁ = max R` and `λₙ = min R`.

Corollary (Weyl). Let `B` be symmetric too, and `i + j − 1 ≤ n`. Then

  ```
  λ_{i+j−1}(A + B) ≤ λᵢ(A) + λⱼ(B)
  ```

To prove it, let `Tᵢ(A)` and `Tⱼ(B)` be the eigenvector spans above for each
matrix, and `k = i + j − 1`. By the second step's formula,

  ```
  dim(Tᵢ(A) ∩ Tⱼ(B)) ≥ (n − i + 1) + (n − j + 1) − n
                     = n − k + 1
  ```

so the intersection contains a subspace `W` with `dim W = n − k + 1`. For unit
`x ∈ W`, by the first step applied to `A` and to `B`,

  ```
  xᵀ(A + B)x = xᵀAx + xᵀBx
             ≤ λᵢ(A) + λⱼ(B)
  ```

and the min-max form gives `λₖ(A + B) ≤ max_{x∈W, ‖x‖=1} xᵀ(A + B)x`. Setting
`j = 1`, and applying the result again to `A = (A + B) + (−B)`, yields

  ```
  λᵢ(A) + λₙ(B) ≤ λᵢ(A + B) ≤ λᵢ(A) + λ₁(B)
  ```

so `|λᵢ(A + B) − λᵢ(A)| ≤ ‖B‖₂`, using `λ₁(−B) = −λₙ(B)`.

Regularity: the proof needs `A` and `B` real symmetric (for the orthonormal
eigenbasis), the dimension formula for sums of subspaces, and compactness of
the unit sphere in `V` so each inner min and max is attained.
