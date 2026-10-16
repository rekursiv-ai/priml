# Singular Value Decomposition

Every real matrix factors as `UΣVᵀ`. The proof applies the spectral theorem
(see spectral-theorem.md) to the positive semidefinite matrix `AᵀA`.

Let,

  ```
  A ∈ ℝᵐˣⁿ
  AᵀA = V Λ Vᵀ,   VᵀV = I,   Λ ≝ diag(λ₁, …, λₙ),   λ₁ ≥ … ≥ λₙ
  r   ≝ #{ i : λᵢ > 0 }
  σᵢ  ≝ √λᵢ
  ```

where `V = [v₁, …, vₙ]` exists by the spectral theorem since `AᵀA` is symmetric.

First, the eigenvalues are nonnegative, so `σᵢ` is real,

  ```
  λᵢ = vᵢᵀ AᵀA vᵢ
     = ‖Avᵢ‖²
     ≥ 0
  ```

The same line shows `Avᵢ = 0` for `i > r`.

Second, define `uᵢ ≝ Avᵢ / σᵢ` for `i ≤ r`. These are orthonormal,

  ```
  uᵢᵀuⱼ = vᵢᵀ AᵀA vⱼ / (σᵢσⱼ)
        = λⱼ vᵢᵀvⱼ / (σᵢσⱼ)
        = δᵢⱼ
  ```

where the last step uses `vᵢᵀvⱼ = δᵢⱼ` and `λᵢ = σᵢ²`. In particular
`r ≤ min(m, n)`. Extend `u₁, …, u_r` to an orthonormal basis `u₁, …, u_m` of
`ℝᵐ` by Gram–Schmidt, and let `U ≝ [u₁, …, u_m]`.

Third, let `Σ ∈ ℝᵐˣⁿ` have `Σᵢᵢ ≝ σᵢ` for `i ≤ r` and zeros elsewhere. Column
`i` of each side agrees,

  ```
  (AV)ᵢ = Avᵢ
        = σᵢ uᵢ      for i ≤ r
        = 0          for i > r
        = (UΣ)ᵢ
  ```

so `AV = UΣ`. Multiplying on the right by `Vᵀ` and using `VVᵀ = I`,

  ```
  A = AVVᵀ
    = UΣVᵀ
  ```

Hence,

  ```
  A = UΣVᵀ = Σ_{i=1}^{r} σᵢ uᵢ vᵢᵀ,   UᵀU = I,   VᵀV = I,   σ₁ ≥ … ≥ σ_r > 0
  ```

The singular values are unique: `AᵀA = VΣᵀΣVᵀ`, so `σᵢ²` are the eigenvalues of
`AᵀA`. Also `AAᵀuᵢ = σᵢ²uᵢ`, so `U` diagonalizes `AAᵀ`. The pair `(uᵢ, vᵢ)` is
unique up to a joint sign flip iff `σᵢ` is simple and positive.

Regularity: the proof needs only the spectral theorem for real symmetric
matrices and Gram–Schmidt to complete `U`; `A` may be any real `m × n` matrix.
