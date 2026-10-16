# Mercer's Theorem

Mercer's theorem says a continuous PSD kernel is an inner product of feature
maps built from its eigenfunctions. It follows from two facts:

1. the spectral theorem diagonalizes the kernel's integral operator, and,
2. positive semidefiniteness makes every eigenvalue nonnegative, so its
   square root is real.

Statement. Let `X` be a compact metric space, `μ` a finite Borel measure on
`X` with full support, and `k: X × X → ℝ` continuous, symmetric, and PSD,
i.e., `Σᵢⱼ cᵢ cⱼ k(xᵢ, xⱼ) ≥ 0` for all finite `x₁..x_m` and `c ∈ ℝᵐ`. Then
there are `λ₁ ≥ λ₂ ≥ … ≥ 0` and continuous `φⱼ`, orthonormal in `L²(μ)`, with

  ```
  k(x, x′) = Σⱼ λⱼ φⱼ(x) φⱼ(x′)
  ```

converging absolutely and uniformly on `X × X`. So `Φ(x) ≝ (√λⱼ φⱼ(x))ⱼ ∈ ℓ²`
satisfies `k(x, x′) = ⟨Φ(x), Φ(x′)⟩`.

The finite case is proved in full. The continuous case is sketched; the
spectral theorem and the uniform-convergence step are cited.

Let, for the finite case,

  ```
  X   ≝ {x₁..x_m}
  Kᵢₗ ≝ k(xᵢ, xₗ)            Gram matrix, symmetric, PSD
  ```

First, diagonalize. The spectral theorem for real symmetric matrices gives an
orthogonal `U` (columns `u₁..u_m`) and real `Λ = diag(λ₁..λ_m)` with,

  ```
  K = U Λ Uᵀ
    = Σⱼ λⱼ uⱼ uⱼᵀ
  ```

Second, eigenvalues are nonnegative. Using `K uⱼ = λⱼ uⱼ` and `‖uⱼ‖₂ = 1`,

  ```
  λⱼ = λⱼ ⋅ uⱼᵀ uⱼ
     = uⱼᵀ K uⱼ
     ≥ 0
  ```

where the last step is PSD with `c = uⱼ`.

Third, define `Φ(xᵢ) ≝ (√λ₁ u_{i1} .. √λ_m u_{im}) ∈ ℝᵐ`, real by the second
step. Then,

  ```
  ⟨Φ(xᵢ), Φ(xₗ)⟩ = Σⱼ √λⱼ u_{ij} ⋅ √λⱼ u_{lj}
                 = Σⱼ λⱼ u_{ij} u_{lj}
                 = (U Λ Uᵀ)ᵢₗ
                 = Kᵢₗ
  ```

Hence, in the finite case,

  ```
  k(xᵢ, xₗ) = ⟨Φ(xᵢ), Φ(xₗ)⟩,     Φ(xᵢ)ⱼ = √λⱼ u_{ij}
  ```

This is Mercer with `μ` the counting measure: the integral operator is `K`
itself and `φⱼ(xᵢ) = u_{ij}`. Conversely, any `Φ` gives a PSD Gram matrix,
since `cᵀ K c = ‖Σᵢ cᵢ Φ(xᵢ)‖² ≥ 0`, so PSD is exactly the condition for a
feature map to exist.

Sketch of the continuous case. Let,

  ```
  (T f)(x) ≝ ∫ k(x, x′) f(x′) dμ(x′),     T: L²(μ) → L²(μ)
  ```

Fourth, `T` meets the spectral theorem's hypotheses. It is self-adjoint by
symmetry of `k` and Fubini. It is compact because
`k ∈ L²(μ ⊗ μ)` (bounded on a finite measure), so `T` is Hilbert–Schmidt. It
is positive, `⟨f, T f⟩ ≥ 0`, because the double integral is a limit of PSD
quadratic forms `Σᵢₗ cᵢ cₗ k(xᵢ, xₗ)` (cited; uses continuity of `k`).
The spectral theorem for compact self-adjoint operators (cited) gives an
orthonormal basis `{φⱼ}` of `(ker T)^⊥` with `T φⱼ = λⱼ φⱼ`, `λⱼ > 0`,
`λⱼ → 0`. Each `φⱼ = T φⱼ / λⱼ` is continuous, since `T` maps `L²` into
continuous functions when `k` is uniformly continuous.

Fifth, expand each kernel section (proved). Fix `x`. For `g ∈ ker T`, `T g`
is continuous and zero `μ`-a.e., hence zero everywhere by full support, so
`⟨k(x, ·), g⟩ = (T g)(x) = 0`. So `k(x, ·) ∈ (ker T)^⊥`, and its coefficients
in the basis are,

  ```
  ⟨k(x, ·), φⱼ⟩ = ∫ k(x, x′) φⱼ(x′) dμ(x′)
                = (T φⱼ)(x)
                = λⱼ φⱼ(x)
  ```

So `k(x, ·) = Σⱼ λⱼ φⱼ(x) φⱼ` in `L²(μ)`, for every `x`.

Sixth, upgrade to absolute and uniform convergence (cited). The remainder
kernel `k − Σ_{j ≤ J} λⱼ φⱼ ⊗ φⱼ` is the kernel of the positive operator
`T − Σ_{j ≤ J} λⱼ φⱼ⟨φⱼ, ·⟩`, so it is continuous PSD and nonnegative on the
diagonal. Hence `Σⱼ λⱼ φⱼ(x)² ≤ k(x, x)`, and Cauchy–Schwarz gives absolute
convergence off the diagonal. Dini's theorem on the compact diagonal makes
the convergence uniform.

Hence,

  ```
  k(x, x′) = Σⱼ λⱼ φⱼ(x) φⱼ(x′) = ⟨Φ(x), Φ(x′)⟩_{ℓ²}
  ```

The feature map is finite dimensional iff `T` has finite rank, which always
holds in the finite case (rank `≤ m`). The `λⱼ` depend on `μ`; the RKHS does
not.

Regularity: the finite case needs only `K` symmetric PSD. The continuous case
needs `X` compact, `k` continuous symmetric PSD, and `μ` finite with full
support (used in the fifth step to pass from a.e. to everywhere).
