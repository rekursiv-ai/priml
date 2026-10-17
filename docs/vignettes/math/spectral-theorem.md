# Spectral Theorem

A real symmetric matrix is diagonalized by an orthogonal matrix. The proof
rests on three facts:

1. symmetry forces eigenvalues to be real,
2. symmetry makes eigenvectors of distinct eigenvalues orthogonal, and,
3. the maximizer of `xᵀAx` on the unit sphere is an eigenvector whose
   orthogonal complement is invariant under `A`.

Let,

  ```
  A ∈ ℝⁿˣⁿ,  Aᵀ = A
  S    ≝ { x ∈ ℝⁿ : ‖x‖ = 1 }
  R(x) ≝ xᵀAx
  ```

First, every eigenvalue is real. Let `Av = λv` with `v ∈ ℂⁿ`, `v ≠ 0`, and let
`v̄` be the entrywise conjugate,

  ```
  λ ‖v‖² = v̄ᵀ A v
         = (v̄ᵀ A v)ᵀ
         = vᵀ Aᵀ v̄
         = vᵀ A v̄
         = conj(v̄ᵀ A v)
         = conj(λ) ‖v‖²
  ```

where the second step transposes a scalar, the fourth uses `Aᵀ = A`, and the
fifth uses that `A` is real. Since `‖v‖² > 0`, `λ = conj(λ)`.

Second, eigenvectors of distinct eigenvalues are orthogonal. Let `Au = λu` and
`Av = μv` with `λ ≠ μ`,

  ```
  λ uᵀv = (Au)ᵀ v
        = uᵀ Aᵀ v
        = uᵀ A v
        = μ uᵀv
  ```

so `(λ − μ) uᵀv = 0`, hence `uᵀv = 0`.

Third, a maximizer of `R` on `S` is an eigenvector. `R` is continuous and `S`
is compact, so some `q₁ ∈ S` attains `λ₁ ≝ max_{x∈S} R(x)`. Fix any `y ⊥ q₁`.
Since `‖q₁ + ty‖² = 1 + t²‖y‖²`, maximality of `q₁` gives, for all real `t`,

  ```
  0 ≥ R(q₁ + ty) − λ₁ ‖q₁ + ty‖²
    = λ₁ + 2t yᵀAq₁ + t² yᵀAy − λ₁ (1 + t²‖y‖²)
    = 2t yᵀAq₁ + t² (yᵀAy − λ₁‖y‖²)
  ```

where the first step uses `R(z) ≤ λ₁ ‖z‖²` for all `z` (scale `z` onto `S`),
and the second uses `yᵀAq₁ = q₁ᵀAy` by symmetry. A function `g(t) ≤ 0` with
`g(0) = 0` has `g′(0) = 0`, so `yᵀAq₁ = 0` for every `y ⊥ q₁`. Thus `Aq₁`
lies in `span(q₁)`, i.e., `Aq₁ = c q₁` with `c = q₁ᵀAq₁ = λ₁`.

Fourth, the complement `W = q₁^⊥` is invariant. For `x ∈ W`,

  ```
  q₁ᵀ (Ax) = (Aq₁)ᵀ x
           = λ₁ q₁ᵀx
           = 0
  ```

so `AW ⊆ W`. Let `P ∈ ℝⁿˣ⁽ⁿ⁻¹⁾` have orthonormal columns spanning `W`, so
`PᵀP = I`, `Pᵀq₁ = 0`, and `PPᵀ` is the orthogonal projector onto `W`. Then
`B = PᵀAP` is symmetric of size `n − 1`. By induction on `n` (the case `n = 1`
is trivial), `B = RΛ′Rᵀ` with `R` orthogonal and `Λ′` diagonal.

Fifth, assemble `Q ≝ [q₁, PR]` and `Λ ≝ diag(λ₁, Λ′)`. Its columns are
orthonormal since `(PR)ᵀ(PR) = RᵀPᵀPR = I` and `q₁ᵀPR = 0`. Because the columns
of `AP` lie in `W`, `AP = PPᵀAP = PB`, and so

  ```
  AQ = [Aq₁, APR]
     = [λ₁q₁, PBR]
     = [λ₁q₁, PRΛ′]
     = QΛ
  ```

Hence, since `Q` is square with `QᵀQ = I`,

  ```
  A = QΛQᵀ,   QᵀQ = QQᵀ = I,   Λ = diag(λ₁, …, λₙ) ∈ ℝⁿˣⁿ
  ```

The construction also orders `λ₁ ≥ λ₂ ≥ … ≥ λₙ`, since each `λᵢ` is a maximum
of `R` over a sphere inside the previous one.

The `λᵢ` are unique as the roots of `det(A − λI)`. Each column of `Q` is unique
up to sign iff its eigenvalue is simple; for a repeated eigenvalue any
orthonormal basis of its eigenspace works.

Regularity: the proof needs `A` real and symmetric, compactness of `S`
(Heine–Borel) with continuity of `R` (extreme value theorem), existence of an
orthonormal basis of `W` (Gram–Schmidt), and induction on `n`.
