# Woodbury Identity

The Woodbury identity inverts a low-rank update of `A` using only `A⁻¹` and a
small `k × k` inverse. The proof is direct multiplication.

Let,

  ```
  A ∈ ℝⁿˣⁿ invertible
  C ∈ ℝᵏˣᵏ invertible
  U ∈ ℝⁿˣᵏ,   V ∈ ℝᵏˣⁿ
  S ≝ C⁻¹ + V A⁻¹ U ∈ ℝᵏˣᵏ invertible
  ```

The claim is

  ```
  (A + UCV)⁻¹ = A⁻¹ − A⁻¹ U S⁻¹ V A⁻¹
  ```

First, multiply `A + UCV` by the candidate inverse,

  ```
  (A + UCV)(A⁻¹ − A⁻¹US⁻¹VA⁻¹)
      = I − US⁻¹VA⁻¹ + UCVA⁻¹ − UCVA⁻¹US⁻¹VA⁻¹
      = I + UCVA⁻¹ − U (I + CVA⁻¹U) S⁻¹VA⁻¹
      = I + UCVA⁻¹ − UC (C⁻¹ + VA⁻¹U) S⁻¹VA⁻¹
      = I + UCVA⁻¹ − UC S S⁻¹ VA⁻¹
      = I + UCVA⁻¹ − UCVA⁻¹
      = I
  ```

where the second step factors `U` on the left and `S⁻¹VA⁻¹` on the right of
the second and fourth terms, and the third uses `I = CC⁻¹`.

Second, a right inverse of a square matrix is a two-sided inverse, since
`MN = I` gives `det M ≠ 0`, so `M⁻¹` exists and `N = M⁻¹MN = M⁻¹`. Hence
`A + UCV` is invertible and

  ```
  (A + UCV)⁻¹ = A⁻¹ − A⁻¹ U (C⁻¹ + V A⁻¹ U)⁻¹ V A⁻¹
  ```

Invertibility of `S` is also necessary. By determinant-lemma.md with
`UC` in place of `U` and `Vᵀ` in place of `V`,

  ```
  det(A + UCV) = det(A) det(I + VA⁻¹UC)
               = det(A) det((C⁻¹ + VA⁻¹U) C)
               = det(A) det(S) det(C)
  ```

so, given `A` and `C` invertible, `A + UCV` is invertible iff `S` is.

Sherman–Morrison is the case `k = 1`, `C = 1`, `U = u`, `V = vᵀ`,

  ```
  (A + uvᵀ)⁻¹ = A⁻¹ − A⁻¹uvᵀA⁻¹ / (1 + vᵀA⁻¹u)
  ```

valid iff `1 + vᵀA⁻¹u ≠ 0`.

Regularity: the proof needs `A`, `C`, and `S` invertible; no symmetry or
definiteness is used, and the identity holds verbatim over `ℂ`.
