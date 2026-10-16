# Matrix Determinant Lemma

The determinant of a low-rank update reduces to a `k × k` determinant. The
proof factors one block matrix two ways and compares determinants.

Let,

  ```
  A ∈ ℝⁿˣⁿ invertible
  U, V ∈ ℝⁿˣᵏ
  M ≝ [ A    −U ]
      [ Vᵀ    I ]
  ```

and use the block-triangular rule: if a square block matrix is block upper or
lower triangular with square diagonal blocks, its determinant is the product
of the diagonal blocks' determinants.

First, factor `M` by eliminating with the lower-right identity,

  ```
  [ I  −U ] [ A + UVᵀ  0 ]   [ A + UVᵀ − UVᵀ   −U ]
  [ 0   I ] [ Vᵀ       I ] = [ Vᵀ               I ]
                           = M
  ```

so, by the block-triangular rule,

  ```
  det M = det(I) ⋅ det(A + UVᵀ) ⋅ det(I)
        = det(A + UVᵀ)
  ```

Second, factor `M` by eliminating with `A`,

  ```
  [ A   0 ] [ I   −A⁻¹U       ]   [ A    −U                     ]
  [ Vᵀ  I ] [ 0   I + VᵀA⁻¹U  ] = [ Vᵀ   −VᵀA⁻¹U + I + VᵀA⁻¹U ]
                                = M
  ```

so, by the block-triangular rule,

  ```
  det M = det(A) ⋅ det(I) ⋅ det(I) ⋅ det(I + VᵀA⁻¹U)
        = det(A) ⋅ det(I + VᵀA⁻¹U)
  ```

Hence, equating the two,

  ```
  det(A + UVᵀ) = det(I + VᵀA⁻¹U) ⋅ det A
  ```

The rank-one case `k = 1` reads `det(A + uvᵀ) = (1 + vᵀA⁻¹u) det A`. With
`A = I` the lemma is Sylvester's identity `det(Iₙ + UVᵀ) = det(Iₖ + VᵀU)`. As
an equality it is exact, so `A + UVᵀ` is singular iff `I + VᵀA⁻¹U` is.

Regularity: the proof needs `A` invertible and the block-triangular
determinant rule (from the Leibniz formula or Laplace expansion). No symmetry
is used, and the identity holds verbatim over `ℂ`.
