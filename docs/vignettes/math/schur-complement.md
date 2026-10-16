# Schur Complement

A symmetric block matrix is congruent to a block diagonal of `A` and its Schur
complement. Positivity and the inverse both follow from that factorization.

Let,

  ```
  M ≝ [ A   B ]      A ∈ ℝⁿˣⁿ,  Aᵀ = A,  invertible
      [ Bᵀ  C ]      B ∈ ℝⁿˣᵐ,  C ∈ ℝᵐˣᵐ,  Cᵀ = C
  S ≝ C − Bᵀ A⁻¹ B
  L ≝ [ I       0 ]
      [ BᵀA⁻¹   I ]
  ```

First, `M` has a block LDU factorization. Since `A⁻ᵀ = A⁻¹`,
`Lᵀ = [ I  A⁻¹B ; 0  I ]`, and

  ```
  L [ A  0 ] Lᵀ = [ A    0 ] [ I  A⁻¹B ]
    [ 0  S ]      [ Bᵀ   S ] [ 0  I    ]
                = [ A    B             ]
                  [ Bᵀ   BᵀA⁻¹B + S    ]
                = M
  ```

`L` is unit lower triangular, so `det L = 1` and `det M = det A ⋅ det S`.

Second, complete the square. For `z = (x, y)` with `x ∈ ℝⁿ`, `y ∈ ℝᵐ`,

  ```
  zᵀMz = xᵀAx + 2 xᵀBy + yᵀCy
       = xᵀAx + 2 xᵀBy + yᵀBᵀA⁻¹By + yᵀ(C − BᵀA⁻¹B)y
       = (x + A⁻¹By)ᵀ A (x + A⁻¹By) + yᵀSy
  ```

which is the first step's factorization evaluated at `z`.

Third, `M ≻ 0` implies `A ≻ 0` and `S ≻ 0`. Taking `y = 0`, `x ≠ 0` gives
`xᵀAx = zᵀMz > 0`. Taking `y ≠ 0` and `x = −A⁻¹By` gives

  ```
  yᵀSy = zᵀMz
       > 0
  ```

Fourth, `A ≻ 0` and `S ≻ 0` imply `M ≻ 0`. Both terms of the second step are
nonnegative, so `zᵀMz ≥ 0`. If `zᵀMz = 0` then `yᵀSy = 0`, so `y = 0`; then
`xᵀAx = 0`, so `x = 0`. Hence `zᵀMz > 0` for `z ≠ 0`.

Fifth, the block inverse. With `S` invertible,
`L⁻¹ = [ I  0 ; −BᵀA⁻¹  I ]`, and inverting the first step's factorization,

  ```
  M⁻¹ = L⁻ᵀ [ A⁻¹  0   ] L⁻¹
            [ 0    S⁻¹ ]
      = [ I  −A⁻¹B ] [ A⁻¹  0   ] [ I       0 ]
        [ 0   I    ] [ 0    S⁻¹ ] [ −BᵀA⁻¹  I ]
      = [ A⁻¹  −A⁻¹BS⁻¹ ] [ I       0 ]
        [ 0     S⁻¹     ] [ −BᵀA⁻¹  I ]
      = [ A⁻¹ + A⁻¹BS⁻¹BᵀA⁻¹   −A⁻¹BS⁻¹ ]
        [ −S⁻¹BᵀA⁻¹             S⁻¹      ]
  ```

Hence,

  ```
  M   ≻ 0  ⇔  A ≻ 0  and  C − BᵀA⁻¹B ≻ 0

  M⁻¹ = [ A⁻¹ + A⁻¹BS⁻¹BᵀA⁻¹   −A⁻¹BS⁻¹ ]
        [ −S⁻¹BᵀA⁻¹             S⁻¹      ]
  ```

The same argument with `≥` gives: if `A ≻ 0`, then `M ⪰ 0` iff `S ⪰ 0`. The
lower-right block of `M⁻¹` is `S⁻¹`, so the Schur complement is the inverse of
the corresponding block of `M⁻¹`; for a Gaussian with covariance `M`, `S` is
the conditional covariance of the second block given the first.

Regularity: the factorization and inverse need `A` invertible (and `S`
invertible for `M⁻¹`); the positivity equivalence needs `M` symmetric, and in
the `⇒` direction `A` invertible follows from `A ≻ 0`. Without symmetry, the
LDU and inverse hold with `Bᵀ` replaced by any lower-left block `D`.
