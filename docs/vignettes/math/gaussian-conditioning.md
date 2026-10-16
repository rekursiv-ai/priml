# Gaussian Conditioning

Gaussian conditioning follows from two facts:

1. subtracting the linear regression of `x_a` on `x_b` leaves a residual
   uncorrelated with `x_b`, and,
2. jointly Gaussian variables that are uncorrelated are independent.

Let,

  ```
  x ≝ (x_a, x_b) ~ N(μ, Σ)
  μ ≝ ⎡ μ_a ⎤      Σ ≝ ⎡ Σ_aa  Σ_ab ⎤      Σ_ba = Σ_abᵀ,  Σ_bb ≻ 0
      ⎣ μ_b ⎦          ⎣ Σ_ba  Σ_bb ⎦
  K ≝ Σ_ab Σ_bb⁻¹
  z ≝ x_a − K x_b
  ```

First, the residual `z` is uncorrelated with `x_b`,

  ```
  𝕍(z, x_b) = 𝕍(x_a, x_b) − K 𝕍(x_b, x_b)
            = Σ_ab − Σ_ab Σ_bb⁻¹ Σ_bb
            = 0
  ```

Second, `(z, x_b)` is a linear map of `x`, so it is jointly Gaussian with
block-diagonal covariance `diag(C, Σ_bb)`, where `C ≝ 𝕍(z)`. Its
characteristic function factors,

  ```
  𝔼[ e^{i(sᵀz + tᵀx_b)} ] = exp( i sᵀ𝔼z + i tᵀμ_b − (sᵀCs + tᵀΣ_bb t)/2 )
                          = exp( i sᵀ𝔼z − sᵀCs/2 ) ⋅ exp( i tᵀμ_b − tᵀΣ_bb t/2 )
                          = 𝔼[ e^{isᵀz} ] ⋅ 𝔼[ e^{itᵀx_b} ]
  ```

where the first step uses that the cross term `2 sᵀ 𝕍(z, x_b) t` is zero.
A factored characteristic function means `z` and `x_b` are independent.

Third, the mean and covariance of `z`,

  ```
  𝔼[z] = μ_a − K μ_b
  C    = Σ_aa − K Σ_ba − Σ_ab Kᵀ + K Σ_bb Kᵀ
       = Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba − Σ_ab Σ_bb⁻¹ Σ_ba + Σ_ab Σ_bb⁻¹ Σ_bb Σ_bb⁻¹ Σ_ba
       = Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba
  ```

where the second step uses `Kᵀ = Σ_bb⁻¹ Σ_ba`, since `Σ_bb` is symmetric.

Fourth, condition. Write `x_a = z + K x_b`. Given `x_b = b`, the term `Kb`
is a constant, and `z` keeps its marginal law by independence,

  ```
  x_a | x_b = b  ~  N( 𝔼[z] + K b, C )
                 =  N( μ_a − K μ_b + K b, C )
                 =  N( μ_a + K (b − μ_b), C )
  ```

Hence,

  ```
  μ_{a|b} = μ_a + Σ_ab Σ_bb⁻¹ (b − μ_b)
  Σ_{a|b} = Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba
  ```

`Σ_{a|b}` is the Schur complement of `Σ_bb` in `Σ`. It does not depend on
`b`, and `Σ_{a|b} ⪯ Σ_aa`, since `Σ_ab Σ_bb⁻¹ Σ_ba ⪰ 0`. In precision form,
with `Λ ≝ Σ⁻¹`, block inversion gives `Λ_aa = Σ_{a|b}⁻¹` and
`Λ_ab = −Λ_aa Σ_ab Σ_bb⁻¹`, so,

  ```
  Σ_{a|b} = Λ_aa⁻¹
  μ_{a|b} = μ_a − Λ_aa⁻¹ Λ_ab (b − μ_b)
  ```

The conditional mean is the best linear predictor, and here it is also the
best predictor of any kind, `𝔼[x_a | x_b]`.

Regularity: the proof needs `x` jointly Gaussian (marginal Gaussianity of
`x_a` and `x_b` is not enough) and `Σ_bb` invertible. For singular `Σ_bb`, the
same formulas hold with the Moore–Penrose pseudoinverse `Σ_bb⁺`. The precision
form also needs `Σ` invertible.
