# Power Iteration

Power iteration converges to the dominant eigenvector at rate `|λ₂/λ₁|ᵏ`.
The proof rests on one fact: `Aᵏ` scales each eigen-component by `λᵢᵏ`, so
after dividing by `λ₁ᵏ` every other component decays geometrically.

Let,

  ```
  A ∈ ℂⁿˣⁿ,   A xᵢ = λᵢ xᵢ,   x₁, …, xₙ linearly independent,   ‖xᵢ‖ = 1
  |λ₁| > |λ₂| ≥ |λ₃| ≥ … ≥ |λₙ|
  b₀   = Σᵢ cᵢ xᵢ,   c₁ ≠ 0
  bₖ   ≝ Aᵏ b₀ / ‖Aᵏ b₀‖
  ρ    ≝ |λ₂ / λ₁| < 1
  ```

where `‖·‖` is the Euclidean norm, the `xᵢ` exist because `A` is
diagonalizable, and `λ₁ ≠ 0` because `|λ₁| > |λ₂| ≥ 0`.

First, expand the iterate in the eigenbasis,

  ```
  Aᵏ b₀ = Σᵢ cᵢ λᵢᵏ xᵢ
        = c₁ λ₁ᵏ ( x₁ + Σ_{i≥2} (cᵢ/c₁) (λᵢ/λ₁)ᵏ xᵢ )
        = c₁ λ₁ᵏ ( x₁ + rₖ )
  ```

This is nonzero since `c₁λ₁ᵏ ≠ 0` and the `xᵢ` are independent, so `bₖ` is
defined.

Second, the residual decays at rate `ρ`. With `C ≝ Σ_{i≥2} |cᵢ/c₁|`,

  ```
  ‖rₖ‖ ≤ Σ_{i≥2} |cᵢ/c₁| |λᵢ/λ₁|ᵏ ‖xᵢ‖
       ≤ Σ_{i≥2} |cᵢ/c₁| ρᵏ
       = C ρᵏ
  ```

by the triangle inequality, `|λᵢ| ≤ |λ₂|`, and `‖xᵢ‖ = 1`.

Third, the direction converges. Write `bₖ = sₖ (x₁ + rₖ) / ‖x₁ + rₖ‖` with
`sₖ ≝ c₁λ₁ᵏ / |c₁λ₁ᵏ|`, so `|sₖ| = 1`. Choose `α ≝ sₖ / ‖x₁ + rₖ‖`. Once
`Cρᵏ ≤ 1/2`, the reverse triangle inequality gives
`‖x₁ + rₖ‖ ≥ 1 − ‖rₖ‖ ≥ 1/2`, and

  ```
  min_{β∈ℂ} ‖bₖ − β x₁‖ ≤ ‖bₖ − α x₁‖
                        = ‖sₖ rₖ‖ / ‖x₁ + rₖ‖
                        ≤ ‖rₖ‖ / (1 − ‖rₖ‖)
                        ≤ 2 C ρᵏ
  ```

Fourth, the Rayleigh quotient converges to `λ₁` at the same rate. Let
`μₖ ≝ bₖ*Abₖ` and write `bₖ = αx₁ + eₖ` with `‖eₖ‖ ≤ 2Cρᵏ` from the third step.
Since `(A − λ₁I)x₁ = 0` and `‖bₖ‖ = 1`,

  ```
  |μₖ − λ₁| = |bₖ* (A − λ₁I) bₖ|
            ≤ ‖(A − λ₁I) bₖ‖
            = ‖(A − λ₁I) eₖ‖
            ≤ ‖A − λ₁I‖₂ ⋅ 2 C ρᵏ
  ```

where the second line is Cauchy–Schwarz.

Hence, for all `k` with `Cρᵏ ≤ 1/2`,

  ```
  dist(bₖ, span(x₁)) ≤ 2 C |λ₂/λ₁|ᵏ
  |μₖ − λ₁|          ≤ 2 C ‖A − λ₁I‖₂ |λ₂/λ₁|ᵏ
  ```

Convergence is in direction: `bₖ` itself carries the phase `sₖ`, which
rotates unless `λ₁ > 0`. The rate is sharp when `c₂ ≠ 0` and `|λ₂| > |λ₃|`,
since then `rₖ = (c₂/c₁)(λ₂/λ₁)ᵏ x₂ + o(ρᵏ)`.

Regularity: the proof needs `A` diagonalizable (so `b₀` expands in
eigenvectors), a strictly dominant `|λ₁|`, and `c₁ ≠ 0`. The constant `C`
grows with the conditioning of the eigenbasis; for symmetric `A` the `xᵢ` are
orthonormal and `C ≤ √(n − 1) ‖b₀‖ / |c₁|`.
