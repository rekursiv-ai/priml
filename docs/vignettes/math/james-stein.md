# James–Stein

James–Stein follows from two facts:

1. Stein's lemma turns the risk into an expectation free of `θ` (SURE), and,
2. in `d ≥ 3` dimensions, `x/‖x‖²` has positive divergence `(d−2)/‖x‖²`.

Let,

  ```
  X ~ N(θ, I_d),   d ≥ 3
  R(θ, δ) ≝ 𝔼_θ[ ‖δ(X) − θ‖² ]
  g(x)    ≝ −(d−2) ⋅ x / ‖x‖²
  δ_JS(X) ≝ X + g(X) = (1 − (d−2)/‖X‖²) X
  ```

First, the identity estimator has risk `d`,

  ```
  R(θ, X) = Σᵢ 𝔼_θ[ (Xᵢ − θᵢ)² ]
          = d
  ```

Second, expand the risk of `X + g(X)`,

  ```
  R(θ, δ_JS) = 𝔼_θ[ ‖X − θ‖² ] + 2 𝔼_θ[ (X − θ)ᵀ g(X) ] + 𝔼_θ[ ‖g(X)‖² ]
             = d + 2 Σᵢ 𝔼_θ[ ∂ᵢ gᵢ(X) ] + 𝔼_θ[ ‖g(X)‖² ]
             = 𝔼_θ[ d + 2 ∇⋅g(X) + ‖g(X)‖² ]
  ```

where the second step applies Stein's lemma (stein-lemma.md) to each
coordinate, `𝔼[(Xᵢ − θᵢ) gᵢ(X)] = 𝔼[∂ᵢ gᵢ(X)]`, holding the other coordinates
fixed. The bracket is Stein's unbiased risk estimate (SURE).

Third, compute the divergence. For `x ≠ 0`,

  ```
  ∂ᵢ (xᵢ / ‖x‖²)    = 1/‖x‖² − 2 xᵢ² / ‖x‖⁴
  Σᵢ ∂ᵢ (xᵢ / ‖x‖²) = d/‖x‖² − 2 ‖x‖²/‖x‖⁴
                    = (d − 2)/‖x‖²
  ```

so `∇⋅g(x) = −(d−2)²/‖x‖²`. Also `‖g(x)‖² = (d−2)² ‖x‖²/‖x‖⁴ = (d−2)²/‖x‖²`.

Fourth, substitute into the second step,

  ```
  R(θ, δ_JS) = 𝔼_θ[ d − 2 (d−2)²/‖X‖² + (d−2)²/‖X‖² ]
             = d − (d−2)² ⋅ 𝔼_θ[ 1/‖X‖² ]
  ```

Fifth, `0 < 𝔼_θ[1/‖X‖²] < ∞` for `d ≥ 3`. Positivity holds since the
integrand is positive. For finiteness, the `N(θ, I_d)` density is at most
`(2π)^{−d/2}`, and `|S^{d−1}|` is the area of the unit sphere,

  ```
  𝔼_θ[ 1/‖X‖² ] ≤ 𝔼_θ[ 1{‖X‖ ≤ 1}/‖X‖² ] + 1
                ≤ (2π)^{−d/2} ∫_{‖x‖≤1} ‖x‖⁻² dx + 1
                = (2π)^{−d/2} |S^{d−1}| ∫₀¹ r^{d−3} dr + 1
                = (2π)^{−d/2} |S^{d−1}| / (d − 2) + 1
  ```

where the first step bounds `1/‖X‖² ≤ 1` on `{‖X‖ > 1}`, and the third uses
polar coordinates. The radial integral is finite iff `d ≥ 3`.

Hence,

  ```
  R(θ, δ_JS) = d − (d−2)² ⋅ 𝔼_θ[ 1/‖X‖² ] < d = R(θ, X)   for every θ
  ```

so `X` is inadmissible under squared loss for `d ≥ 3`. The improvement is
largest at `θ = 0`. There `‖X‖² ~ χ²_d` and `𝔼[1/χ²_d] = 1/(d−2)`, so
`R(0, δ_JS) = d − (d−2) = 2`. For general `θ`, the Poisson-mixture
representation of the noncentral `χ²` (cited) gives the exact form

  ```
  R(θ, δ_JS) = d − (d−2)² ⋅ 𝔼[ 1/(d − 2 + 2K) ],   K ~ Poisson(‖θ‖²/2)
  ```

which increases to `d` as `‖θ‖ → ∞`. The inequality is strict for every `θ`,
so no `θ` gives equality.

Regularity: the proof needs `X` Gaussian with identity covariance (known
variance), `d ≥ 3`, and Stein's lemma for `g`, which is smooth off the origin
(a null set) with `|∂ᵢ gᵢ(x)| ≤ (d−2)/‖x‖²`, integrable by the fifth step.
For `d ≤ 2`, `𝔼[1/‖X‖²] = ∞` and the argument fails.
