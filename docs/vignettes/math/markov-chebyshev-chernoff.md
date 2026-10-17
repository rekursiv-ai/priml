# Markov, Chebyshev, Chernoff

All three bounds follow from one fact: an indicator lies below any
nonnegative function that is at least `1` where the indicator is `1`.

Let,

  ```
  X       a real random variable
  t > 0   a threshold
  1{A}    the indicator of event A
  ```

First, Markov. Let `X ≥ 0`. Pointwise, `t ⋅ 1{X ≥ t} ≤ X`, since the left
side is `0 ≤ X` off the event and `t ≤ X` on it. Take expectations,

  ```
  t ⋅ P(X ≥ t) = 𝔼[ t ⋅ 1{X ≥ t} ]
               ≤ 𝔼[X]
  ```

so `P(X ≥ t) ≤ 𝔼[X] / t`.

Second, Chebyshev. Let `μ ≝ 𝔼[X]` and `σ² ≝ 𝕍(X) < ∞`. Apply Markov to the
nonnegative `(X − μ)²` at threshold `t²`,

  ```
  P(|X − μ| ≥ t) = P( (X − μ)² ≥ t² )
                 ≤ 𝔼[ (X − μ)² ] / t²
                 = σ² / t²
  ```

where the first step uses that `u ↦ u²` is increasing on `u ≥ 0`.

Third, Chernoff. For any `λ > 0`, `x ↦ e^{λx}` is increasing and positive.
Apply Markov to `e^{λX}` at threshold `e^{λt}`,

  ```
  P(X ≥ t) = P( e^{λX} ≥ e^{λt} )
           ≤ 𝔼[ e^{λX} ] / e^{λt}
           = e^{−λt} ⋅ 𝔼[ e^{λX} ]
  ```

The bound holds for every `λ > 0`, and trivially (as `1`) at `λ = 0`, so it
holds at the infimum. Here `t` may be any real number.

Fourth, the Gaussian example. Let `X ~ N(μ, σ²)`. Completing the square
in the exponent gives the moment generating function,

  ```
  𝔼[ e^{λX} ] = ∫ (2πσ²)^{−1/2} ⋅ exp( λx − (x − μ)² / (2σ²) ) dx
              = exp( λμ + λ²σ²/2 ) ⋅ ∫ (2πσ²)^{−1/2} ⋅ exp( −(x − μ − λσ²)² / (2σ²) ) dx
              = exp( λμ + λ²σ²/2 )
  ```

where the second step uses
`λx − (x − μ)²/(2σ²) = λμ + λ²σ²/2 − (x − μ − λσ²)²/(2σ²)`, and the last
step integrates a `N(μ + λσ², σ²)` density. For `t > μ`, minimize the
exponent of the Chernoff bound over `λ`,

  ```
  −λt + λμ + λ²σ²/2   minimized at   λ* = (t − μ) / σ² > 0
  −λ*t + λ*μ + λ*²σ²/2 = −(t − μ)²/σ² + (t − μ)²/(2σ²)
                       = −(t − μ)² / (2σ²)
  ```

Hence,

  ```
  P(X ≥ t)       ≤ 𝔼[X] / t                      (Markov, X ≥ 0)
  P(|X − μ| ≥ t) ≤ σ² / t²                       (Chebyshev)
  P(X ≥ t)       ≤ inf_{λ ≥ 0} e^{−λt} 𝔼[e^{λX}]   (Chernoff)
  P(X ≥ t)       ≤ exp( −(t − μ)² / (2σ²) )      (Gaussian, t > μ)
  ```

Markov is tight iff `X ∈ {0, t}` a.s., where `t ⋅ 1{X ≥ t} = X`. Chebyshev
is tight iff `|X − μ| ∈ {0, t}` a.s. Chernoff is not tight for a single
variable, but for sums of `n` iid copies its exponent is exact as `n → ∞`
(Cramér's theorem, cited, not proved).

Regularity: Markov needs `X ≥ 0` and `𝔼[X] < ∞`; Chebyshev needs
`𝕍(X) < ∞`; Chernoff needs `𝔼[e^{λX}] < ∞` for the `λ` used (otherwise the
bound is vacuous). The Gaussian step needs only the Gaussian density.
