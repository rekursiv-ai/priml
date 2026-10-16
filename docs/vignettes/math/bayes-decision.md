# Bayes Decision Rules

Bayes optimality follows from one fact: Bayes risk is the average over `Y` of
posterior expected loss, so minimizing it pointwise in `y` minimizes the
average.

Let,

  ```
  θ ~ π(θ),   Y | θ ~ p(Y|θ)
  δ(Y)              a decision rule, taking values in actions a
  L(θ, a) ≥ 0       a loss
  R(θ, δ) ≝ 𝔼_{p(Y|θ)}[ L(θ, δ(Y)) ]
  r(δ)    ≝ 𝔼_π[ R(θ, δ) ]
  ρ(a|y)  ≝ 𝔼_{p(θ|y)}[ L(θ, a) ]
  δ_B(y) ∈ argmin_a ρ(a|y)
  ```

First, Bayes risk averages posterior loss. Swap the order of integration over
the joint `p(θ, y) = π(θ) p(y|θ) = p(y) p(θ|y)`,

  ```
  r(δ) = ∫ π(θ) ∫ p(y|θ) L(θ, δ(y)) dy dθ
       = ∫ p(y) ∫ p(θ|y) L(θ, δ(y)) dθ dy
       = ∫ p(y) ⋅ ρ(δ(y)|y) dy
  ```

where the second step is Tonelli, valid since `L ≥ 0`.

Second, `δ_B` wins pointwise, so it wins on average,

  ```
  r(δ_B) = ∫ p(y) ⋅ ρ(δ_B(y)|y) dy
         ≤ ∫ p(y) ⋅ ρ(δ(y)|y) dy
         = r(δ)
  ```

where the middle step uses `ρ(δ_B(y)|y) ≤ ρ(δ(y)|y)` for every `y`.

Hence,

  ```
  r(δ_B) ≤ r(δ)   for every rule δ
  ```

Third, squared loss `L(θ, a) = ‖θ − a‖²` is minimized by the posterior mean
`m(y) ≝ 𝔼[θ|y]`,

  ```
  ρ(a|y) = 𝔼[ ‖θ − m(y) + m(y) − a‖² | y ]
         = 𝔼[ ‖θ − m(y)‖² | y ] + 2 (m(y) − a)ᵀ 𝔼[ θ − m(y) | y ] + ‖m(y) − a‖²
         = 𝔼[ ‖θ − m(y)‖² | y ] + ‖m(y) − a‖²
  ```

since `𝔼[θ − m(y) | y] = 0`. The minimizer is `a = m(y)`, uniquely.

Fourth, absolute loss `L(θ, a) = |θ − a|` (scalar `θ`) is minimized by any
posterior median `m`, i.e., `P(θ ≤ m | y) ≥ 1/2` and `P(θ ≥ m | y) ≥ 1/2`.
Take `a > m`. If `θ ≤ m`, then `|θ − a| − |θ − m| = (a − θ) − (m − θ) = a − m`.
If `θ > m`, the triangle inequality gives `|θ − a| − |θ − m| ≥ −(a − m)`. So,

  ```
  ρ(a|y) − ρ(m|y) = 𝔼[ |θ − a| − |θ − m| | y ]
                  ≥ (a − m) ⋅ P(θ ≤ m | y) − (a − m) ⋅ P(θ > m | y)
                  = (a − m) ⋅ (2 P(θ ≤ m | y) − 1)
                  ≥ 0
  ```

The case `a < m` is the mirror image, using `P(θ ≥ m | y) ≥ 1/2`.

Fifth, 0-1 loss `L(θ, a) = 1{θ ≠ a}` (discrete `θ`) is minimized by the
posterior mode (MAP),

  ```
  ρ(a|y) = Σ_θ p(θ|y) ⋅ 1{θ ≠ a}
         = 1 − p(θ = a | y)
  ```

which is smallest at `a = argmax_θ p(θ|y)`.

Equality `r(δ) = r(δ_B)` holds iff `δ(y)` also minimizes `ρ(⋅|y)` for
`p(y)`-almost every `y`. Under squared loss the Bayes rule is unique; under
absolute and 0-1 loss, ties among medians or modes give multiple Bayes rules.

Regularity: the proof needs `L ≥ 0` (for Tonelli), a proper prior so `p(y)`
and `p(θ|y)` exist, `r(δ_B) < ∞` for the comparison to be informative, a
measurable choice of `δ_B`, `𝔼[‖θ‖² | y] < ∞` for squared loss,
`𝔼[|θ| | y] < ∞` for absolute loss, and discrete `θ` for 0-1 loss.
