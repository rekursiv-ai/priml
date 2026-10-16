# Glivenko–Cantelli

Glivenko–Cantelli follows from two facts:

1. the SLLN makes `Fₙ` converge at any finitely many points, and,
2. monotonicity of `Fₙ` and `F` brackets every other `x` between grid points.

Let,

  ```
  X₁, X₂, … iid ~ F,   F(x) ≝ P(X ≤ x)
  Fₙ(x)  ≝ (1/n) Σᵢ 1{Xᵢ ≤ x}
  Fₙ(x−) ≝ (1/n) Σᵢ 1{Xᵢ < x},   F(x−) ≝ P(X < x)
  ```

Fix an integer `k ≥ 2`.

First, build a grid where `F` rises by at most `1/k` between neighbors. Set
`x₀ ≝ −∞`, `x_k ≝ +∞`, and

  ```
  xⱼ ≝ inf{ x : F(x) ≥ j/k },   j = 1, …, k−1
  ```

Each `xⱼ` is finite since `j/k ∈ (0, 1)`, and `x₀ ≤ x₁ ≤ … ≤ x_k`. By
right-continuity `F(xⱼ) ≥ j/k`, and by the infimum `F(xⱼ−) ≤ j/k`. With the
conventions `F(−∞) = 0` and `F(+∞−) = 1`, for every `j = 1, …, k`,

  ```
  F(xⱼ−) − F(xⱼ₋₁) ≤ j/k − (j−1)/k
                   = 1/k
  ```

Second, the SLLN controls the grid. Apply it to the iid indicators
`1{Xᵢ ≤ xⱼ}` (mean `F(xⱼ)`) and `1{Xᵢ < xⱼ}` (mean `F(xⱼ−)`),

  ```
  Δₙ ≝ maxⱼ max( |Fₙ(xⱼ) − F(xⱼ)|, |Fₙ(xⱼ−) − F(xⱼ−)| )
     → 0   almost surely
  ```

since a maximum of finitely many terms that each tend to `0` tends to `0`.
At `x₀` and `x_k` the terms are `0` by convention.

Third, bracket an arbitrary `x`. Pick `j` with `xⱼ₋₁ ≤ x < xⱼ`. Both `Fₙ`
and `F` are nondecreasing, so

  ```
  Fₙ(x) ≤ Fₙ(xⱼ−)
        ≤ F(xⱼ−) + Δₙ
        ≤ F(xⱼ₋₁) + 1/k + Δₙ
        ≤ F(x) + 1/k + Δₙ
  ```

and

  ```
  Fₙ(x) ≥ Fₙ(xⱼ₋₁)
        ≥ F(xⱼ₋₁) − Δₙ
        ≥ F(xⱼ−) − 1/k − Δₙ
        ≥ F(x) − 1/k − Δₙ
  ```

where the third lines use the first step, and the fourth lines use
monotonicity of `F` with `xⱼ₋₁ ≤ x < xⱼ`.

Fourth, take the sup and the limit,

  ```
  limsupₙ sup_x |Fₙ(x) − F(x)| ≤ limsupₙ (1/k + Δₙ)
                               = 1/k   almost surely
  ```

Hence, intersecting the probability-one events over `k = 2, 3, …`,

  ```
  sup_x |Fₙ(x) − F(x)| → 0   almost surely
  ```

The bracket is tight to within `1/k`. No rate follows, since the SLLN gives
none; dkwm.md supplies an exponential one.

Fine print: `F` need not be continuous. Atoms are why the grid tracks both
`Fₙ(xⱼ)` and `Fₙ(xⱼ−)`. Repeated grid points (`xⱼ₋₁ = xⱼ`) give empty
intervals and are harmless. The sup is over uncountably many `x`. Since
`Fₙ − F` is right-continuous, it equals a sup over rational `x`, so it is
measurable. The argument is the template for uniform laws over function
classes with finite bracketing numbers.

Regularity: the proof needs `X₁, X₂, …` iid real-valued, and the SLLN for
bounded iid indicators (cited). No moment, density, or continuity condition
on `F` is used.
