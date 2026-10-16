# Bias–Variance Decomposition

The decomposition is two applications of one fact: a cross term between a
zero-mean quantity and an independent one has zero expectation.

Let,

  ```
  y    = f(x) + ε,   𝔼[ε|x] = 0,      𝕍(ε|x) = σ²(x)
  D                  training set, independent of the test pair (x, ε)
  f̂    ≝ f̂_D         predictor fit on D
  f̄(x) ≝ 𝔼_D[ f̂_D(x) ]
  ```

Fix the test input `x`. All expectations below are conditional on `x`.

First, split off the noise. Write `y − f̂(x) = ε + (f(x) − f̂(x))`,

  ```
  𝔼[(y − f̂(x))²] = 𝔼[ε²] + 2 ⋅ 𝔼[ε ⋅ (f(x) − f̂(x))] + 𝔼_D[(f(x) − f̂(x))²]
                 = σ²(x) + 2 ⋅ 𝔼[ε] ⋅ 𝔼_D[f(x) − f̂(x)] + 𝔼_D[(f(x) − f̂(x))²]
                 = σ²(x) + 0 + 𝔼_D[(f(x) − f̂(x))²]
  ```

where the second step uses that `ε` is independent of `D` given `x`, and the
third uses `𝔼[ε|x] = 0`.

Second, split the remaining term around the mean prediction `f̄(x)`,

  ```
  𝔼_D[(f(x) − f̂(x))²] = 𝔼_D[((f(x) − f̄(x)) + (f̄(x) − f̂(x)))²]
                      = (f(x) − f̄(x))² + 2 ⋅ (f(x) − f̄(x)) ⋅ 𝔼_D[f̄(x) − f̂(x)]
                                       + 𝔼_D[(f̄(x) − f̂(x))²]
                      = (f(x) − f̄(x))² + 2 ⋅ (f(x) − f̄(x)) ⋅ 0
                                       + 𝕍_D(f̂(x))
                      = (f(x) − f̄(x))² + 𝕍_D(f̂(x))
  ```

where the second step pulls the constant `f(x) − f̄(x)` out of `𝔼_D`, and the
third uses `𝔼_D[f̂(x)] = f̄(x)` by definition.

Hence,

  ```
  𝔼[(y − f̂(x))²] = σ²(x) + (f(x) − f̄(x))² + 𝕍_D(f̂(x))
                 = noise  + bias²           + variance
  ```

Taking `𝔼_x` of both sides gives the same decomposition for the risk, with
each term averaged over `x`.

This is an identity, so it always holds with equality. The noise term is the
Bayes risk and no predictor can remove it.

Regularity: the proof needs `𝔼[ε|x] = 0`, `ε` independent of `D` given `x`,
and finite second moments `𝔼[ε²] < ∞` and `𝔼_D[f̂(x)²] < ∞`. It uses squared
loss; other losses do not split additively.
