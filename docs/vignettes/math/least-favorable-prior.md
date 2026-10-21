# Least Favorable Prior

The minimax rule follows from one fact: the risk of any rule averaged under a
prior is at least that prior's Bayes risk, and no more than its worst case.
So if a Bayes rule's risk never exceeds its own Bayes risk, it is minimax and
its prior is the hardest one.

Let,

  ```
  θ ∈ Θ,   Y | θ ~ p(Y|θ)
  δ(Y)                 a decision rule
  L(θ, a) ≥ 0          a loss
  R(θ, δ) ≝ 𝔼_{p(Y|θ)}[ L(θ, δ(Y)) ]
  r(π, δ) ≝ 𝔼_π[ R(θ, δ) ]
  r(π)    ≝ inf_δ r(π, δ)          the Bayes risk of π
  ```

A prior `π⋆` is least favorable if `r(π⋆) ≥ r(π)` for every prior `π`: the
prior under which even the best rule does worst. A rule `δ⋆` is minimax if
`sup_θ R(θ, δ⋆) = inf_δ sup_θ R(θ, δ)`.

First, every prior's Bayes risk bounds the minimax risk from below. For any
`π` and any `δ`,

  ```
  r(π) ≤ r(π, δ)
       = 𝔼_π[ R(θ, δ) ]
       ≤ sup_θ R(θ, δ)
  ```

so taking `inf_δ` on the right and `sup_π` on the left,

  ```
  sup_π r(π) ≤ inf_δ sup_θ R(θ, δ)
  ```

Second, suppose `δ⋆` is Bayes for `π⋆` (bayes-decision.md) and its risk
never exceeds its Bayes risk,

  ```
  R(θ, δ⋆) ≤ r(π⋆)    for all θ
  ```

Then

  ```
  inf_δ sup_θ R(θ, δ) ≤ sup_θ R(θ, δ⋆)
                      ≤ r(π⋆)
                      ≤ sup_π r(π)
  ```

Hence, combined with the first step,

  ```
  sup_θ R(θ, δ⋆) = r(π⋆) = sup_π r(π) = inf_δ sup_θ R(θ, δ)
  ```

`δ⋆` is minimax, `π⋆` is least favorable, and `(π⋆, δ⋆)` is a saddle point:
nature's best response to `δ⋆` is `π⋆`, and the statistician's best response
to `π⋆` is `δ⋆`. The condition holds in particular for an equalizer: a Bayes
rule with constant risk. Then `π⋆` concentrates its mass where the risk
peaks, which for an equalizer is everywhere.

Examples.

- Bernoulli, squared loss. `Y ~ Binomial(n, q)`, `δ(Y) = (Y + a)/(n + 2a)`,
  the Bayes rule for `Beta(a, a)`. Its risk is

    ```
    R(q, δ) = ( n q(1 − q) + a²(1 − 2q)² ) / (n + 2a)²
    ```

  constant in `q` iff `a = √n / 2`. So `Beta(√n/2, √n/2)` is least
  favorable, `δ⋆ = (Y + √n/2)/(n + √n)` is minimax, and the minimax risk is
  `1/(4(1 + √n)²)`. The MLE `Y/n` has risk `q(1 − q)/n`, worse near `q = ½`.
- Normal mean, squared loss. `Y ~ 𝒩(θ, 1)`. No proper prior is least
  favorable, but the sequence `𝒩(0, k)` has Bayes risk `k/(k + 1) → 1`, the
  constant risk of `δ(Y) = Y`. So `δ(Y) = Y` is minimax as a limit of Bayes
  rules: the least favorable prior is flat, approached but not attained.
- Mutual information. Take `δ` to be a predictive density `q(y)` and
  `L(θ, q) = 𝕂(p(⋅|θ) ‖ q)`. The Bayes rule for `π` is the mixture
  `q_π = ∫ π p(⋅|θ)`, with Bayes risk `I(θ; Y)` under `π`. The least
  favorable prior maximizes `I(θ; Y)`: the capacity-achieving prior, which
  for `n` iid draws tends to Jeffreys (reference-priors.md). The reference
  prior is the least favorable prior for prediction under log loss.

Compare least-favorable-density.md, the same saddle with nature choosing
the noise `h` instead of the parameter `θ`.

Regularity: `L ≥ 0` and measurable; Bayes rules exist for the priors used;
for the limit form, a sequence `πₖ` with `r(πₖ) → sup_θ R(θ, δ⋆)`. A least
favorable prior need not exist (normal mean on `ℝ`), and when it does, on a
compact `Θ` it is often discrete, with finitely many atoms at the points of
maximal risk.
