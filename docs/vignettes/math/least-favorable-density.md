# Least Favorable Distributions

A robust loss is the likelihood of the hardest distribution near the model.
Two facts make this precise:

1. the hardest member of a convex neighborhood maximizes the criterion's
   difficulty (entropy for log loss, inverse Fisher information for
   location), and its own likelihood is the best response to it, so the pair
   is a saddle point, and,
2. the hardest member keeps the model where the neighborhood binds and is as
   flat as the criterion allows where it does not.

Two criteria give two instances: log loss on a finite outcome gives a
probability floor; estimation variance for a location gives Huber's clipped
score.

Throughout, a neighborhood of a model `p` is the ε-contamination set

  ```
  𝒬_ε(p) ≝ { (1 − ε) p + ε h : h any distribution },    ε ∈ (0, 1)
  ```

## Log loss on a finite outcome

Let `Y` take values in `{1, …, K}`, `H(m) ≝ −Σ_k m_k log m_k`, and let the
statistician predict `q` and pay `−log q_Y`. The game is

  ```
  sup_{Q∈𝒬_ε(p)} inf_q 𝔼_Q[−log q_Y]
  ```

First, the inner problem is entropy. Log loss is proper (gibbs-inequality.md),
so `inf_q 𝔼_Q[−log q_Y] = H(Q)` at `q = Q`, and nature's problem is
`max_{Q∈𝒬_ε(p)} H(Q)`. Let `m` be its maximizer.

Second, `m` is a saddle point. For any `Q ∈ 𝒬_ε(p)`, the segment
`m_t ≝ (1 − t) m + t Q` stays in the convex set, and `H(m_t) ≤ H(m)`, so the
derivative at `t = 0` is nonpositive,

  ```
  0 ≥ d/dt H(m_t) |_{t=0}
    = −Σ_k (Q_k − m_k) ⋅ (log m_k + 1)
    = 𝔼_Q[−log m_Y] − H(m)
  ```

So `𝔼_Q[−log m_Y] ≤ H(m)` for every `Q`: predicting `m` never costs more than
`H(m)`, and `H(m)` is the most nature can force. Hence

  ```
  sup_Q 𝔼_Q[−log m_Y] = H(m) = sup_Q inf_q 𝔼_Q[−log q_Y]
  ```

Third, the shape. Maximize `H(m)` subject to `m_k ≥ (1 − ε) p_k` and
`Σ_k m_k = 1`. Stationarity of the Lagrangian with multipliers `ν` (sum) and
`μ_k ≥ 0` (floors) gives

  ```
  0   = −log m_k − 1 − ν + μ_k
  m_k = e^{−1−ν} ≝ λ                  where the floor is slack (μ_k = 0)
  m_k = (1 − ε) p_k                   where it binds, with (1 − ε) p_k ≥ λ
  ```

so

  ```
  m_k = max( (1 − ε) p_k, λ ),    λ set by Σ_k m_k = 1
  ```

The least favorable predictive keeps the model's large probabilities and
water-fills the small ones up to a common level `λ`. The robust loss is
`−log m_Y`, bounded by `−log λ`: a confidently wrong label costs at most
`−log λ`, and its gradient through `p` vanishes once its class is floored.

Examples.

- Binary, `p = (σ, 1 − σ)` with `σ ≥ ½`. If `(1 − ε)σ ≤ ½` both floors are
  slack and `m = (½, ½)`: nature makes the label a coin flip. Otherwise all
  the contamination goes to the minority class,
  `m = ((1 − ε)σ, 1 − (1 − ε)σ)`, and the minority's loss is at most
  `−log ε`.
- Uniform label noise, `p̃ = (1 − ε) p + ε/K`. This is the exact likelihood
  when a fraction `ε` of labels are redrawn uniformly; its MLE is consistent
  under that model. It is not the minimax predictive above, which floors at
  `λ` instead of adding `ε/K` to every class.
- Softmax with no contamination (`ε → 0`). `λ → 0` and `m → p`: the log loss
  itself, whose logit gradient `p − e_y` is already bounded but never
  vanishes on a confident error.

## Variance of a location estimate

For a continuous location the log-loss game has no saddle: differential
entropy over `𝒬_ε` is unbounded (spread `h` to infinity). The criterion that
works is asymptotic variance. Let `f` be a density on `ℝ`, symmetric about
`0` and log-concave, with score `ψ_f ≝ −(log f)′` (odd, nondecreasing), and,

  ```
  I(p)   ≝ ∫ p′(z)² / p(z) dz
  V(ψ, p) ≝ 𝔼_p[ψ²] / (𝔼_p[ψ′])²
  k      ≝ ψ_f(c)
  ψ_c(z) ≝ clip(ψ_f(z), −k, k)
  ```

with `c > 0` the root of `∫_{−c}^{c} f + 2f(c)/k = 1/(1 − ε)`. An
M-estimator solving `Σᵢ ψ(yᵢ − θ̂) = 0` has `√n (θ̂ − θ) → N(0, V(ψ, p))`
under `p`, and `V(ψ, p) ≥ I(p)⁻¹` (cramer-rao.md), with equality at the MLE
score. The difficulty is `I(p)⁻¹`.

First, the shape. Define

  ```
  p_c(z) ≝ (1 − ε) f(z)                          |z| ≤ c
           (1 − ε) f(c) ⋅ exp(−k(|z| − c))       |z| > c
  ```

It is a density by the choice of `c`, and lies in `𝒬_ε(f)` because
log-concavity puts `log f` below its tangent at `c`,
`log f(z) ≤ log f(c) − k(|z| − c)`. Its score is `ψ_c`, so
`I(p_c) = 𝔼_{p_c}[ψ_c²]`.

Second, `p_c` is the hardest. Substitute `u ≝ √p`,

  ```
  I(p) = ∫ (p′)²/p
       = ∫ (2u u′)²/u²
       = 4 ∫ (u′)²
  ```

a convex functional, minimized subject to `u² ≥ (1 − ε) f` and `∫ u² = 1`.
Where the constraint is slack, Euler–Lagrange gives `u″ = (λ/4) u`, so
`p ∝ exp(−√λ |z|)`; where it binds, `p = (1 − ε) f`. The minimizer is `C¹`,
so `(log p)′` matches at `|z| = c`, where `f`'s slope is `−k`. That is `p_c`,
and convexity makes it the minimum: `I(p) ≥ I(p_c)` on `𝒬_ε(f)`.
Log-concavity makes the binding set the interval `[−c, c]`.

Third, `p_c` is the worst case for `ψ_c`. For `p = (1 − ε) f + ε h`,

  ```
  𝔼_p[ψ_c²] ≤ (1 − ε) 𝔼_f[ψ_c²] + ε k²
  𝔼_p[ψ_c′] ≥ (1 − ε) 𝔼_f[ψ_c′]
  ```

since `ψ_c² ≤ k²` and `ψ_c′ ≥ 0`; both bounds are attained by `h` on
`|z| > c`, where `ψ_c′ = 0`, as in `p_c`. So `V(ψ_c, p) ≤ V(ψ_c, p_c)`.

Hence the saddle,

  ```
  sup_{p∈𝒬_ε} V(ψ_c, p) = V(ψ_c, p_c) = I(p_c)⁻¹ = inf_ψ sup_{p∈𝒬_ε} V(ψ, p)
  ```

since any `ψ` has `sup_p V(ψ, p) ≥ V(ψ, p_c) ≥ I(p_c)⁻¹`. As a loss with scale
`s` and `z = (y − m)/s`, `−log p_c` is the base NLL inside `c`, extended
linearly with slope `k` outside; its pull on the location is at most `k/s`.

Examples.

- Gaussian, `f = φ`, `ψ_f(z) = z`, `k = c`. The Huber loss,

    ```
    −log p_c(y | m, s) = −log 𝒩(y | m, s) − 𝟙[|z| > c] ⋅ ½ (|z| − c)² + const
    ```

  with `2φ(c)/c − 2Φ(−c) = ε / (1 − ε)`.
- Logistic, `ψ_f(z) = tanh(z/2)`. The score `tanh(z/2)` clipped at
  `tanh(c/2)`.
- Laplace, `ψ_f = sign z`, already bounded: any `k ≥ 1` leaves it unchanged,
  which is why the median needs no robustifying.

## The parallel

  ```
                   difficulty   binds              flattest admissible
  log loss, finite  H(m)         m = (1 − ε) p     constant level λ
  location variance I(p)⁻¹       p = (1 − ε) f     exponential, slope k
  ```

In both, nature spends its `ε` where the model is thin, flattening it as far
as the criterion allows, and the statistician's best response is that
flattened model's own likelihood. Contaminating the data instead (adding noise
to `y`, smoothing labels) changes what is fit, not how hard a bad point pulls,
and is not robust.

Regularity: finite `Y` for the log-loss game (countable `Y` with bounded
entropy works the same way). For the location game: `f` symmetric,
log-concave, absolutely continuous with `I(f) < ∞`; `√p ∈ H¹`; symmetric `p`
so `𝔼_p[ψ_c] = 0` at the truth; `𝔼_p[ψ′] > 0`. Without log-concavity the
binding set need not be an interval. As `ε → 0`, `λ → 0` and `c → ∞`, and
both least favorable members return to the model; for the Gaussian at
`c = 10`, `ε ≈ 10⁻²⁵`.
