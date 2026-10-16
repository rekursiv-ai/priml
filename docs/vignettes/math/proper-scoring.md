# Proper Scoring Rules

A score built from a convex `G` is proper. This follows from one fact: its
expected score is the tangent of `G` at the forecast, so the shortfall from
honesty is the gap between `G` and its tangent, a Bregman divergence.

Let,

  ```
  y ∈ {1, …, K}   outcomes,   δ_y ∈ ℝᴷ   the one-hot vector of y
  p, q ∈ Δ        forecasts on the probability simplex
  S(p, y)         score of forecast p when y occurs (higher is better)
  S(p, q) ≝ Σ_y q(y) ⋅ S(p, y)   expected score when Y ~ q
  ```

`S` is proper if `S(q, q) ≥ S(p, q)` for all `p, q`. It is strictly proper
if equality forces `p = q`. Let `G : Δ → ℝ` be convex and differentiable,
and set

  ```
  S(p, y)   ≝ G(p) + ⟨∇G(p), δ_y − p⟩
  D_G(q, p) ≝ G(q) − G(p) − ⟨∇G(p), q − p⟩
  ```

First, the expected score is the tangent plane of `G` at `p`, evaluated at
`q`. `S(p, y)` is affine in `δ_y`, and `Σ_y q(y) δ_y = q`,

  ```
  S(p, q) = Σ_y q(y) ⋅ ( G(p) + ⟨∇G(p), δ_y − p⟩ )
          = G(p) + ⟨∇G(p), Σ_y q(y) δ_y − p⟩
          = G(p) + ⟨∇G(p), q − p⟩
  ```

In particular, `S(q, q) = G(q)`.

Second, the gap from honest forecasting is a Bregman divergence,

  ```
  S(q, q) − S(p, q) = G(q) − G(p) − ⟨∇G(p), q − p⟩
                    = D_G(q, p)
                    ≥ 0
  ```

by convexity of `G`, which lies above its tangents.

Hence,

  ```
  S(q, q) − S(p, q) = D_G(q, p) ≥ 0
  ```

So `S` is proper, and strictly proper if `G` is strictly convex, since then
`D_G(q, p) = 0` only at `p = q`.

Third, the log score is strictly proper. Take `G(p) = Σₖ pₖ log pₖ`, so
`∂ₖG(p) = log pₖ + 1`,

  ```
  S(p, y)   = Σₖ pₖ log pₖ + (log p_y + 1) − Σₖ pₖ (log pₖ + 1)
            = log p_y + 1 − Σₖ pₖ
            = log p_y
  D_G(q, p) = Σₖ qₖ log qₖ − Σₖ pₖ log pₖ − Σₖ (log pₖ + 1)(qₖ − pₖ)
            = Σₖ qₖ log qₖ − Σₖ qₖ log pₖ − Σₖ (qₖ − pₖ)
            = 𝕂(q‖p)
  ```

where the last steps use `Σₖ pₖ = Σₖ qₖ = 1`. Negative entropy is strictly
convex, so the gap `𝕂(q‖p)` vanishes only at `p = q`.

Fourth, the Brier score is strictly proper. Take `G(p) = ‖p‖² − 1`, so
`∇G(p) = 2p`,

  ```
  S(p, y)   = ‖p‖² − 1 + 2⟨p, δ_y − p⟩
            = −‖p‖² + 2 p_y − 1
            = −‖p − δ_y‖²
  D_G(q, p) = ‖q‖² − ‖p‖² − 2⟨p, q − p⟩
            = ‖q‖² − 2⟨p, q⟩ + ‖p‖²
            = ‖q − p‖²
  ```

where the third line expands `‖p − δ_y‖² = ‖p‖² − 2p_y + 1`. `‖⋅‖²` is
strictly convex, so the gap `‖q − p‖²` vanishes only at `p = q`.

Fine print: the converse is cited, not proved (Savage 1971; Gneiting and
Raftery 2007, Theorem 1). Every proper `S` has this form, with
`G(q) ≝ S(q, q)` and a subgradient in place of `∇G`. Convexity of that `G` is
immediate: `G(q) = sup_p S(p, q)` is a sup of functions linear in `q`. The
link to bregman-conditional-mean.md is `−S(p, y) = D_G(δ_y, p) − G(δ_y)`.
So maximizing `𝔼[S(p, Y)]` over `p` minimizes `𝔼[D_G(δ_Y, p)]`, and the
minimizer is `𝔼[δ_Y] = q`. `∇G` is defined only up to adding `c ⋅ 𝟏` on
the simplex. This is harmless, since `⟨𝟏, δ_y − p⟩ = 0`. For the log score,
`p` must be interior for `∇G` to exist. At the boundary, `p_y = 0` gives
`S = −∞`, which only strengthens strict propriety.

Regularity: the proof needs finitely many outcomes, `G` convex and
differentiable at the forecast `p` (strictly convex for strict propriety),
and, for the log score, `p` in the interior of the simplex.
