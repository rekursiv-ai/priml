# Reference Priors

The reference prior follows from two facts:

1. the prior that maximizes the information an experiment carries about the
   parameter is the one whose mass follows the experiment's own resolution,
   and,
2. for `n` iid draws in a regular model the posterior is asymptotically
   `N(θ̂, I(θ)⁻¹/n)`, so that resolution is `√det I(θ)`, Jeffreys' prior.

Jeffreys' prior is then also the unique prior invariant under
reparameterization, and the two derivations agree.

Let `p(y | θ)` be a regular model with `θ ∈ Θ ⊆ ℝᵈ`, `π` a prior, and,

  ```
  I(θ)       ≝ 𝔼_{p(Y|θ)}[ ∇log p(Y|θ) ∇log p(Y|θ)ᵀ ]
  Yⁿ          n iid draws from p(y | θ)
  𝕀_n(π)     ≝ I(θ; Yⁿ) = 𝔼_π 𝕂( p(θ | Yⁿ) ‖ π(θ) )
  π_J(θ)     ≝ √det I(θ) / ∫ √det I
  ```

`𝕀_n(π)` is the mutual information between parameter and data when the
parameter is drawn from `π` (mutual-information.md): the expected KL from
prior to posterior. A reference prior is the limit of
`argmax_π 𝕀_n(π)` as `n → ∞` (Bernardo 1979; Berger–Bernardo 1992).

First, Jeffreys is invariant. Let `φ = g(θ)` be a smooth bijection with
Jacobian `J ≝ ∂θ/∂φ`. The score transforms as
`∇_φ log p = Jᵀ ∇_θ log p`, so

  ```
  I_φ(φ)          = Jᵀ I_θ(θ) J
  √det I_φ(φ)     = √det I_θ(θ) ⋅ |det J|
  ```

which is exactly how a density transforms (change-of-variables.md). So
computing `π_J` in `θ` and transforming to `φ` gives the same prior as
computing `π_J` in `φ` directly. No other prior built from the model alone
has this property: invariance fixes the density as a multiple of the
volume element of the Fisher metric.

Second, the information is asymptotically a KL to Jeffreys. Under
regularity, the posterior is asymptotically `N(θ̂, (n I(θ))⁻¹)` with entropy
`½ log det(2πe (n I(θ))⁻¹)`. Since `𝕀_n = H(θ) − H(θ | Yⁿ)`,

  ```
  𝕀_n(π) = −∫ π log π  −  𝔼_π[ ½ log det(2πe (n I(θ))⁻¹) ] + o(1)
         = (d/2) log( n / (2πe) ) + ∫ π ⋅ log( √det I / π ) + o(1)
         = (d/2) log( n / (2πe) ) + log ∫ √det I − 𝕂( π ‖ π_J ) + o(1)
  ```

where the last step writes `√det I = π_J ⋅ ∫ √det I`.

Third, maximize. Only `−𝕂(π‖π_J)` depends on `π`, and it is maximized, at
`0`, iff `π = π_J` (gibbs-inequality.md).

Hence, in regular one-parameter (or one-block) models,

  ```
  argmax_π 𝕀_n(π) → π_J ∝ √det I(θ)
  𝕀_n(π_J) = (d/2) log( n / (2πe) ) + log ∫ √det I(θ) dθ + o(1)
  ```

The constant `log ∫ √det I` is the log-volume of `Θ` under the Fisher
metric: the number of distinguishable parameter values grows like
`(√n)ᵈ ⋅ ∫ √det I`.

Examples.

- Normal mean, `p(y | μ) = 𝒩(μ, σ²)` with `σ` known. `I(μ) = 1/σ²` is
  constant, so `π_J` is flat on `μ`: every location is equally resolvable.
- Bernoulli, `p(y | q) = q^y (1 − q)^{1−y}`. `I(q) = 1/(q(1 − q))`, so

    ```
    π_J(q) = 1 / (π √(q(1 − q)))  = Beta(½, ½)
    ```

  more mass near `0` and `1`, where a few draws already pin `q` down.
- Normal scale, `𝒩(0, σ²)`. `I(σ) = 2/σ²`, so `π_J ∝ 1/σ`: flat on `log σ`,
  invariant to the choice of units.

Where they differ. With a parameter of interest `ψ` and a nuisance `λ`,
Jeffreys uses the joint `√det I(ψ, λ)`, while the reference prior is built
sequentially: the conditional `π(λ | ψ)` is the reference prior for `λ` with
`ψ` fixed, then `π(ψ)` maximizes the information about `ψ` in the model with
`λ` integrated out. For `𝒩(μ, σ²)` with both unknown, Jeffreys gives
`π ∝ 1/σ²`, the reference prior gives `π ∝ 1/σ`. In the Neyman–Scott problem,
`(Y_{i1}, Y_{i2}) ~ 𝒩(μᵢ, σ²)` for `i = 1, …, n`, the Jeffreys posterior for
`σ²` concentrates on `σ²/2`, an inconsistent answer, while the reference prior
with `σ` of interest gives a consistent posterior.

Regularity: the asymptotic posterior normality needs a regular model
(identifiable, `I(θ)` continuous and positive definite, `log p` smooth in
`θ`, `θ` interior). If `∫ √det I = ∞` (unbounded `Θ`, e.g. normal mean on
`ℝ`), `π_J` is improper; the reference prior is then defined as the limit
over compact subsets `Θ_k ↑ Θ`, and is used only through its posterior, which
must be proper. The equality of the reference prior and Jeffreys holds for a
single parameter or a single block; with ordered groups of parameters the
sequential construction governs.
