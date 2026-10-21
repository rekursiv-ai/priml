# Majorization–Minimization

Majorization–minimization descends from one fact: a surrogate that lies above
`f` and touches it at the current point can only be lowered by moving, and
`f` must fall at least as much.

Let `f : X → ℝ`, and for each `x ∈ X` let `g(· | x)` be a majorizer,

  ```
  g(y | x) ≥ f(y)    for all y        (majorizes)
  g(x | x) = f(x)                     (touches at x)
  x_{k+1} ∈ argmin_y g(y | x_k)
  ```

First, monotone descent. Chain the two properties around the update,

  ```
  f(x_{k+1}) ≤ g(x_{k+1} | x_k)
             ≤ g(x_k | x_k)
             = f(x_k)
  ```

where the first step is majorization, the second that `x_{k+1}` minimizes
`g(· | x_k)`, and the third the touching condition.

Second, the decrease is at least the surrogate's. Subtracting,

  ```
  f(x_k) − f(x_{k+1}) ≥ g(x_k | x_k) − g(x_{k+1} | x_k) ≥ 0
  ```

so any progress on the surrogate is progress on `f`. The update need not
minimize `g` exactly: any `y` with `g(y | x_k) ≤ g(x_k | x_k)` keeps descent
(generalized MM).

Third, fixed points are stationary. If `g(· | x)` and `f` are differentiable
and `g(· | x) − f` is minimized at `y = x` (it is `≥ 0` and `= 0` there), its
gradient vanishes, `∇_y g(x | x) = ∇f(x)`. So `x_{k+1} = x_k` implies
`∇f(x_k) = ∇_y g(x_k | x_k) = 0`.

Hence,

  ```
  f(x_{k+1}) ≤ f(x_k),    and x = argmin g(· | x) ⇒ ∇f(x) = 0
  ```

Examples. Each picks a majorizer whose minimum is cheap.

- Gradient descent. For `L`-smooth `f`, the descent lemma
  (descent-lemma.md) gives
  `g(y | x) = f(x) + ⟨∇f(x), y − x⟩ + (L/2)‖y − x‖²`, minimized at
  `x − ∇f(x)/L`. Keeping a nonsmooth `h` exact in the surrogate gives the
  proximal gradient step (gd-as-proximal.md).
- EM. For `f(θ) = −log p(y | θ)` with latent `z`, Jensen gives
  `g(θ | θ_k) = −𝔼_{p(z|y,θ_k)}[ log p(y, z | θ) ] + 𝔼[ log p(z|y,θ_k) ]`,
  which majorizes and touches at `θ_k`; minimizing it is the M-step
  (em-monotonicity.md).
- IRLS for a robust loss. For Huber's `ρ_c`, `u ↦ ρ_c(√u)` is concave on
  `u ≥ 0` (its derivative `½ min(1, c/√u)` is nonincreasing), so its tangent
  line majorizes it,

    ```
    ρ_c(z) ≤ ρ_c(z₀) + ½ w₀ (z² − z₀²),    w₀ ≝ ψ_c(z₀)/z₀ = min(1, c/|z₀|)
    ```

  Summing over residuals, each MM step is a weighted least squares with
  weights `min(1, c/|zᵢ|)`: points past the cap are down-weighted, never
  dropped (least-favorable-density.md).

Regularity: descent needs only the two majorizer properties and an update
that does not raise `g`. Convergence of `f(x_k)` follows if `f` is bounded
below; convergence of `x_k` to a stationary point needs more (continuity of
`g` in both arguments and compact level sets), cited, not proved here.
