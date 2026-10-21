# Adam

Adam's behavior follows from two facts:

1. its step is a proximal gradient step in a diagonal metric set by the
   running root-mean-square of the gradient, and,
2. Cauchy–Schwarz on the two exponential averages bounds every coordinate's
   step, however large the gradient.

The bounded step makes Adam scale-free per coordinate; it does not make it
robust to outlying samples.

Let `g_t` be a stochastic gradient, `η > 0`, `0 ≤ β₁ < √β₂ < 1`, `ε ≥ 0`,
and, coordinatewise,

  ```
  m_t ≝ β₁ m_{t−1} + (1 − β₁) g_t,         m₀ = 0
  v_t ≝ β₂ v_{t−1} + (1 − β₂) g_t²,        v₀ = 0
  D_t ≝ diag( √v_t + ε )
  x_{t+1} ≝ x_t − η D_t⁻¹ m_t
  ```

Bias correction divides `m_t` by `1 − β₁ᵗ` and `v_t` by `1 − β₂ᵗ`; it
changes the early scale, not the argument below.

First, Adam is a proximal step. Completing the square as in
gd-as-proximal.md, with `‖d‖²_D ≝ dᵀ D d`,

  ```
  argmin_y { ⟨m_t, y − x_t⟩ + (1/2η) ‖y − x_t‖²_{D_t} } = x_t − η D_t⁻¹ m_t
  ```

so Adam is gradient descent on a linear model of `f` built from the momentum
`m_t`, penalized in a metric that stretches coordinates whose gradients are
large. Newton's step (newton-as-proximal.md) is the same form with `D` the
Hessian; Adam's `D` is the square root of a running second moment of the
gradient, which equals curvature only when the gradient noise is Fisher noise
and the empirical Fisher is the Hessian.

Second, the step is bounded per coordinate. Unroll both averages (`ε = 0`),

  ```
  m_t = (1 − β₁) Σ_{k=0}^{t−1} β₁ᵏ g_{t−k}
  v_t = (1 − β₂) Σ_{k=0}^{t−1} β₂ᵏ g_{t−k}²
  ```

and write `β₁ᵏ = (β₁/√β₂)ᵏ ⋅ β₂^{k/2}`. Cauchy–Schwarz gives

  ```
  |m_t| ≤ (1 − β₁) Σ_k (β₁/√β₂)ᵏ ⋅ β₂^{k/2} |g_{t−k}|
        ≤ (1 − β₁) ⋅ √( Σ_k (β₁²/β₂)ᵏ ) ⋅ √( Σ_k β₂ᵏ g_{t−k}² )
        ≤ (1 − β₁) ⋅ (1 − β₁²/β₂)^{−½} ⋅ √( v_t / (1 − β₂) )
  ```

Hence,

  ```
  |x_{t+1} − x_t| = η |m_t| / √v_t ≤ η (1 − β₁) / √( (1 − β₂)(1 − β₁²/β₂) )
  ```

per coordinate, independent of the gradient's scale. Multiplying a
coordinate's gradients by any constant leaves its steps unchanged.

Third, the limits. With `β₁ = β₂ = 0`, the step is
`η g_t/|g_t| = η sign(g_t)`, steepest descent in `‖·‖∞`:

  ```
  argmin_{‖d‖∞ ≤ 1} ⟨g, d⟩ = −sign(g)
  ```

since each coordinate of `d` independently takes the value opposite `gᵢ`.
So Adam interpolates between sign descent (short memory) and a
per-coordinate normalized SGD (long memory).

Examples.

- A coordinate with gradients `c ⋅ g` for any `c ≠ 0` takes the same steps as
  one with gradients `g`: Adam needs no per-layer learning rates.
- One huge gradient `G` at step `t` gives a step at most the bound above,
  about `η √(1/(1 − β₂))` for `β₁ = 0`; it cannot take an unbounded step,
  but `√v` stays inflated for about `1/(1 − β₂)` steps, shrinking every later
  step on that coordinate.
- One outlying sample in a batch shifts the averaged `g_t`, and Adam rescales
  that shifted direction. Robustness to a sample needs the sample's influence
  bounded before averaging (polyak-tsypkin.md), which Adam does not do.

Regularity: the step bound needs `β₁² < β₂`. Convergence is not guaranteed:
there are convex problems on which Adam with fixed `β₁, β₂` does not converge
(Reddi, Kale & Kumar 2018), fixed by keeping `v_t` nondecreasing (AMSGrad),
cited, not proved here. Decoupled weight decay (AdamW) adds `−ηλ x_t` outside
the preconditioner.
