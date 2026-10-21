# Continuous-Time Optimization

The discrete methods of optimizer-taxonomy.md are time samples of flows. That
follows from two facts:

1. a step `x_{k+1} = x_k − η v(x_k)` is the explicit Euler discretization of
   `ẋ = −v(x)` with time step `η`, and,
2. along a gradient flow `f` decreases at rate `‖∇f‖²`, so the flow's
   Lyapunov function carries over to the steps when `η` is small enough for
   the discretization error to be dominated.

Each taxonomy axis then has a continuous meaning: geometry is a metric,
momentum is inertia, memory is a bank of integrators, influence is
saturation.

Let `f : ℝⁿ → ℝ` be `C¹`, bounded below, and,

  ```
  ẋ(t) = −∇f(x(t))                          gradient flow
  ```

First, the flow descends. By the chain rule,

  ```
  d/dt f(x(t)) = ⟨∇f(x), ẋ⟩
               = −‖∇f(x)‖²
               ≤ 0
  ```

and integrating from `0` to `T` (fundamental-theorem-calculus.md),

  ```
  ∫₀^T ‖∇f(x(t))‖² dt = f(x(0)) − f(x(T)) ≤ f(x(0)) − inf f
  ```

so `min_{t≤T} ‖∇f‖² ≤ (f(x₀) − inf f)/T`. Gradient descent with step `1/L`
satisfies the same bound with `T` replaced by `k/(2L)` (descent-lemma.md):
the step is a unit of time.

Second, geometry is a metric on the flow. With a positive-definite metric
`G(x)`,

  ```
  ẋ = −G(x)⁻¹ ∇f(x)           d/dt f = −∇fᵀ G⁻¹ ∇f ≤ 0
  ```

`G = I` is GD; `G = ∇²f` Newton's flow; `G` the Fisher metric the natural
gradient flow (natural-gradient.md); `G = ∇²φ` the mirror flow
`d/dt ∇φ(x) = −∇f(x)` (mirror-descent.md). Steepest descent in a norm is the
flow whose velocity is that norm's linear minimization oracle.

Third, momentum is inertia. Nesterov's method is a discretization of

  ```
  ẍ + (3/t) ẋ + ∇f(x) = 0
  ```

(Su, Boyd & Candès 2014). For convex `f` with minimizer `x*`, the energy
`ℰ(t) ≝ t² (f(x) − f*) + 2 ‖x + (t/2) ẋ − x*‖²` has `ℰ̇ ≤ 0`, giving
`f(x(t)) − f* ≤ 2 ‖x₀ − x*‖² / t²`: the `O(1/k²)` rate of nesterov.md, with
`t ≈ k √η`. Heavy ball is `ẍ + a ẋ + ∇f = 0` with constant friction `a`.

Fourth, memory is a bank of integrators. An exponential average with
discount `β` per step of length `Δt` is the leaky integrator

  ```
  τ v̇ = −v + g²,    β = exp(−Δt/τ)
  ```

so Adam's `v_t` is a low-pass filter with time constant `τ = Δt/(1 − β₂)`
to first order, and the full sum of AdaGrad is the limit `τ → ∞`. Several
`β` are several time constants.

Fifth, noise and the ODE method. Robbins–Monro
`x_{n+1} = x_n − α_n (h(x_n) + ξ_{n+1})` with `Σ α_n = ∞`, `Σ α_n² < ∞`
tracks the mean flow `ẋ = −h(x)` in the time `t_n = Σ_{k<n} α_k`: the
interpolated iterates are asymptotically a solution of the ODE, so stable
equilibria of the flow are the limits (Ljung 1977; Kushner & Yin;
Borkar 2008, cited). Two step-size schedules on two sets of variables give
two timescales: the fast variables equilibrate while the slow ones see them
as their equilibrium.

Sixth, influence is saturation. A nonlinearity `ψ` on each input to the
flow, `ẋ = −𝔼[ψ(residual)]`, bounds each input's rate of change by
`sup |ψ|`. A saturating unit does this with no explicit clip: its output is
bounded whatever its input.

Hence the correspondence,

  ```
  discrete axis   continuous object               example
  step size η     time step                       GD ↔ gradient flow
  geometry        metric G(x) on the flow         Newton, natural, mirror flows
  momentum        inertia, friction               ẍ + (3/t)ẋ + ∇f = 0
  memory          integrators, time constants τ   τ v̇ = −v + g²
  influence       saturating ψ on each input      bounded firing rates
  noise           mean ODE + timescales           Robbins–Monro ↔ ẋ = −h(x)
  ```

What the continuous view has and the discrete one lacks is locality.
Every method above needs a global gradient, computed for all parameters at
once and applied synchronously. A physical learner relaxes local variables
on many timescales with no global clock; equilibrium propagation and
predictive coding show that such relaxation can compute gradients at a fixed
point, and how far local, asynchronous rules can go is open.

Examples.

- Gradient descent on `f(x) = (μ/2) x²` is `x_{k+1} = (1 − ημ) x_k`, the
  Euler step of `ẋ = −μx`, whose solution `e^{−μt}`; it tracks the flow iff
  `ημ < 1` and oscillates for `1 < ημ < 2`, diverging past `2`.
- Adam with `β₂ = 0.999` integrates `g²` with a time constant of `1,000`
  steps; a gradient spike inflates `v` for about that long.
- Polyak averaging, `x̄_n = (1/n) Σ x_k`, is integrating the trajectory
  and dividing by time: a second, slower integrator on top of the
  iterate.

Regularity: the descent bound needs `f ∈ C¹` bounded below and a flow that
exists for all `t` (true if `∇f` is locally Lipschitz and the sublevel sets
are bounded). The Nesterov ODE bound needs `f` convex and `C¹`; the
correspondence `t ≈ k √η` holds in the small-step limit. The ODE method
needs the Robbins–Monro conditions, bounded iterates, and a globally
asymptotically stable equilibrium of the mean flow.
