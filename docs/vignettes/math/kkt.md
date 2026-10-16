# Karush–Kuhn–Tucker

KKT follows from squeezing the duality gap to zero: under strong duality
every inequality in the weak-duality chain is tight, and the tight
inequalities are exactly complementary slackness and stationarity.

Let,

  ```
  minimize f(x)  subject to  gᵢ(x) ≤ 0,  i = 1, …, m
  f, gᵢ : ℝⁿ → ℝ  differentiable
  L(x,λ) ≝ f(x) + Σᵢ λᵢ gᵢ(x)
  d(λ)   ≝ inf_x L(x,λ)
  ```

The KKT conditions at `(x*, λ*)` are,

  ```
  gᵢ(x*)                  ≤ 0             primal feasibility
  λᵢ*                     ≥ 0             dual feasibility
  λᵢ* gᵢ(x*)              = 0             complementary slackness
  ∇f(x*) + Σᵢ λᵢ* ∇gᵢ(x*) = 0             stationarity
  ```

First, necessity. Let `x*` be primal optimal and `λ* ≥ 0` dual optimal with
zero gap, `f(x*) = d(λ*)`. For convex `f`, `gᵢ` under Slater such a `λ*`
exists (see Lagrange duality). Then,

  ```
  f(x*) = d(λ*)
        = inf_x L(x,λ*)
        ≤ L(x*,λ*)
        = f(x*) + Σᵢ λᵢ* gᵢ(x*)
        ≤ f(x*)
  ```

where the last step uses `λᵢ* ≥ 0` and `gᵢ(x*) ≤ 0`. The two ends agree, so
both inequalities are equalities.

Second, complementary slackness from the last equality,

  ```
  Σᵢ λᵢ* gᵢ(x*) = 0
  ```

Each term is `≤ 0`, and terms `≤ 0` summing to `0` are each `0`. So
`λᵢ* gᵢ(x*) = 0` for every `i`: an inactive constraint has zero multiplier.

Third, stationarity from the first equality. It says
`L(x*,λ*) = inf_x L(x,λ*)`, so `x*` is an unconstrained minimizer of the
differentiable function `L(·,λ*)` on `ℝⁿ`. Hence its gradient vanishes,

  ```
  0 = ∇ₓ L(x*,λ*)
    = ∇f(x*) + Σᵢ λᵢ* ∇gᵢ(x*)
  ```

Fourth, sufficiency. Let `f`, `gᵢ` be convex and let `(x*,λ*)` satisfy KKT.
Then `L(·,λ*)` is convex, since `λ* ≥ 0`. A convex differentiable function
lies above its tangent plane, `L(x,λ*) ≥ L(x*,λ*) + ∇ₓL(x*,λ*)ᵀ(x − x*)`,
and stationarity zeros the gradient. So for any feasible `x`,

  ```
  f(x) ≥ f(x) + Σᵢ λᵢ* gᵢ(x)
       = L(x,λ*)
       ≥ L(x*,λ*)
       = f(x*) + Σᵢ λᵢ* gᵢ(x*)
       = f(x*)
  ```

where the first step uses `λᵢ* ≥ 0`, `gᵢ(x) ≤ 0`, and the last uses
complementary slackness.

Hence,

  ```
  KKT at (x*,λ*)  ⇒  x* primal optimal,  λ* dual optimal,  f(x*) = d(λ*)
  x* optimal, Slater, convex  ⇒  ∃ λ* with (x*,λ*) satisfying KKT
  ```

The dual claim in the first line holds because the fourth step's chain
shows `d(λ*) = inf_x L(x,λ*) = L(x*,λ*) = f(x*)`, which meets the
weak-duality bound. Without a constraint qualification KKT can fail at an
optimum: minimize `x` subject to `x² ≤ 0` has `x* = 0` and needs
`1 + λ ⋅ 0 = 0`, which has no solution.

Regularity: necessity needs `f`, `gᵢ` differentiable at `x*` and a zero-gap
dual optimum `λ*` (supplied by convexity plus Slater). Sufficiency needs
`f`, `gᵢ` convex and differentiable; it needs no constraint qualification.
