# Gradient Descent Convergence

Gradient descent with step `1/L` converges because of two facts:

1. each step decreases `f` by `‖∇f‖²/(2L)` (descent-lemma.md), and,
2. a lower bound (convexity, strong convexity, or PL) ties `‖∇f‖` to the
   suboptimality.

Let,

  ```
  f : ℝⁿ → ℝ L-smooth,   f* ≝ inf f,   x* a minimizer (when used)
  x_{k+1} ≝ x_k − ∇f(x_k)/L
  g_k     ≝ ∇f(x_k)
  ```

First, the descent lemma at `x = x_k` gives monotone decrease,

  ```
  f(x_{k+1}) ≤ f(x_k) − ‖g_k‖²/(2L)
  ```

(a) Convex `f`. Second, combine the decrease with the convexity inequality
`f(x_k) ≤ f* + ⟨g_k, x_k − x*⟩`, then complete the square,

  ```
  f(x_{k+1}) − f* ≤ ⟨g_k, x_k − x*⟩ − ‖g_k‖²/(2L)
                  = (L/2) ⋅ ( ‖x_k − x*‖² − ‖x_k − x* − g_k/L‖² )
                  = (L/2) ⋅ ( ‖x_k − x*‖² − ‖x_{k+1} − x*‖² )
  ```

where the second step expands `‖a − g/L‖² = ‖a‖² − 2⟨a, g⟩/L + ‖g‖²/L²`.

Third, sum over `k = 0, …, K−1`. The right side telescopes, and `f(x_k)` is
nonincreasing by the first step,

  ```
  K ⋅ (f(x_K) − f*) ≤ Σ_{k=0}^{K−1} (f(x_{k+1}) − f*)
                    ≤ (L/2) ⋅ ( ‖x₀ − x*‖² − ‖x_K − x*‖² )
                    ≤ (L/2) ⋅ ‖x₀ − x*‖²
  ```

Hence,

  ```
  f(x_K) − f* ≤ L ⋅ ‖x₀ − x*‖² / (2K)
  ```

The order `1/K` is tight for this method: Drori and Teboulle (2014) show the
exact worst case is `L ⋅ ‖x₀ − x*‖² / (4K + 2)`. Acceleration improves it
to `1/K²` (nesterov.md).

(c) PL inequality. Assume, with no convexity, that for some `μ > 0`,

  ```
  ‖∇f(x)‖² ≥ 2μ ⋅ (f(x) − f*)      for all x       (PL)
  ```

Fourth, substitute PL into the first step,

  ```
  f(x_{k+1}) − f* ≤ f(x_k) − f* − ‖g_k‖²/(2L)
                  ≤ f(x_k) − f* − (μ/L) ⋅ (f(x_k) − f*)
                  = (1 − μ/L) ⋅ (f(x_k) − f*)
  ```

Combining PL with `‖∇f(x)‖² ≤ 2L ⋅ (f(x) − f*)` (descent-lemma.md) gives
`μ ≤ L`, so the factor lies in `[0, 1)`. Unrolling,

  ```
  f(x_K) − f* ≤ (1 − μ/L)^K ⋅ (f(x₀) − f*)
  ```

(b) μ-strongly convex `f`, i.e.,
`f(y) ≥ f(x) + ⟨∇f(x), y − x⟩ + (μ/2) ⋅ ‖y − x‖²`. Fifth, strong convexity
implies PL. Minimize both sides over `y`; the right side is minimized at
`y = x − ∇f(x)/μ`,

  ```
  f* = inf_y f(y) ≥ inf_y [ f(x) + ⟨∇f(x), y − x⟩ + (μ/2) ⋅ ‖y − x‖² ]
                  = f(x) − ‖∇f(x)‖²/μ + ‖∇f(x)‖²/(2μ)
                  = f(x) − ‖∇f(x)‖²/(2μ)
  ```

which rearranges to PL with the same `μ`. So (c) applies.

Sixth, strong convexity at `x*`, where `∇f(x*) = 0`, converts the function
gap into a distance,

  ```
  (μ/2) ⋅ ‖x_K − x*‖² ≤ f(x_K) − f* − ⟨∇f(x*), x_K − x*⟩
                      = f(x_K) − f*
                      ≤ (1 − μ/L)^K ⋅ (f(x₀) − f*)
  ```

Hence,

  ```
  f(x_K) − f* ≤ (1 − μ/L)^K ⋅ (f(x₀) − f*)
  ‖x_K − x*‖² ≤ (2/μ) ⋅ (1 − μ/L)^K ⋅ (f(x₀) − f*)
  ```

The linear rate is tight up to a factor of 2 in the exponent: on
`f(x) = ½ xᵀAx` with eigenvalues `μ ≤ L`, the component along the
`μ`-eigenvector contracts in `f` by exactly `(1 − μ/L)²` per step.

Regularity: the proof needs `f` L-smooth (first step), `f* > −∞`, and for (a)
convexity and an attained minimizer `x*`; for (b) μ-strong convexity, which
gives a unique `x*`; for (c) only PL, with `f*` the infimum. The step size is
exactly `1/L`.
