# Subgradient Method

The projected subgradient method converges because of two facts:

1. projection onto `C` never increases the distance to `x* ∈ C`, and,
2. a subgradient at `x_k` points away from `x*` by at least `f(x_k) − f*`.

Let,

  ```
  C ⊆ ℝⁿ closed and convex,   f : ℝⁿ → ℝ convex
  x* ∈ argmin_C f,   f* ≝ f(x*),   R ≥ ‖x₀ − x*‖
  g_k ∈ ∂f(x_k),   ‖g_k‖ ≤ G
  x_{k+1} ≝ Π_C( x_k − ηₖ g_k ),   ηₖ > 0
  ```

First, one step shrinks the squared distance up to a step-size penalty,

  ```
  ‖x_{k+1} − x*‖² = ‖Π_C(x_k − ηₖ g_k) − Π_C(x*)‖²
                  ≤ ‖x_k − ηₖ g_k − x*‖²
                  = ‖x_k − x*‖² − 2ηₖ ⋅ ⟨g_k, x_k − x*⟩ + ηₖ² ⋅ ‖g_k‖²
                  ≤ ‖x_k − x*‖² − 2ηₖ ⋅ (f(x_k) − f*) + ηₖ² ⋅ G²
  ```

where the first step uses `x* ∈ C`, the second uses nonexpansiveness of
`Π_C`, and the last uses the subgradient inequality
`f(x*) ≥ f(x_k) + ⟨g_k, x* − x_k⟩` and `‖g_k‖ ≤ G`.

Second, sum over `k = 0, …, K−1`. The distances telescope,

  ```
  2 Σₖ ηₖ ⋅ (f(x_k) − f*) ≤ ‖x₀ − x*‖² − ‖x_K − x*‖² + G² Σₖ ηₖ²
                          ≤ R² + G² Σₖ ηₖ²
  ```

Third, the best iterate is no worse than the `ηₖ`-weighted average,

  ```
  min_{k<K} f(x_k) − f* ≤ Σₖ ηₖ ⋅ (f(x_k) − f*) / Σₖ ηₖ
                        ≤ ( R² + G² Σₖ ηₖ² ) / (2 Σₖ ηₖ)
  ```

Fourth, take a constant step `ηₖ = R / (G√K)`. Then `Σₖ ηₖ = R√K / G` and
`Σₖ ηₖ² = R² / G²`,

  ```
  ( R² + G² Σₖ ηₖ² ) / (2 Σₖ ηₖ) = ( R² + R² ) / (2R√K / G)
                                 = R ⋅ G / √K
  ```

Hence,

  ```
  min_{k<K} f(x_k) − f* ≤ ( R² + G² Σₖ ηₖ² ) / (2 Σₖ ηₖ)
                        ≤ R ⋅ G / √K           for ηₖ = R / (G√K)
  ```

The rate is optimal up to a constant. For `K ≤ n − 1` there is a convex
G-Lipschitz `f` on which every method whose iterates lie in `x₀` plus the
span of past subgradients has `f(x_K) − f* = Ω(RG / √K)`
(Nesterov, Introductory Lectures on Convex Optimization, 2004, Thm 3.2.1;
Nemirovski and Yudin, 1983).

Fine print: `f(x_k)` need not decrease; a subgradient step can increase
`f`, which is why the bound is on the best iterate. Nonexpansiveness of
`Π_C` follows from the obtuse-angle condition
`⟨a − Π_C(a), y − Π_C(a)⟩ ≤ 0` for all `y ∈ C`, applied at `a` and `b` and
added, then Cauchy–Schwarz. With `C = ℝⁿ` the projection is the identity.
Diminishing steps with `Σηₖ = ∞` and `Σηₖ² < ∞` drive the bound to 0 without
knowing `K`. The same argument with stochastic unbiased subgradients gives
sgd-convergence.md.

Regularity: the proof needs `C` closed, convex, and nonempty (so `Π_C` is
defined and nonexpansive), `f` convex with a minimizer `x*` over `C`,
subgradients bounded by `G` at the iterates, and `R ≥ ‖x₀ − x*‖`.
