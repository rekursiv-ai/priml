# Banach Fixed-Point

Banach's theorem follows from two facts:

1. a contraction shrinks consecutive steps geometrically, so iterates are
   Cauchy, and,
2. completeness supplies the limit, and continuity of `T` makes it fixed.

Let,

  ```
  (X, d) complete metric space
  T : X → X,   d(Tx, Ty) ≤ γ d(x, y)   for all x, y,   with 0 ≤ γ < 1
  x₀ ∈ X,      xₖ₊₁ ≝ T xₖ
  ```

First, steps shrink geometrically. By induction on `k`,

  ```
  d(xₖ₊₁, xₖ) = d(T xₖ, T xₖ₋₁)
              ≤ γ d(xₖ, xₖ₋₁)
              ≤ γᵏ d(x₁, x₀)
  ```

Second, the iterates are Cauchy. For `m > k`, by the triangle inequality and
the geometric series,

  ```
  d(xₘ, xₖ) ≤ Σ_{j=k}^{m−1} d(xⱼ₊₁, xⱼ)
            ≤ Σ_{j=k}^{m−1} γʲ d(x₁, x₀)
            ≤ Σ_{j=k}^{∞} γʲ d(x₁, x₀)
            = γᵏ / (1 − γ) ⋅ d(x₁, x₀)
  ```

which tends to `0` as `k → ∞`. By completeness `xₖ → x*` for some `x* ∈ X`.

Third, the limit is fixed,

  ```
  d(T x*, x*) ≤ d(T x*, T xₖ) + d(xₖ₊₁, x*)
              ≤ γ d(x*, xₖ) + d(xₖ₊₁, x*)
              → 0
  ```

using `T xₖ = xₖ₊₁`. So `T x* = x*`.

Fourth, the fixed point is unique. If `T y* = y*` too,

  ```
  d(x*, y*) = d(T x*, T y*)
            ≤ γ d(x*, y*)
  ```

so `(1 − γ) d(x*, y*) ≤ 0`, and `γ < 1` gives `x* = y*`.

Fifth, the rates. Let `m → ∞` in the second step; `d(·, xₖ)` is continuous,
so the a-priori bound follows. Separately, contracting toward `x*`,

  ```
  d(xₖ, x*) = d(T xₖ₋₁, T x*)
            ≤ γ d(xₖ₋₁, x*)
            ≤ γᵏ d(x₀, x*)
  ```

Hence,

  ```
  T has a unique fixed point x*, xₖ → x* for every x₀, and
  d(xₖ, x*) ≤ γᵏ d(x₀, x*)
  d(xₖ, x*) ≤ γᵏ / (1 − γ) ⋅ d(x₁, x₀)
  ```

The a-priori bound is attained. On `X = ℝ` take `T x = γ x` and `x₀ = 1`.
Then `xₖ = γᵏ`, `x* = 0`, `d(x₁, x₀) = 1 − γ`, and the bound equals
`γᵏ = d(xₖ, x*)`. Applying it from `xₖ₋₁` instead of `x₀` gives the
a-posteriori bound `d(xₖ, x*) ≤ γ / (1 − γ) ⋅ d(xₖ, xₖ₋₁)`.

Regularity: the proof needs `X` complete (for the limit to exist in `X`),
`T` to map `X` into itself, and a uniform `γ < 1`. The weaker condition
`d(Tx, Ty) < d(x, y)` is not enough: `T x = x + 1/x` on `[1, ∞)` has no
fixed point.
