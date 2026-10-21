# Fundamental Theorem of Calculus

The fundamental theorem follows from two facts:

1. the integral's difference quotient is an average of `f` over a shrinking
   interval, which tends to `f` at a point of continuity, and,
2. two functions with the same derivative on an interval differ by a
   constant (the mean value theorem).

Let `f : [a, b] → ℝ` be continuous, and,

  ```
  F(x) ≝ ∫_a^x f(t) dt
  ```

First, `F′ = f`. For `x ∈ (a, b)` and small `h ≠ 0`,

  ```
  (F(x + h) − F(x)) / h − f(x) = (1/h) ∫_x^{x+h} ( f(t) − f(x) ) dt
  ```

Given `ε > 0`, continuity gives `δ` with `|f(t) − f(x)| < ε` for
`|t − x| < δ`, so for `|h| < δ`,

  ```
  | (F(x + h) − F(x)) / h − f(x) | ≤ (1/|h|) ⋅ |h| ⋅ ε
                                  = ε
  ```

Hence `F′(x) = f(x)`; one-sided at `a` and `b`.

Second, any antiderivative gives the integral. Let `G` be differentiable on
`[a, b]` with `G′ = f`. Then `(G − F)′ = 0`, and the mean value theorem gives,
for any `x`, a `ξ` between `a` and `x` with

  ```
  (G − F)(x) − (G − F)(a) = (G − F)′(ξ) ⋅ (x − a)
                          = 0
  ```

so `G − F` is constant. Since `F(a) = 0`,

  ```
  ∫_a^b f(t) dt = F(b) − F(a)
                = G(b) − G(a)
  ```

Hence,

  ```
  d/dx ∫_a^x f(t) dt = f(x),       ∫_a^b G′(t) dt = G(b) − G(a)
  ```

Examples. Most first-order arguments in this collection apply the second
form along a segment.

- Taylor with integral remainder. For `g ∈ C¹` along `z(t) = x + t(y − x)`,

    ```
    g(y) − g(x) = ∫₀¹ ⟨∇g(z(t)), y − x⟩ dt
    ```

  which with a Lipschitz `∇g` gives the descent lemma's
  `(L/2) ‖y − x‖²` bound (descent-lemma.md).
- Newton's error. Applied to `∇f` between `x*` and `x`,
  `∇f(x) − ∇f(x*) = ∫₀¹ ∇²f(z(t)) (x − x*) dt`, the integral that a
  Lipschitz Hessian bounds by `(M/2) ‖x − x*‖²` (newton-as-proximal.md).
- Integration by parts. The product rule `(uv)′ = u′v + uv′` integrated
  gives `∫ u′v = [uv] − ∫ uv′`, the step behind Stein's lemma
  (stein-lemma.md) and the zero-mean score (cramer-rao.md).

Regularity: the first form needs `f` continuous at `x` (Riemann or
Lebesgue integrable elsewhere); the second needs `G` differentiable with
`G′ = f` Riemann integrable. For Lebesgue integrals the second form holds iff
`G` is absolutely continuous, and the first holds almost everywhere for any
integrable `f` (Lebesgue differentiation).
