# Fubini–Tonelli

Swapping the order of integration follows from two facts:

1. for nonnegative functions, both iterated integrals equal the integral
   against the product measure, because they agree on rectangles and both
   are limits of simple functions (monotone convergence), and,
2. a signed function splits as `f = f⁺ − f⁻`, so the swap holds whenever
   `∫∫|f| < ∞` makes both parts finite.

Let `(X, 𝒜, μ)` and `(Y, ℬ, ν)` be σ-finite measure spaces, `μ ⊗ ν` the
product measure on `𝒜 ⊗ ℬ`, and `f : X × Y → [−∞, ∞]` measurable.

First, Tonelli, for `f ≥ 0`. On an indicator of a rectangle,
`f = 𝟙_{A×B}`,

  ```
  ∫_X ∫_Y 𝟙_{A×B} dν dμ = ∫_X 𝟙_A(x) ν(B) dμ
                        = μ(A) ν(B)
                        = ∫_Y ∫_X 𝟙_{A×B} dμ dν
  ```

and `μ(A) ν(B)` is `(μ ⊗ ν)(A × B)` by definition of the product measure. The
measurable sets on which both iterated integrals agree with `μ ⊗ ν` form a
monotone class containing the rectangles (σ-finiteness keeps the measures
finite on an exhausting sequence), so they include all of `𝒜 ⊗ ℬ`. Linearity
extends this to nonnegative simple functions, and monotone convergence to
every measurable `f ≥ 0`,

  ```
  ∫_X ∫_Y f dν dμ = ∫_{X×Y} f d(μ ⊗ ν) = ∫_Y ∫_X f dμ dν    in [0, ∞]
  ```

Second, Fubini, for integrable `f`. If `∫|f| d(μ ⊗ ν) < ∞`, apply Tonelli to
`f⁺` and `f⁻`, both nonnegative with finite integral. Subtracting,

  ```
  ∫_X ∫_Y f dν dμ = ∫ f⁺ − ∫ f⁻
                  = ∫_Y ∫_X f dμ dν
  ```

where finiteness rules out `∞ − ∞`. Tonelli applied to `|f|` checks the
hypothesis in either order.

Hence,

  ```
  f ≥ 0:                      ∫∫ f dν dμ = ∫∫ f dμ dν          (Tonelli)
  ∫∫ |f| < ∞:                ∫∫ f dν dμ = ∫∫ f dμ dν          (Fubini)
  ```

Examples.

- Tower property. `𝔼[𝔼[X|Y]] = Σ_y Σₓ x p(x, y) = Σₓ x Σ_y p(x, y) = 𝔼[X]`
  swaps two sums; `𝔼|X| < ∞` licenses it (law-of-total-expectation.md).
- Bayes risk. `r(δ) = ∫ π(θ) ∫ p(y|θ) L dy dθ = ∫ p(y) ∫ p(θ|y) L dθ dy`
  swaps parameter and data; `L ≥ 0` makes it Tonelli, no integrability
  needed (bayes-decision.md).
- The layer-cake formula. For `X ≥ 0`,
  `𝔼[X] = 𝔼 ∫₀^∞ 𝟙[t < X] dt = ∫₀^∞ P(X > t) dt`, Tonelli on
  `𝟙[t < X(ω)] ≥ 0`.

The hypotheses are needed. On `ℕ × ℕ` with counting measure, `f(m, n) = 1`
if `m = n`, `−1` if `m = n + 1`, else `0`: summing each row first gives `0`,
each column first gives `1`. Here `Σ|f| = ∞`.

Regularity: σ-finiteness (Lebesgue measure on `ℝ` with counting measure on
`[0, 1]` breaks Tonelli); measurability of `f` with respect to the product
σ-algebra; for Fubini, `∫|f| < ∞` on the product. The inner integrals are
then measurable functions of the outer variable and finite almost
everywhere.
