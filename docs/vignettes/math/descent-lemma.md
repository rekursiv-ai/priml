# Descent Lemma

The descent lemma follows from the fundamental theorem of calculus along a
segment, plus Cauchy–Schwarz on the gradient change.

Let,

  ```
  f : ℝⁿ → ℝ differentiable
  ‖∇f(x) − ∇f(y)‖ ≤ L ⋅ ‖x − y‖       for all x, y      (L-smooth)
  z(t)            ≝ x + t ⋅ (y − x)               for t ∈ [0, 1]
  ```

First, write the change in `f` as an integral along the segment and split
off the gradient at `x`,

  ```
  f(y) − f(x) = ∫₀¹ ⟨∇f(z(t)), y − x⟩ dt
              = ⟨∇f(x), y − x⟩ + ∫₀¹ ⟨∇f(z(t)) − ∇f(x), y − x⟩ dt
  ```

where the first step is the fundamental theorem of calculus applied to
`t ↦ f(z(t))`, whose derivative is `⟨∇f(z(t)), y − x⟩`.

Second, bound the remainder by Cauchy–Schwarz, then L-smoothness,

  ```
  ∫₀¹ ⟨∇f(z(t)) − ∇f(x), y − x⟩ dt ≤ ∫₀¹ ‖∇f(z(t)) − ∇f(x)‖ ⋅ ‖y − x‖ dt
                                   ≤ ∫₀¹ L ⋅ ‖z(t) − x‖ ⋅ ‖y − x‖ dt
                                   = ∫₀¹ L ⋅ t ⋅ ‖y − x‖² dt
                                   = (L/2) ⋅ ‖y − x‖²
  ```

where the third step uses `z(t) − x = t ⋅ (y − x)`.

Hence,

  ```
  f(y) ≤ f(x) + ⟨∇f(x), y − x⟩ + (L/2) ⋅ ‖y − x‖²
  ```

Bounding the remainder from below the same way gives the two-sided version,
`|f(y) − f(x) − ⟨∇f(x), y − x⟩| ≤ (L/2) ⋅ ‖y − x‖²`.

Third, the right side is a quadratic in `y` minimized at
`y = x − ∇f(x)/L`. Substituting that `y`,

  ```
  f(x − ∇f(x)/L) ≤ f(x) + ⟨∇f(x), −∇f(x)/L⟩ + (L/2) ⋅ ‖∇f(x)/L‖²
                 = f(x) − ‖∇f(x)‖²/L + ‖∇f(x)‖²/(2L)
                 = f(x) − ‖∇f(x)‖²/(2L)
  ```

Hence,

  ```
  f(x − ∇f(x)/L) ≤ f(x) − ‖∇f(x)‖²/(2L)
  ```

If `f* = inf f > −∞`, then `f* ≤ f(x − ∇f(x)/L)`, which rearranges to
`‖∇f(x)‖² ≤ 2L ⋅ (f(x) − f*)`.

Equality holds for `f(x) = (L/2) ⋅ ‖x‖² + ⟨b, x⟩ + c`: the remainder is
exactly `(L/2) ⋅ ‖y − x‖²`, and Cauchy–Schwarz is tight because
`∇f(z(t)) − ∇f(x)` is parallel to `y − x`.

Regularity: the proof needs `f` differentiable with `∇f` L-Lipschitz on the
segment `[x, y]` (so `t ↦ f(z(t))` is C¹ and the integral exists), and the
Euclidean norm, for which Cauchy–Schwarz holds with `⟨·,·⟩`. No convexity is
used.
