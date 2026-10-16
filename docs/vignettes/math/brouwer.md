# Brouwer Fixed-Point

Brouwer follows from one topological fact: a ball cannot be continuously
retracted onto its boundary sphere. In 1-D that fact is the intermediate
value theorem, and the proof is complete. In `n`-D the no-retraction theorem
is cited; the reduction to it is proved.

Let,

  ```
  Bⁿ ≝ { x ∈ ℝⁿ : ‖x‖ ≤ 1 },   Sⁿ⁻¹ ≝ { x ∈ ℝⁿ : ‖x‖ = 1 }
  f : Bⁿ → Bⁿ   continuous
  ```

First, the 1-D case. Let `f : [a,b] → [a,b]` be continuous and
`g(x) ≝ f(x) − x`. Since `f` maps into `[a,b]`,

  ```
  g(a) = f(a) − a ≥ 0
  g(b) = f(b) − b ≤ 0
  ```

`g` is continuous, so by the intermediate value theorem some `c ∈ [a,b]` has
`g(c) = 0`, i.e., `f(c) = c`.

The cited theorem (no-retraction): there is no continuous `r : Bⁿ → Sⁿ⁻¹`
with `r(x) = x` for all `x ∈ Sⁿ⁻¹`. For `n = 1` this is the fact that the
connected `[−1,1]` cannot map continuously onto the disconnected `{−1,1}`.
For `n ≥ 2` it is cited; standard proofs use
`H_{n−1}(Sⁿ⁻¹) = ℤ ≠ 0 = H_{n−1}(Bⁿ)`, or Sperner's lemma, or Stokes'
theorem for a smooth approximation.

Second, a fixed-point-free `f` yields a retraction. Suppose `f(x) ≠ x` for
all `x`, and let `u(x) ≝ x − f(x) ≠ 0`. Send `x` along the ray from `f(x)`
through `x` to the sphere: `r(x) ≝ x + t(x) u(x)` with `t(x) ≥ 0` solving
`‖x + t u‖² = 1`, i.e.,

  ```
  ‖u‖² t² + 2⟨x,u⟩ t + (‖x‖² − 1) = 0
  t(x) = ( −⟨x,u⟩ + √(⟨x,u⟩² + ‖u‖²(1 − ‖x‖²)) ) / ‖u‖²
  ```

The discriminant is `≥ 0` since `‖x‖ ≤ 1`, and `t(x) ≥ 0` since the root
dominates `|⟨x,u⟩|`. `t` is continuous since `‖u‖ > 0` on the compact `Bⁿ`.
So `r` is continuous and `‖r(x)‖ = 1`.

Third, `r` fixes the sphere. For `‖x‖ = 1`, by Cauchy–Schwarz,

  ```
  ⟨x, u⟩ = ‖x‖² − ⟨x, f(x)⟩
         ≥ 1 − ‖x‖ ‖f(x)‖
         ≥ 1 − 1
         = 0
  ```

so `t(x) = ( −⟨x,u⟩ + |⟨x,u⟩| ) / ‖u‖² = 0` and `r(x) = x`. Thus `r` is a
retraction of `Bⁿ` onto `Sⁿ⁻¹`, contradicting no-retraction.

Fourth, extend to any nonempty compact convex `K ⊂ ℝⁿ`. Let `f : K → K` be
continuous, `P_K` the Euclidean projection onto `K` (continuous, indeed
`1`-Lipschitz), and `R` large enough that `K ⊂ R Bⁿ`. Then
`f ∘ P_K : R Bⁿ → K ⊂ R Bⁿ` is continuous and, after scaling by `R`, has a
fixed point `x`,

  ```
  x = f(P_K x) ∈ K   ⇒   P_K x = x   ⇒   f(x) = x
  ```

Hence,

  ```
  every continuous f : K → K, K ⊂ ℝⁿ nonempty compact convex, has a fixed point
  ```

The fixed point need not be unique (`f = id`), and the proof gives no way to
compute it. Each hypothesis is needed: `x ↦ x + 1` on `ℝ` (not compact), a
rotation of the circle `S¹` (not convex), and `f = 1_{x < 0} − 1_{x ≥ 0}`
on `[−1,1]` (not continuous) all have no fixed point.

Regularity: the 1-D proof needs only continuity of `f` and the intermediate
value theorem. The `n`-D proof needs continuity of `f`, `K` nonempty,
compact, and convex (for `P_K`), and the cited no-retraction theorem.
