# Jensen's Inequality

Jensen follows from one fact: a convex function lies above its supporting
line at the mean.

Let,

  ```
  φ : I → ℝ convex on an interval I
  X ∈ I a.s., with 𝔼|X| < ∞ and 𝔼|φ(X)| < ∞
  μ ≝ 𝔼[X]
  ```

If `μ` is an endpoint of `I`, then `X = μ` a.s. and the claim is trivial.
So assume `μ` is interior to `I`.

First, convexity gives a supporting line at `μ`. Pick `c` between the left
and right derivatives of `φ` at `μ` (both exist and are finite in the
interior), so that for all `x ∈ I`,

  ```
  φ(x) ≥ φ(μ) + c ⋅ (x − μ)
  ```

This holds because the slope `(φ(x) − φ(μ)) / (x − μ)` is nondecreasing in
`x`, so it is `≥ c` for `x > μ` and `≤ c` for `x < μ`.

Second, substitute `X` and take expectations,

  ```
  𝔼[φ(X)] ≥ 𝔼[ φ(μ) + c ⋅ (X − μ) ]
          = φ(μ) + c ⋅ (𝔼[X] − μ)
          = φ(μ) + c ⋅ 0
          = φ(μ)
  ```

where the first step uses monotonicity of `𝔼`, and the second uses linearity.

Hence,

  ```
  φ(𝔼[X]) ≤ 𝔼[φ(X)]
  ```

Equality holds iff the nonnegative gap `φ(X) − φ(μ) − c ⋅ (X − μ)` has zero
mean, i.e., is zero a.s. Then `φ` meets the line `ℓ(x) ≝ φ(μ) + c ⋅ (x − μ)`
at every support point. If it meets `ℓ` at `x₁ < x₂`, convexity puts `φ`
below the chord, which is `ℓ`, on `[x₁, x₂]`; with `φ ≥ ℓ` this forces
`φ = ℓ` there. So equality holds iff `φ` is affine on the convex hull of the
support of `X`. If `φ` is strictly convex, this means `X` is a.s. constant.

The same proof works for convex `φ : ℝᵈ → ℝ`, with a subgradient `c ∈ ∂φ(μ)`
replacing the slope and `cᵀ(x − μ)` replacing `c ⋅ (x − μ)`.

Regularity: the proof needs `φ` convex on an interval containing `X` a.s.,
`𝔼|X| < ∞` so `μ` exists, and `𝔼|φ(X)| < ∞` so the right side is defined
(it may also be `+∞` with the inequality trivial).
