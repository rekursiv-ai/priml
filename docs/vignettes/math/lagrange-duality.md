# Lagrange Duality

Weak duality is one inequality chain. Strong duality follows from separating
the set of achievable (constraint, objective) values from the points that
would beat the optimum; Slater rules out a vertical separator.

Let,

  ```
  f : ℝⁿ → ℝ,   g ≝ (g₁, …, gₘ) : ℝⁿ → ℝᵐ
  p*     ≝ inf { f(x) : g(x) ≤ 0 }
  L(x,λ) ≝ f(x) + λᵀ g(x)
  d(λ)   ≝ inf_x L(x,λ)
  d*     ≝ sup_{λ ≥ 0} d(λ)
  ```

First, weak duality. For any feasible `x` and any `λ ≥ 0`,

  ```
  d(λ) ≤ L(x,λ)
       = f(x) + λᵀ g(x)
       ≤ f(x)
  ```

since `λ ≥ 0` and `g(x) ≤ 0`. Taking `sup` over `λ` and `inf` over `x`
gives `d* ≤ p*`.

For strong duality, assume `f` and each `gᵢ` are convex, and Slater's
condition: some `x̄` has `g(x̄) < 0` componentwise. Then `p* < ∞`. If
`p* = −∞`, weak duality forces `d* = −∞ = p*`. So take `p*` finite.

The separation theorem used: if `A, B ⊂ ℝᵏ` are nonempty, convex, and
disjoint, there is `a ≠ 0` and `α` with `⟨a,z⟩ ≥ α` on `A` and `⟨a,z⟩ ≤ α`
on `B`.

Second, build the two sets in `ℝᵐ × ℝ`,

  ```
  A ≝ { (g(x) + u, f(x) + t) : x ∈ ℝⁿ, u ≥ 0, t ≥ 0 }
  B ≝ { (0, s) : s < p* }
  ```

`A` is convex: averaging two points of `A` and using convexity of `f`, `gᵢ`
lands above `(g(x), f(x))` at the averaged `x`. `B` is convex. They are
disjoint: `(0,s) ∈ A` would give an `x` with `g(x) ≤ 0` and
`f(x) ≤ s < p*`, which contradicts the definition of `p*`.

Third, separate. There is `(λ̃, μ) ≠ 0` and `α` with

  ```
  λ̃ᵀ(g(x) + u) + μ (f(x) + t) ≥ α     for all x, u ≥ 0, t ≥ 0
  μ s ≤ α                             for all s < p*
  ```

Letting a single `uᵢ → ∞` or `t → ∞` forces `λ̃ ≥ 0` and `μ ≥ 0`. Letting
`s ↑ p*` gives `μ p* ≤ α`. Putting `u = 0`, `t = 0`,

  ```
  μ p* ≤ α
       ≤ λ̃ᵀ g(x) + μ f(x)             for all x
  ```

Fourth, Slater gives `μ > 0`. If `μ = 0`, then `λ̃ ≥ 0`, `λ̃ ≠ 0`, and the
chain at `x̄` reads

  ```
  0 ≤ λ̃ᵀ g(x̄)
    < 0
  ```

since every `gᵢ(x̄) < 0` and some `λ̃ᵢ > 0`. Contradiction.

Fifth, divide by `μ`. Put `λ* ≝ λ̃ / μ ≥ 0`,

  ```
  p* ≤ f(x) + λ*ᵀ g(x)                for all x
  p* ≤ inf_x L(x,λ*)
     = d(λ*)
     ≤ d*
  ```

Hence, with weak duality,

  ```
  d* = d(λ*) = p*
  ```

The dual optimum is attained at `λ*`, even when the primal optimum is not.
Without Slater the gap can be positive or the dual unattained; e.g.,
minimize `x` subject to `x² ≤ 0` has `p* = 0` but `d(λ) = −1/(4λ)`, so
`d* = 0` is not attained.

Regularity: weak duality needs nothing beyond `λ ≥ 0`. Strong duality needs
`f`, `gᵢ` convex and finite on `ℝⁿ` (so `A` is convex), Slater's strict
feasibility (so `μ > 0`), and the finite-dimensional separation theorem.
Affine equality constraints can be added, but need a relative-interior
version of Slater; they are omitted here.
