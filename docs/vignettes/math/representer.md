# Representer Theorem

The representer theorem follows from two facts:

1. the reproducing property makes `f(xᵢ)` blind to the component of `f`
   orthogonal to the data span, and,
2. Pythagoras makes that component strictly increase `‖f‖_H`.

Let,

  ```
  H                          RKHS on X with kernel k
  ⟨f, k(x, ·)⟩_H = f(x)      reproducing property, all f ∈ H, x ∈ X
  x₁..xₙ ∈ X                 data points
  V              ≝ span{k(x₁, ·)..k(xₙ, ·)}
  L: ℝⁿ → ℝ ∪ {∞}            arbitrary loss on the fitted values
  Ω: [0, ∞) → ℝ              strictly increasing
  J(f)           ≝ L(f(x₁)..f(xₙ)) + Ω(‖f‖_H)
  ```

First, decompose. `V` is finite dimensional, hence closed, so every `f ∈ H`
splits orthogonally,

  ```
  f = f_V + f_⊥,     f_V ∈ V,     ⟨f_⊥, g⟩_H = 0 for all g ∈ V
  ```

Second, the fitted values ignore `f_⊥`. For each `i`, `k(xᵢ, ·) ∈ V`,

  ```
  f(xᵢ) = ⟨f, k(xᵢ, ·)⟩_H
        = ⟨f_V, k(xᵢ, ·)⟩_H + ⟨f_⊥, k(xᵢ, ·)⟩_H
        = ⟨f_V, k(xᵢ, ·)⟩_H + 0
        = f_V(xᵢ)
  ```

where the first and last steps are the reproducing property.

Third, the norm grows with `f_⊥`. Pythagoras gives,

  ```
  ‖f‖_H = √( ‖f_V‖²_H + ‖f_⊥‖²_H )
        ≥ ‖f_V‖_H
  ```

with equality iff `f_⊥ = 0`.

Combining, since the `L` terms agree and `Ω` is strictly increasing,

  ```
  J(f) = L(f_V(x₁)..f_V(xₙ)) + Ω(‖f‖_H)
       ≥ L(f_V(x₁)..f_V(xₙ)) + Ω(‖f_V‖_H)
       = J(f_V)
  ```

with strict inequality whenever `f_⊥ ≠ 0` and `J(f) < ∞`.

Hence, any minimizer `f*` of `J` with `J(f*) < ∞` has `f*_⊥ = 0`, i.e.,

  ```
  f* = Σᵢ αᵢ k(xᵢ, ·)        for some α ∈ ℝⁿ
  ```

Then `f*(xⱼ) = (K α)ⱼ` and `‖f*‖²_H = αᵀ K α` with `Kᵢⱼ = k(xᵢ, xⱼ)`, so the
infinite-dimensional problem reduces to one over `α ∈ ℝⁿ`.

If `Ω` is only nondecreasing, the inequality is not strict: every minimizer's
projection `f*_V` is also a minimizer, so some minimizer lies in `V`, but
others may not. The theorem asserts the form of minimizers, not their
existence; existence needs, e.g., `L` lower semicontinuous and `Ω` coercive.

Regularity: the proof needs `H` a Hilbert space with the reproducing property
(so evaluation is the inner product with `k(xᵢ, ·)`), `Ω` strictly increasing,
and `L` depending on `f` only through `f(x₁)..f(xₙ)`. No convexity of `L` or
`Ω` is used.
