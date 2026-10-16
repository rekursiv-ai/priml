# Ledoux–Talagrand Contraction

The contraction lemma follows from two facts:

1. averaging over one sign turns a sup into a sup over pairs, and,
2. a sup over pairs is symmetric under swapping the pair, so `|t₁ − s₁|` can
   be replaced by `t₁ − s₁`.

Let,

  ```
  T ⊂ ℝⁿ                     bounded set
  φᵢ: ℝ → ℝ                  |φᵢ(a) − φᵢ(b)| ≤ L ⋅ |a − b|
  σ ≝ (σ₁..σₙ)               iid uniform on {−1, +1}
  ```

The claim is `𝔼 sup_t Σᵢ σᵢ φᵢ(tᵢ) ≤ L ⋅ 𝔼 sup_t Σᵢ σᵢ tᵢ`. Replacing `φᵢ` by
`φᵢ / L` scales the left side by `1/L`, so take `L = 1`.

First, contract one coordinate. Fix any `g: T → ℝ` with `sup g < ∞` and a
1-Lipschitz `φ`,

  ```
  𝔼_{σ₁} sup_t [g(t) + σ₁ φ(t₁)] = ½ sup_t [g(t) + φ(t₁)] + ½ sup_s [g(s) − φ(s₁)]
                                 = ½ sup_{t,s} [g(t) + g(s) + φ(t₁) − φ(s₁)]
                                 ≤ ½ sup_{t,s} [g(t) + g(s) + |t₁ − s₁|]
                                 = ½ sup_{t,s} [g(t) + g(s) + t₁ − s₁]
                                 = ½ sup_t [g(t) + t₁] + ½ sup_s [g(s) − s₁]
                                 = 𝔼_{σ₁} sup_t [g(t) + σ₁ t₁]
  ```

where the second and fifth steps use `sup_t a(t) + sup_s b(s) =
sup_{t,s} (a(t) + b(s))`, and the third is the Lipschitz bound. The fourth
step is the crux. `g(t) + g(s)` is symmetric in `(t, s)`, so the pair `(t, s)`
and the swapped pair `(s, t)` have the same `|t₁ − s₁|`. One of the two has
first coordinate at least the second, where `|t₁ − s₁| = t₁ − s₁`. So the
sup with `|·|` is attained by terms of the sup without it, and
`t₁ − s₁ ≤ |t₁ − s₁|` gives the reverse.

Second, peel one coordinate at a time. Define the hybrids,

  ```
  Hₖ ≝ 𝔼_σ sup_t [ Σ_{i ≤ k} σᵢ tᵢ + Σ_{i > k} σᵢ φᵢ(tᵢ) ],     k = 0..n
  ```

so `H₀` is the left side and `Hₙ` the right side. Fix all signs except `σₖ`.
Then `g(t) ≝ Σ_{i < k} σᵢ tᵢ + Σ_{i > k} σᵢ φᵢ(tᵢ)` is a fixed function, and
the first step applies to coordinate `k`,

  ```
  H_{k−1} = 𝔼_{σ_{−k}} 𝔼_{σₖ} sup_t [g(t) + σₖ φₖ(tₖ)]
          ≤ 𝔼_{σ_{−k}} 𝔼_{σₖ} sup_t [g(t) + σₖ tₖ]
          = Hₖ
  ```

where `σ_{−k}` denotes the other signs, independent of `σₖ`.

Hence, chaining `H₀ ≤ H₁ ≤ … ≤ Hₙ` and restoring `L`,

  ```
  𝔼_σ sup_{t ∈ T} Σᵢ σᵢ φᵢ(tᵢ) ≤ L ⋅ 𝔼_σ sup_{t ∈ T} Σᵢ σᵢ tᵢ
  ```

With `T = {(f(z₁)..f(zₙ)) : f ∈ F}` and dividing by `n`, this is
`Rad̂_S(φ ∘ F) ≤ L ⋅ Rad̂_S(F)`. For a loss `ℓ(f(x), y)` that is `L`-Lipschitz
in its first argument, take `φᵢ = ℓ(·, yᵢ)`.

Equality holds for `φᵢ(a) = L a + cᵢ`: the constants contribute `Σᵢ σᵢ cᵢ`,
which leaves the sup and has mean zero.

Fine print: the `φᵢ` may differ across `i`, which is what lets them carry
the labels `yᵢ`. No `φᵢ(0) = 0` or `0 ∈ T` is needed for this one-sided form.
The absolute-value form `𝔼 sup |Σ σᵢ φᵢ(tᵢ)| ≤ 2L ⋅ 𝔼 sup |Σ σᵢ tᵢ|` needs
`φᵢ(0) = 0` and is cited (Ledoux and Talagrand, 1991), not proved. If `T` is
unbounded the right side may be `∞`, and the bound is trivially true.

Regularity: the proof needs `σᵢ` independent and uniform on `{−1, +1}` (the
first step writes `𝔼_{σ₁}` as an equal-weight average), each `φᵢ`
`L`-Lipschitz, and finite sups, e.g., `T` bounded. `σ` takes finitely many
values, so no measurability condition is needed.
