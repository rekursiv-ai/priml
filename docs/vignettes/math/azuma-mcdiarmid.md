# Azuma–Hoeffding and McDiarmid

Both bounds follow from two facts:

1. Hoeffding's lemma holds conditionally, one martingale step at a time, and,
2. any function of independent inputs is the endpoint of its Doob martingale.

Let,

  ```
  F₀ ⊆ F₁ ⊆ … ⊆ Fₙ           a filtration
  M₀, M₁, …, Mₙ               a martingale: 𝔼[Mₖ | Fₖ₋₁] = Mₖ₋₁
  Dₖ      ≝ Mₖ − Mₖ₋₁          the differences, 𝔼[Dₖ | Fₖ₋₁] = 0
  Aₖ      ≤ Dₖ ≤ Bₖ            with Aₖ, Bₖ Fₖ₋₁-measurable
  Bₖ − Aₖ ≤ cₖ                 with cₖ a constant
  ```

First, the conditional Hoeffding lemma. Given `Fₖ₋₁`, `Dₖ` has mean zero
and lies in `[Aₖ, Bₖ]`, an interval of length at most `cₖ`. Hoeffding's
lemma (see `hoeffding.md`), applied under the conditional law, gives,

  ```
  𝔼[ e^{λDₖ} | Fₖ₋₁ ] ≤ e^{λ²(Bₖ − Aₖ)²/8}
                      ≤ e^{λ²cₖ²/8}
  ```

The lemma's proof uses only the endpoints, the zero mean, and convexity, so
it holds with `Fₖ₋₁`-measurable endpoints.

Second, peel off one step at a time with the tower property,

  ```
  𝔼[ e^{λ(Mₙ − M₀)} ] = 𝔼[ e^{λ(Mₙ₋₁ − M₀)} ⋅ e^{λDₙ} ]
                      = 𝔼[ e^{λ(Mₙ₋₁ − M₀)} ⋅ 𝔼[ e^{λDₙ} | Fₙ₋₁ ] ]
                      ≤ e^{λ²cₙ²/8} ⋅ 𝔼[ e^{λ(Mₙ₋₁ − M₀)} ]
                      ≤ exp( λ² Σₖ cₖ² / 8 )
  ```

where the second step pulls out the `Fₙ₋₁`-measurable factor, and the last
step repeats the argument `n − 1` more times.

Third, apply Chernoff with `λ = 4t / Σₖ cₖ²`, exactly as in Hoeffding's
inequality, for `t > 0`,

  ```
  P(Mₙ − M₀ ≥ t) ≤ e^{−λt} ⋅ 𝔼[ e^{λ(Mₙ − M₀)} ]
                 ≤ exp( −λt + λ² Σₖ cₖ² / 8 )
                 = exp( −2t² / Σₖ cₖ² )
  ```

This is Azuma–Hoeffding. If instead `|Dₖ| ≤ cₖ`, the interval
`[−cₖ, cₖ]` has length `2cₖ`, and the bound becomes `exp(−t² / (2 Σₖ cₖ²))`.

Fourth, McDiarmid. Let `X₁, …, Xₙ` be independent and let `f` satisfy the
bounded-difference condition: changing only coordinate `i` changes `f` by at
most `cᵢ`. Let `Fₖ ≝ σ(X₁, …, Xₖ)` and define the Doob martingale and its
partial averages,

  ```
  Mₖ            ≝ 𝔼[ f(X) | Fₖ ],    M₀ = 𝔼[f(X)],    Mₙ = f(X)
  gₖ(x₁, …, xₖ) ≝ 𝔼[ f(x₁, …, xₖ, Xₖ₊₁, …, Xₙ) ]
  ```

By independence, `Mₖ = gₖ(X₁, …, Xₖ)` and, with `X'ₖ` an independent copy
of `Xₖ`, `Mₖ₋₁ = 𝔼_{X'ₖ}[gₖ(X₁, …, Xₖ₋₁, X'ₖ)]`. Set `Aₖ ≝ inf_x gₖ(X₁, …, Xₖ₋₁, x) − Mₖ₋₁` and
`Bₖ ≝ sup_x gₖ(X₁, …, Xₖ₋₁, x) − Mₖ₋₁`. Both are `Fₖ₋₁`-measurable, and
`Aₖ ≤ Dₖ ≤ Bₖ`. The range is bounded,

  ```
  Bₖ − Aₖ = sup_{x, x'} [ gₖ(…, Xₖ₋₁, x) − gₖ(…, Xₖ₋₁, x') ]
          = sup_{x, x'} 𝔼[ f(…, Xₖ₋₁, x, Xₖ₊₁, …) − f(…, Xₖ₋₁, x', Xₖ₊₁, …) ]
          ≤ cₖ
  ```

where the last step applies the bounded-difference condition inside the
expectation. The third step then applies with `Mₙ − M₀ = f(X) − 𝔼[f(X)]`.

Hence,

  ```
  P(Mₙ   − M₀    ≥ t) ≤ exp( −2t² / Σₖ cₖ² )     (Azuma–Hoeffding)
  P(f(X) − 𝔼f(X) ≥ t) ≤ exp( −2t² / Σᵢ cᵢ² )     (McDiarmid)
  ```

Applying each to `−M` or `−f` gives the lower tail, and a factor `2` gives
the two-sided bound. With `f(x) = Σᵢ xᵢ` and `xᵢ ∈ [aᵢ, bᵢ]`, `cᵢ = bᵢ − aᵢ`
and McDiarmid recovers Hoeffding's inequality exactly, so the constant is the
same as Hoeffding's.

Regularity: Azuma needs a martingale with differences in predictable
intervals of length at most constant `cₖ`, and `t > 0`. McDiarmid also needs
`X₁, …, Xₙ` mutually independent (so `gₖ` is the conditional expectation),
`f(X)` integrable, and `sup`/`inf` of `gₖ` measurable (automatic for countable
ranges or continuous `gₖ`).
