# Central Limit Theorem

The CLT follows from two facts:

1. the characteristic function of a standardized variable is
   `1 − s²/2 + o(s²)` near zero, and,
2. Lévy's continuity theorem turns pointwise convergence of characteristic
   functions into convergence in distribution (cited, not proved).

Let,

  ```
  X₁, X₂, …   iid, μ ≝ 𝔼[Xᵢ], σ² ≝ 𝕍(Xᵢ) ∈ (0, ∞)
  Yᵢ   ≝ (Xᵢ − μ) / σ,   so 𝔼[Yᵢ] = 0, 𝔼[Yᵢ²] = 1
  Zₙ   ≝ (1/√n) Σᵢ₌₁ⁿ Yᵢ
  φ(s) ≝ 𝔼[ e^{isY₁} ],   φₙ(t) ≝ 𝔼[ e^{itZₙ} ]
  ```

Fix `t ∈ ℝ`. The goal is `φₙ(t) → e^{−t²/2}`, the characteristic function
of `N(0, 1)`.

First, independence factors `φₙ`,

  ```
  φₙ(t) = 𝔼[ Πᵢ e^{i(t/√n)Yᵢ} ]
        = Πᵢ 𝔼[ e^{i(t/√n)Yᵢ} ]
        = φ(t/√n)ⁿ
  ```

Second, bound the Taylor remainder of `e^{ix}` for real `x`. Let
`R₁(x) ≝ e^{ix} − 1 − ix` and `R₂(x) ≝ R₁(x) + x²/2`. Integral forms give,

  ```
  R₁(x)   = −∫₀ˣ (x − u) e^{iu} du,          |R₁(x)| ≤ x²/2
  R₂(x)   = −(i/2) ∫₀ˣ (x − u)² e^{iu} du,   |R₂(x)| ≤ |x|³/6
  |R₂(x)| ≤ |R₁(x)| + x²/2 ≤ x²
  ```

so `|R₂(x)| ≤ min(|x|³/6, x²)`.

Third, expand `φ` near zero. Using `𝔼[Y] = 0` and `𝔼[Y²] = 1`,

  ```
  |φ(s) − (1 − s²/2)| = |𝔼[ e^{isY} − 1 − isY + s²Y²/2 ]|
                      = |𝔼[ R₂(sY) ]|
                      ≤ 𝔼[ min(|s|³|Y|³/6, s²Y²) ]
                      = s² ⋅ 𝔼[ min(|s||Y|³/6, Y²) ]
  ```

As `s → 0`, the integrand `min(|s||Y|³/6, Y²)` tends to `0` pointwise and is
dominated by the integrable `Y²`. By dominated convergence its expectation
tends to `0`. So `φ(s) = 1 − s²/2 + o(s²)`.

Fourth, compare the `n`-th powers. Let `a ≝ φ(t/√n)` and `b ≝ 1 − t²/(2n)`.
For `n ≥ t²/2`, `|b| ≤ 1`, and `|a| ≤ 1` always. Telescoping,

  ```
  |aⁿ − bⁿ| = |Σₖ₌₀ⁿ⁻¹ aᵏ (a − b) bⁿ⁻¹⁻ᵏ|
            ≤ Σₖ₌₀ⁿ⁻¹ |a|ᵏ |a − b| |b|ⁿ⁻¹⁻ᵏ
            ≤ n ⋅ |a − b|
            = n ⋅ o(t²/n)
            → 0
  ```

where the third line uses `|a|, |b| ≤ 1`, and `|a − b| = o(t²/n)` is the
third step at `s = t/√n`.

Fifth, the comparison sequence converges,

  ```
  bⁿ = (1 − t²/(2n))ⁿ → e^{−t²/2}
  ```

by the classical limit `(1 + c/n)ⁿ → eᶜ` for real `c`.

Hence, `φₙ(t) → e^{−t²/2}` for every `t`, and by Lévy's continuity theorem,

  ```
  √n (X̄ₙ − μ) / σ → N(0, 1)   in distribution
  ```

The `o(s²)` in the third step gives no rate. With `𝔼|Y|³ < ∞` the same bound
gives `|φ(s) − (1 − s²/2)| ≤ |s|³ 𝔼|Y|³ / 6`, and Berry–Esseen gives a CDF
error of order `𝔼|Y|³ / √n` (cited, not proved).

Regularity: the proof needs iid `Xᵢ` (to factor `φₙ` into a power), finite
positive variance (to standardize and to dominate the remainder by `Y²`),
and Lévy's continuity theorem, which is cited.
