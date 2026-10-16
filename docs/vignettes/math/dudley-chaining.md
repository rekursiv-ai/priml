# Dudley's Entropy Integral

Dudley's bound follows from three facts:

1. each `f` telescopes through covers at geometrically shrinking scales,
2. each telescoping link ranges over a finite set, so Massart (see massart.md)
   bounds it, and,
3. the sum of links at scales `εⱼ = D 2⁻ʲ` is a Riemann sum for an integral.

Let,

  ```
  S        ≝ (z₁..zₙ)                fixed sample
  ‖f‖ₙ     ≝ √( (1/n) Σᵢ f(zᵢ)² )         L²(Pₙ) norm; identify f with (f(zᵢ))ᵢ ∈ ℝⁿ
  D        ≝ sup_{f ∈ F} ‖f‖ₙ
  N(ε)     ≝ N(ε, F, L²(Pₙ))         fewest centers in ℝⁿ whose ε-balls cover F
  Rad̂_S(F) ≝ 𝔼_σ[ sup_f (1/n) Σᵢ σᵢ f(zᵢ) ]
  εⱼ       ≝ D ⋅ 2⁻ʲ,     j = 0, 1, 2, …
  ```

`N(ε)` is nonincreasing in `ε`, and `N(ε) = 1` for `ε ≥ D`, since the center
`0` covers `F`.

First, telescope. For `j ≥ 1` let `Cⱼ` be an `εⱼ`-cover of size `N(εⱼ)`, and
let `πⱼ f ∈ Cⱼ` with `‖f − πⱼ f‖ₙ ≤ εⱼ`. Set `π₀ f ≝ 0`, which is within `ε₀`.
For any `J ≥ 1`,

  ```
  f = (f − π_J f) + Σ_{j=1}^{J} (πⱼ f − π_{j−1} f) + π₀ f
  ```

so, using `sup (a + b) ≤ sup a + sup b` and `π₀ f = 0`,

  ```
  Rad̂_S(F) ≤ 𝔼 sup_f (1/n) ⟨σ, f − π_J f⟩ + Σ_{j=1}^{J} 𝔼 sup_f (1/n) ⟨σ, πⱼ f − π_{j−1} f⟩
  ```

Second, bound the remainder by Cauchy–Schwarz, using `‖σ‖₂ = √n` and
`‖v‖₂ = √n ‖v‖ₙ`,

  ```
  (1/n) ⟨σ, f − π_J f⟩ ≤ (1/n) ⋅ ‖σ‖₂ ⋅ ‖f − π_J f‖₂
                       = (1/n) ⋅ √n ⋅ √n ‖f − π_J f‖ₙ
                       ≤ ε_J
  ```

Third, bound each link by Massart. The set
`Aⱼ ≝ {πⱼ f − π_{j−1} f : f ∈ F}` has `|Aⱼ| ≤ N(εⱼ) N(ε_{j−1}) ≤ N(εⱼ)²`.
By the triangle inequality each element has
`‖a‖ₙ ≤ εⱼ + ε_{j−1} = 3εⱼ`, so `‖a‖₂ ≤ 3εⱼ √n`. Massart gives,

  ```
  𝔼 max_{a ∈ Aⱼ} (1/n) ⟨σ, a⟩ ≤ 3εⱼ √n ⋅ √(2 log |Aⱼ|) / n
                              ≤ 3εⱼ √n ⋅ √(4 log N(εⱼ)) / n
                              = (6/√n) ⋅ εⱼ √(log N(εⱼ))
  ```

Fourth, convert the sum to an integral. Since `εⱼ = 2(εⱼ − ε_{j+1})` and
`N` is nonincreasing,

  ```
  εⱼ √(log N(εⱼ)) = 2 (εⱼ − ε_{j+1}) ⋅ √(log N(εⱼ))
                  ≤ 2 ∫_{ε_{j+1}}^{εⱼ} √(log N(ε)) dε
  ```

Summing over `j = 1..J` and combining the three bounds,

  ```
  Rad̂_S(F) ≤ ε_J + (6/√n) Σ_{j=1}^{J} εⱼ √(log N(εⱼ))
           ≤ ε_J + (12/√n) ∫_{ε_{J+1}}^{ε₁} √(log N(ε)) dε
           ≤ ε_J + (12/√n) ∫_0^{D/2} √(log N(ε)) dε
  ```

Let `J → ∞`, so `ε_J → 0`.

Hence,

  ```
  Rad̂_S(F) ≤ (12/√n) ∫_0^{D/2} √(log N(ε, F, L²(Pₙ))) dε
           ≤ (12/√n) ∫_0^{∞} √(log N(ε, F, L²(Pₙ))) dε
  ```

where the second line uses `log N(ε) = 0` for `ε ≥ D`.

For VC classes, let `F ⊆ {0, 1}^X` have VC dimension `d ≥ 1`, so `D ≤ 1`.
Haussler's packing bound (cited) gives, for `ε ∈ (0, 1]`,
`N(ε) ≤ e(d + 1) (2e/ε²)^d`. Using `√(a + b) ≤ √a + √b`,
`1 + log(d + 1) ≤ 2d`, and `∫_0^1 √(log(1/ε)) dε = Γ(3/2) = √π/2`,

  ```
  ∫_0^{1/2} √(log N(ε)) dε ≤ ∫_0^{1/2} [ √(1 + log(d+1)) + √(d log(2e)) + √(2d log(1/ε)) ] dε
                           ≤ ½ √(2d) + ½ √(d log(2e)) + √(2d) ⋅ √π/2
                           ≤ (0.71 + 0.66 + 1.26) ⋅ √d
                           ≤ 2.63 √d
  ```

So `Rad̂_S(F) ≤ 12 ⋅ 2.63 ⋅ √(d/n) ≤ 32 √(d/n)`, with no `log n` factor, versus
`√(2d log(en/d)/n)` from Massart alone (see vc-sauer-shelah.md).

The integral is not always tight. It can exceed the truth by a log factor,
which Talagrand's generic chaining removes. The constant 12 is not optimal.

Fine print: the cover centers need not lie in `F`; a proper cover at most
changes `N(ε)` to `N(ε/2)`. The truncated form
`Rad̂_S(F) ≤ inf_{α > 0} [ 4α + (12/√n) ∫_α^{D/2} √(log N(ε)) dε ]` follows by
stopping at the `J` with `ε_{J+2} < α ≤ ε_{J+1}`, so `ε_J < 4α` (for
`α > D/4`, Cauchy–Schwarz gives `Rad̂_S(F) ≤ D < 4α` directly). It is finite
even when the integral diverges at 0. Haussler's bound (cited) is stated for
`L¹(P)` packings, `M(ε, L¹) ≤ e(d+1)(2e/ε)^d`. For `{0, 1}`-valued functions
`‖f − g‖²_{L²} = ‖f − g‖_{L¹}`, so an `L²` `ε`-packing is an `L¹` `ε²`-packing,
and covering numbers are at most packing numbers. For a 0-1 loss class,
`|1[h(x) ≠ y] − 1[h′(x) ≠ y]| = |h(x) − h′(x)|`, so the loss class has the
same covering numbers as `H`.

Regularity: the proof needs `N(ε) < ∞` for every `ε > 0` (`F` totally bounded
in `L²(Pₙ)`), `D < ∞`, and the sample fixed (the bound is on the empirical
Rademacher complexity; take `𝔼_S` for `Radₙ`). The VC corollary needs `d ≥ 1`
and `{0, 1}`-valued `F`.
