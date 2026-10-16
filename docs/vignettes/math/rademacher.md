# Rademacher Generalization Bound

The bound follows from two facts:

1. a ghost sample turns `R − R̂` into a difference of two empirical means, and,
2. swapping a real point with its ghost leaves the joint law unchanged, so
   random signs can be inserted for free.

Let,

  ```
  F                            class of functions f: Z → [0, 1]
  S        ≝ (z₁..zₙ) ~ Dⁿ     iid sample
  S′       ≝ (z′₁..z′ₙ) ~ Dⁿ   iid ghost sample, independent of S
  σ        ≝ (σ₁..σₙ)          iid uniform on {−1, +1}, independent of S, S′
  P f      ≝ 𝔼_{z~D}[ f(z) ]
  Pₙ f     ≝ (1/n) Σᵢ f(zᵢ),   P′ₙ f ≝ (1/n) Σᵢ f(z′ᵢ)
  Rad̂_S(F) ≝ 𝔼_σ[ sup_f (1/n) Σᵢ σᵢ f(zᵢ) ]
  Radₙ(F)  ≝ 𝔼_S[ Rad̂_S(F) ]
  ```

For a loss class, `f(z) = ℓ(h, z)`, so `P f = R(h)` and `Pₙ f = R̂(h)`.

First, replace `P f` with the ghost mean. Since `𝔼_{S′}[P′ₙ f] = P f`,

  ```
  𝔼_S sup_f (P f − Pₙ f) = 𝔼_S sup_f 𝔼_{S′}[ P′ₙ f − Pₙ f | S ]
                         ≤ 𝔼_S 𝔼_{S′} sup_f (P′ₙ f − Pₙ f)
                         = 𝔼_{S,S′} sup_f (1/n) Σᵢ (f(z′ᵢ) − f(zᵢ))
  ```

where the second step uses that a sup of expectations is at most the
expectation of the sup (`sup_f 𝔼 X_f ≤ 𝔼 sup_f X_f`, since `X_g ≤ sup_f X_f`
for each `g`).

Second, insert signs. Fix `σ`. Swapping `zᵢ ↔ z′ᵢ` for every `i` with
`σᵢ = −1` is a measure-preserving map of `Dⁿ ⊗ Dⁿ`, because `zᵢ` and `z′ᵢ` are
iid. It sends `f(z′ᵢ) − f(zᵢ)` to `σᵢ (f(z′ᵢ) − f(zᵢ))`. So the expectation is
the same for every `σ`, and therefore equals its average over `σ`,

  ```
  𝔼_{S,S′} sup_f (1/n) Σᵢ (f(z′ᵢ) − f(zᵢ))
      = 𝔼_{S,S′,σ} sup_f (1/n) Σᵢ σᵢ (f(z′ᵢ) − f(zᵢ))
      ≤ 𝔼_{S′,σ} sup_f (1/n) Σᵢ σᵢ f(z′ᵢ) + 𝔼_{S,σ} sup_f (1/n) Σᵢ (−σᵢ) f(zᵢ)
      = Radₙ(F) + Radₙ(F)
  ```

where the second step uses `sup (a + b) ≤ sup a + sup b`, and the third uses
that `−σ` has the same law as `σ` and `S′` the same law as `S`.

Hence,

  ```
  𝔼_S sup_{f ∈ F} (P f − Pₙ f) ≤ 2 ⋅ Radₙ(F)
  ```

The same argument with `f` replaced by `−f` bounds `𝔼_S sup_f (Pₙ f − P f)`,
since `Radₙ(−F) = Radₙ(F)`.

Third, concentrate. Let `Φ(S) ≝ sup_f (P f − Pₙ f)`. Replacing one `zᵢ` by
`z̃ᵢ` changes each `Pₙ f` by `(f(z̃ᵢ) − f(zᵢ))/n ∈ [−1/n, 1/n]`, so it
changes `Φ` by at most `cᵢ = 1/n`. McDiarmid (see azuma-mcdiarmid.md) gives

  ```
  P( Φ − 𝔼Φ ≥ t ) ≤ exp(−2 t² / Σᵢ cᵢ²)
                  = exp(−2 t² / (n ⋅ 1/n²))
                  = exp(−2 n t²)
  ```

Setting this to `δ` gives `t = √(log(1/δ)/(2n))`. With probability at least
`1 − δ`, for all `f ∈ F`,

  ```
  P f − Pₙ f ≤ Φ(S)
             ≤ 𝔼Φ + √( log(1/δ) / (2n) )
             ≤ 2 ⋅ Radₙ(F) + √( log(1/δ) / (2n) )
  ```

For a data-dependent bound, `Rad̂_S(F)` also has bounded differences `1/n`:
for each fixed `σ` the sup moves by at most `1/n`, so the average over `σ`
does too. McDiarmid then gives `Radₙ(F) ≤ Rad̂_S(F) + √(log(2/δ)/(2n))` with
probability at least `1 − δ/2`. Running the bound above at `δ/2` and taking a
union bound, with probability at least `1 − δ`, for all `f ∈ F`,

  ```
  P f − Pₙ f ≤ 2 ⋅ Radₙ(F)  + √( log(2/δ) / (2n) )
             ≤ 2 ⋅ Rad̂_S(F) + 3 ⋅ √( log(2/δ) / (2n) )
  ```

The factor 2 is tight up to constants: a matching lower bound
`𝔼 sup_f |P f − Pₙ f| ≥ Radₙ(F)/2 − O(1/√n)` holds by the reverse
desymmetrization argument.

Regularity: the proof needs `S` iid, `f ∈ [0, 1]` (range `b − a` scales `cᵢ`
to `(b − a)/n`), `F` fixed before seeing `S`, and `sup_f` measurable (e.g., `F`
countable or pointwise separable) so that Fubini applies.
