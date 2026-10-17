# Doob Decomposition

The Doob decomposition follows from two facts:

1. each increment splits into its predictable part `𝔼[ΔXₖ | ℱₖ₋₁]` and a
   zero-mean remainder, and,
2. a predictable martingale is constant.

Let `(ℱₙ)ₙ≥₀` be a filtration, `(Xₙ)` adapted with `𝔼|Xₙ| < ∞`, and,

  ```
  ΔXₖ ≝ Xₖ − Xₖ₋₁
  Aₙ  ≝ Σₖ₌₁ⁿ 𝔼[ ΔXₖ | ℱₖ₋₁ ],              A₀ ≝ 0
  Mₙ  ≝ Σₖ₌₁ⁿ ( ΔXₖ − 𝔼[ ΔXₖ | ℱₖ₋₁ ] ),     M₀ ≝ 0
  ```

A process `(Aₙ)` is predictable if `Aₙ` is `ℱₙ₋₁`-measurable for `n ≥ 1`.

First, the pieces add up. Summing the two definitions term by term,

  ```
  X₀ + Mₙ + Aₙ = X₀ + Σₖ₌₁ⁿ ( ΔXₖ − 𝔼[ΔXₖ|ℱₖ₋₁] ) + Σₖ₌₁ⁿ 𝔼[ΔXₖ|ℱₖ₋₁]
               = X₀ + Σₖ₌₁ⁿ ΔXₖ
               = Xₙ
  ```

Second, `A` is predictable, since each summand of `Aₙ` is
`ℱₖ₋₁ ⊆ ℱₙ₋₁`-measurable. And `M` is a martingale, since `Mₙ₋₁` is
`ℱₙ₋₁`-measurable and,

  ```
  𝔼[ Mₙ − Mₙ₋₁ | ℱₙ₋₁ ] = 𝔼[ ΔXₙ − 𝔼[ΔXₙ|ℱₙ₋₁] | ℱₙ₋₁ ]
                        = 𝔼[ΔXₙ|ℱₙ₋₁] − 𝔼[ΔXₙ|ℱₙ₋₁]
                        = 0
  ```

where the second step pulls out the `ℱₙ₋₁`-measurable term.

Third, uniqueness. Suppose `Xₙ = X₀ + M′ₙ + A′ₙ` is another such split, with
`M′₀ = A′₀ = 0`. Then `Dₙ = Mₙ − M′ₙ = A′ₙ − Aₙ` is both a martingale and
predictable, so,

  ```
  Dₙ = 𝔼[ Dₙ | ℱₙ₋₁ ]
     = Dₙ₋₁
     = ⋯
     = D₀
     = 0
  ```

where the first step uses predictability and the second the martingale
property.

Fourth, `A` is increasing iff `X` is a submartingale. Both say the same
thing about each conditional increment,

  ```
  Aₙ − Aₙ₋₁ ≥ 0  ⟺  𝔼[ ΔXₙ | ℱₙ₋₁ ] ≥ 0
               ⟺  𝔼[ Xₙ | ℱₙ₋₁ ] ≥ Xₙ₋₁
  ```

Hence,

  ```
  Xₙ = X₀ + Mₙ + Aₙ,     M martingale, A predictable, M₀ = A₀ = 0
  ```

and the split is unique a.s. `A` is nondecreasing iff `X` is a
submartingale, nonincreasing iff a supermartingale, and zero iff a
martingale.

For example, let `(Mₙ)` be a martingale with `𝔼[Mₙ²] < ∞`. Its square is a
submartingale, and its compensator is the predictable quadratic variation
`⟨M⟩`. Fifth, expand the increment of the square,

  ```
  𝔼[ Mₖ² − Mₖ₋₁² | ℱₖ₋₁ ] = 𝔼[ (Mₖ − Mₖ₋₁)² + 2 Mₖ₋₁ (Mₖ − Mₖ₋₁) | ℱₖ₋₁ ]
                          = 𝔼[ (Mₖ − Mₖ₋₁)² | ℱₖ₋₁ ] + 2 Mₖ₋₁ ⋅ 𝔼[ Mₖ − Mₖ₋₁ | ℱₖ₋₁ ]
                          = 𝔼[ (Mₖ − Mₖ₋₁)² | ℱₖ₋₁ ]
  ```

using `Mₖ² − Mₖ₋₁² = (Mₖ − Mₖ₋₁)² + 2 Mₖ₋₁ (Mₖ − Mₖ₋₁)`, then pulling out
`Mₖ₋₁`, then the martingale property. So,

  ```
  Mₙ² = M₀² + ( martingale ) + ⟨M⟩ₙ,     ⟨M⟩ₙ ≝ Σₖ₌₁ⁿ 𝔼[ (Mₖ − Mₖ₋₁)² | ℱₖ₋₁ ]
  ```

and `⟨M⟩` is nondecreasing. For a simple random walk with `±1` steps,
`⟨M⟩ₙ = n`, so `Mₙ² − n` is a martingale.

Fine print: uniqueness holds only up to a.s. equality, as conditional
expectations do. The normalization `M₀ = A₀ = 0` is what makes the split
unique; without it, a constant can move between `M` and `A`. The
continuous-time analogue, the Doob–Meyer decomposition, needs `X` of class
(D) and is not proved here.

Regularity: `𝔼|Xₙ| < ∞` for every `n`, so every `𝔼[ΔXₖ|ℱₖ₋₁]` exists and
`Mₙ` is integrable; `𝔼[Mₙ²] < ∞` in the example, so `(ΔMₖ)²` and
`Mₖ₋₁ ΔMₖ` are integrable (Cauchy–Schwarz) and the conditional expectations
above exist.
