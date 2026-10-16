# Weak Law of Large Numbers

The weak law follows from two facts:

1. the variance of a mean of uncorrelated terms shrinks as `1/n`, and,
2. Chebyshev's inequality turns small variance into small tail probability.

Let,

  ```
  X₁, X₂, …   identically distributed, pairwise uncorrelated
  μ  ≝ 𝔼[Xᵢ],  σ² ≝ 𝕍(Xᵢ) < ∞
  X̄ₙ ≝ (1/n) Σᵢ₌₁ⁿ Xᵢ
  ε  > 0
  ```

First, the sample mean is unbiased,

  ```
  𝔼[X̄ₙ] = (1/n) Σᵢ 𝔼[Xᵢ]
        = (1/n) ⋅ n μ
        = μ
  ```

Second, its variance is `σ²/n`,

  ```
  𝕍(X̄ₙ) = (1/n²) ⋅ 𝕍( Σᵢ Xᵢ )
        = (1/n²) ⋅ ( Σᵢ 𝕍(Xᵢ) + Σ_{i≠j} 𝕍(Xᵢ, Xⱼ) )
        = (1/n²) ⋅ ( n σ² + 0 )
        = σ² / n
  ```

where the third step uses that the `Xᵢ` are pairwise uncorrelated.

Third, apply Chebyshev to `X̄ₙ`,

  ```
  P( |X̄ₙ − μ| ≥ ε ) ≤ 𝕍(X̄ₙ) / ε²
                    = σ² / (n ε²)
                    → 0   as n → ∞
  ```

Hence,

  ```
  X̄ₙ → μ   in probability
  ```

The `1/n` rate from Chebyshev is crude: for bounded `Xᵢ`, Hoeffding gives
`exp(−2nε²/(b − a)²)`. Finite variance is not necessary: Khinchin's WLLN
needs only `𝔼|Xᵢ| < ∞` for iid `Xᵢ`, via truncation (cited, not proved).

Regularity: the proof needs a common mean `μ`, a common finite variance
`σ²`, and pairwise uncorrelatedness. Independence and identical
distribution beyond the first two moments are not used.
