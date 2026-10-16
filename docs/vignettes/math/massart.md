# Massart's Finite-Class Lemma

Massart's lemma follows from two facts:

1. Jensen moves the max inside an exponential, where a sum bounds it, and,
2. Hoeffding's lemma bounds each sign's moment generating function.

Let,

  ```
  A ⊂ ℝⁿ                     finite set, |A| ≥ 2
  r      ≝ max_{a ∈ A} ‖a‖₂
  σ      ≝ (σ₁..σₙ)          iid uniform on {−1, +1}
  Rad(A) ≝ 𝔼_σ[ max_{a ∈ A} (1/n) Σᵢ σᵢ aᵢ ]
  ```

For a function class on a fixed sample, `A = {(f(z₁)..f(zₙ)) : f ∈ F}` and
`Rad(A) = Rad̂_S(F)`.

First, for any `λ > 0`, exponentiate and bound the max by a sum,

  ```
  exp(λ ⋅ 𝔼 max_a Σᵢ σᵢ aᵢ) ≤ 𝔼 exp(λ ⋅ max_a Σᵢ σᵢ aᵢ)
                            = 𝔼 max_a exp(λ Σᵢ σᵢ aᵢ)
                            ≤ Σ_{a ∈ A} 𝔼 exp(λ Σᵢ σᵢ aᵢ)
                            = Σ_{a ∈ A} Πᵢ 𝔼 exp(λ σᵢ aᵢ)
  ```

where the first step is Jensen (see jensen.md) for convex `exp`, the second
uses that `exp` is increasing, the third bounds a max of nonnegative terms by
their sum, and the fourth uses independence of the `σᵢ`.

Second, bound each factor. `σᵢ aᵢ` has mean zero and lies in
`[−|aᵢ|, |aᵢ|]`, an interval of width `2|aᵢ|`. Hoeffding's lemma (see
hoeffding.md) gives,

  ```
  𝔼 exp(λ σᵢ aᵢ) ≤ exp(λ² (2|aᵢ|)² / 8)
                 = exp(λ² aᵢ² / 2)
  ```

So,

  ```
  Σ_{a ∈ A} Πᵢ 𝔼 exp(λ σᵢ aᵢ) ≤ Σ_{a ∈ A} exp(λ² ‖a‖₂² / 2)
                              ≤ |A| ⋅ exp(λ² r² / 2)
  ```

Third, take logs and divide by `λ`,

  ```
  𝔼 max_a Σᵢ σᵢ aᵢ ≤ log|A| / λ + λ r² / 2
  ```

Choose `λ = √(2 log|A|) / r`, which balances the two terms,

  ```
  log|A| / λ + λ r² / 2 = r ⋅ √(log|A| / 2) + r ⋅ √(log|A| / 2)
                        = r ⋅ √(2 log|A|)
  ```

Hence, dividing by `n`,

  ```
  Rad(A) ≤ r ⋅ √(2 log|A|) / n
  ```

If every coordinate is in `[0, 1]`, then `r ≤ √n` and
`Rad(A) ≤ √(2 log|A| / n)`. The bound is tight up to a constant: for
`A = {−1, +1}ⁿ`, `Rad(A) = 1` while the bound is `√(2 log 2) ≈ 1.18`.

Regularity: the proof needs `A` finite with `|A| ≥ 2` (for `|A| = 1`,
`Rad(A) = 0` directly), `σᵢ` independent and symmetric, and `A` fixed, i.e.,
not a function of `σ`.
