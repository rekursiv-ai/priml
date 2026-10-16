# Bootstrap Consistency for the Mean

Bootstrap consistency follows from two facts:

1. given the data, resampling gives a triangular array of iid draws, and,
2. the SLLN makes that array satisfy Lindeberg for almost every data sequence.

Let,

  ```
  X₁, X₂, … iid ~ F,   μ ≝ 𝔼[X],   σ² ≝ 𝕍(X) ∈ (0, ∞)
  X̄ₙ  ≝ (1/n) Σⱼ Xⱼ
  σ̂ₙ² ≝ (1/n) Σⱼ (Xⱼ − X̄ₙ)²
  X*₁, …, X*ₙ   iid uniform draws from {X₁, …, Xₙ}, given the data
  X̄*ₙ ≝ (1/n) Σᵢ X*ᵢ
  𝔼*, P*        expectation, probability given X₁, …, Xₙ
  Zₙᵢ ≝ (X*ᵢ − X̄ₙ) / √n
  ```

The claim: for almost every data sequence, under `P*`,
`√n (X̄*ₙ − X̄ₙ) = Σᵢ Zₙᵢ → N(0, σ²)` in distribution.

First, each row has mean zero and variance `σ̂ₙ²` given the data,

  ```
  𝔼*[ X*ᵢ − X̄ₙ ] = (1/n) Σⱼ Xⱼ − X̄ₙ
                 = 0
  Σᵢ 𝔼*[ Zₙᵢ² ]  = n ⋅ (1/n) ⋅ (1/n) Σⱼ (Xⱼ − X̄ₙ)²
                 = σ̂ₙ²
  ```

Second, `σ̂ₙ² → σ²` almost surely. Apply the SLLN to `Xⱼ` and `Xⱼ²`,

  ```
  σ̂ₙ² = (1/n) Σⱼ Xⱼ² − X̄ₙ²
      → 𝔼[X²] − μ²
      = σ²
  ```

Third, the Lindeberg term vanishes almost surely. Fix `ε > 0`. Given the data,
the `Zₙᵢ` are iid, each uniform over `(Xⱼ − X̄ₙ)/√n`,

  ```
  Lₙ(ε) ≝ Σᵢ 𝔼*[ Zₙᵢ² ⋅ 1{|Zₙᵢ| > ε} ]
        = (1/n) Σⱼ (Xⱼ − X̄ₙ)² ⋅ 1{|Xⱼ − X̄ₙ| > ε√n}
  ```

Fix an integer `M`. By the SLLN, `|X̄ₙ − μ| ≤ 1` eventually, and then
`ε√n ≥ M + 1` eventually. On both events, `|Xⱼ − X̄ₙ| > ε√n` implies
`|Xⱼ − μ| > ε√n − 1 ≥ M`. Using `(u + v)² ≤ 2u² + 2v²`,

  ```
  Lₙ(ε) ≤ (1/n) Σⱼ (2 (Xⱼ − μ)² + 2 (X̄ₙ − μ)²) ⋅ 1{|Xⱼ − μ| > M}
        ≤ 2 ⋅ (1/n) Σⱼ (Xⱼ − μ)² ⋅ 1{|Xⱼ − μ| > M} + 2 (X̄ₙ − μ)²
        → 2 ⋅ 𝔼[ (X − μ)² ⋅ 1{|X − μ| > M} ] + 0
  ```

where the limit applies the SLLN to the iid terms
`(Xⱼ − μ)² 1{|Xⱼ − μ| > M}`, which have finite mean `≤ σ²`, and to `X̄ₙ`. The
countably many `M ∈ ℕ` give countably many probability-one events. On their
intersection,

  ```
  limsup_n Lₙ(ε) ≤ 2 ⋅ 𝔼[ (X − μ)² ⋅ 1{|X − μ| > M} ]   for every M
                 → 0   as M → ∞
  ```

by dominated convergence with dominator `(X − μ)²`. Repeating for rational
`ε` and using that `Lₙ(ε)` is nonincreasing in `ε` covers every `ε > 0`.

Fourth, apply Lindeberg–Feller (cited). Fix a data sequence in the
probability-one event where the second and third steps hold. Under `P*`, row
`n` is `Zₙ₁, …, Zₙₙ`: independent, mean zero, total variance `σ̂ₙ² → σ²`,
and `Lₙ(ε) → 0` for every `ε > 0`. Lindeberg–Feller gives,

  ```
  √n (X̄*ₙ − X̄ₙ) = Σᵢ Zₙᵢ
                → N(0, σ²)   in P*-distribution
  ```

Hence,

  ```
  sup_t | P*( √n (X̄*ₙ − X̄ₙ) ≤ t ) − Φ(t/σ) | → 0   almost surely
  ```

and, by the classical CLT, `√n (X̄ₙ − μ)` has the same `N(0, σ²)` limit. The
uniform-in-`t` form follows from Pólya's theorem, since `Φ` is continuous. So
the bootstrap law estimates the sampling law of the mean consistently.

A triangular-array CLT is needed because the resampling law, the empirical
distribution `Fₙ` of `X₁, …, Xₙ`, changes with `n`. The iid CLT for a fixed
law does not apply. The SLLN enters in two
places: `σ̂ₙ² → σ²` (second step) and `Lₙ(ε) → 0` (third step).

Regularity: the proof needs iid `Xⱼ`, `σ² ∈ (0, ∞)` (finite second moment for
the SLLN on `Xⱼ²` and the Lindeberg bound, positive for a nondegenerate
limit), resampling with replacement of size `n`, and Lindeberg–Feller, SLLN,
and Pólya's theorem, all cited.
