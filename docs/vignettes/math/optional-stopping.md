# Optional Stopping

Optional stopping follows from one fact: a stopped martingale is a sum of
martingale increments, each multiplied by a predictable indicator, and each
such term has zero mean.

Let,

  ```
  F₀ ⊆ F₁ ⊆ …              a filtration
  M₀, M₁, …                a martingale: 𝔼|Mₖ| < ∞, 𝔼[Mₖ | Fₖ₋₁] = Mₖ₋₁
  Dₖ ≝ Mₖ − Mₖ₋₁
  τ                        a stopping time: {τ ≤ k} ∈ Fₖ for all k
  τ  ≤ N a.s.              for a constant N
  ```

First, write `M_τ` as a telescoping sum. For `τ ≤ N`,

  ```
  M_τ = M₀ + Σₖ₌₁^τ Dₖ
      = M₀ + Σₖ₌₁ᴺ 1{τ ≥ k} ⋅ Dₖ
  ```

Second, the indicator is predictable,

  ```
  {τ ≥ k} = {τ ≤ k − 1}ᶜ ∈ Fₖ₋₁
  ```

Third, each term has zero mean,

  ```
  𝔼[ 1{τ ≥ k} ⋅ Dₖ ] = 𝔼[ 𝔼[ 1{τ ≥ k} ⋅ Dₖ | Fₖ₋₁ ] ]
                     = 𝔼[ 1{τ ≥ k} ⋅ 𝔼[ Dₖ | Fₖ₋₁ ] ]
                     = 𝔼[ 1{τ ≥ k} ⋅ 0 ]
                     = 0
  ```

where the first step is the tower property, the second pulls out the
`Fₖ₋₁`-measurable indicator, and the third is the martingale property.

Fourth, sum the `N` terms,

  ```
  𝔼[M_τ] = 𝔼[M₀] + Σₖ₌₁ᴺ 𝔼[ 1{τ ≥ k} ⋅ Dₖ ]
         = 𝔼[M₀] + 0
  ```

Hence, for bounded `τ`,

  ```
  𝔼[M_τ] = 𝔼[M₀]
  ```

For unbounded `τ`, apply the bounded case to `τ ∧ n`, which gives
`𝔼[M_{τ∧n}] = 𝔼[M₀]`, and pass `n → ∞`. Any of these suffices (each cited
except the first, which is proved here):

1. `τ < ∞` a.s. and `|M_{τ∧n}| ≤ K` for all `n`: then `M_{τ∧n} → M_τ` a.s.,
   and dominated convergence gives `𝔼[M_τ] = lim 𝔼[M_{τ∧n}] = 𝔼[M₀]`.
2. `𝔼[τ] < ∞` and `𝔼[|Dₖ| | Fₖ₋₁] ≤ c` for all `k`.
3. `τ < ∞` a.s. and `(M_{τ∧n})ₙ` uniformly integrable.

Some condition is needed. For a simple symmetric random walk from `0` and
`τ` the first hitting time of `1`, `τ < ∞` a.s. but `𝔼[M_τ] = 1 ≠ 0`.

Gambler's ruin. Let `Sₙ ≝ x + Σᵢ₌₁ⁿ ξᵢ` with iid `ξᵢ = ±1` with probability
`1/2` each, `0 < x < N`, and `τ` the first time `S` hits `0` or `N`.
Then `τ < ∞` a.s.: any `N` consecutive `+1` steps end the game, so
`P(τ > kN) ≤ (1 − 2^{−N})ᵏ → 0`. Also `|S_{τ∧n}| ≤ N`, so condition 1
applies. With `P = P(S_τ = N)`,

  ```
  x = 𝔼[S₀]
    = 𝔼[S_τ]
    = N ⋅ P + 0 ⋅ (1 − P)
  ```

so `P(S_τ = N) = x / N`. The process `Sₙ² − n` is also a martingale, since
`𝔼[S²ₙ − S²ₙ₋₁ | Fₙ₋₁] = 𝔼[2Sₙ₋₁ξₙ + ξₙ²] = 1`. The bounded case at `τ ∧ n`
gives `𝔼[S²_{τ∧n}] = x² + 𝔼[τ ∧ n]`. Bounded convergence on the left and
monotone convergence on the right give,

  ```
  𝔼[τ] = 𝔼[S_τ²] − x²
       = N² ⋅ (x/N) − x²
       = x (N − x)
  ```

Regularity: the bounded case needs `M` a martingale (integrable, adapted),
`τ` a stopping time, and `τ ≤ N` a.s. The unbounded extensions need one of
the listed conditions. For a submartingale the same proof gives
`𝔼[M_τ] ≥ 𝔼[M₀]`.
