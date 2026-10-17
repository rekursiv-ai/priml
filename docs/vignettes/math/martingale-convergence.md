# Martingale Convergence

Martingale convergence follows from two facts:

1. a submartingale cannot cross a band `[a, b]` infinitely often without
   unbounded mean (Doob's upcrossing inequality), and,
2. a sequence crossing no rational band infinitely often has a limit.

Let `(Xₙ)ₙ≥₀` be a submartingale for `(ℱₙ)`, with `K = supₙ 𝔼|Xₙ| < ∞`,
and fix `a < b`. Let,

  ```
  Yₙ ≝ (Xₙ − a)⁺
  Uₙ ≝ number of completed upcrossings of [a, b] by X₀, …, Xₙ
  U∞ ≝ limₙ Uₙ
  ```

An upcrossing goes from a time with `X ≤ a` to a later time with `X ≥ b`.
For a supermartingale `X`, apply everything to the submartingale `−X`.

First, `Y` is a submartingale. `y ↦ (y − a)⁺` is convex and
nondecreasing, so by conditional Jensen (jensen.md),

  ```
  𝔼[ Yₙ | ℱₙ₋₁ ] ≥ ( 𝔼[ Xₙ | ℱₙ₋₁ ] − a )⁺
                 ≥ ( Xₙ₋₁ − a )⁺
                 = Yₙ₋₁
  ```

where the second step uses the submartingale property and monotonicity.
Upcrossings of `[a, b]` by `X` are exactly upcrossings of `[0, b − a]` by
`Y`.

Second, the betting strategy. Bet one unit from each visit to `0` until
the next visit to `[b − a, ∞)`. With `τ₀ = −1`, define stopping times,

  ```
  σᵢ ≝ inf{ k > τᵢ₋₁ : Yₖ = 0 }
  τᵢ ≝ inf{ k > σᵢ : Yₖ ≥ b − a }
  Hₖ ≝ 1[ σᵢ < k ≤ τᵢ for some i ]
  ```

`{σᵢ < k ≤ τᵢ} = {σᵢ ≤ k−1} ∖ {τᵢ ≤ k−1}` is in `ℱₖ₋₁`, so `H` is
predictable. Each completed upcrossing adds `Y_τᵢ − Y_σᵢ ≥ b − a` to
`Σ Hₖ ΔYₖ`, and a final incomplete one adds `Yₙ − Y_σᵢ = Yₙ ≥ 0`,

  ```
  Σₖ₌₁ⁿ Hₖ ( Yₖ − Yₖ₋₁ ) ≥ (b − a) ⋅ Uₙ
  ```

Third, the complementary bet has nonnegative mean. Since `1 − Hₖ ∈ {0,1}`
is `ℱₖ₋₁`-measurable and `Y` is a submartingale,

  ```
  𝔼[ (1 − Hₖ)(Yₖ − Yₖ₋₁) ] = 𝔼[ (1 − Hₖ) ⋅ 𝔼[ Yₖ − Yₖ₋₁ | ℱₖ₋₁ ] ]
                           ≥ 0
  ```

Fourth, the upcrossing inequality. Combine the second and third steps,

  ```
  (b − a) ⋅ 𝔼[Uₙ] ≤ 𝔼[ Σₖ Hₖ (Yₖ − Yₖ₋₁) ]
                  = 𝔼[ Yₙ − Y₀ ] − Σₖ 𝔼[ (1 − Hₖ)(Yₖ − Yₖ₋₁) ]
                  ≤ 𝔼[ Yₙ − Y₀ ]
                  ≤ 𝔼[ (Xₙ − a)⁺ ]
                  ≤ K + |a|
  ```

where the second line telescopes `Σₖ (Yₖ − Yₖ₋₁) = Yₙ − Y₀`, the fourth
drops `Y₀ ≥ 0`, and the fifth uses `(x − a)⁺ ≤ |x| + |a|`.

Fifth, finitely many upcrossings. `Uₙ ↑ U∞`, so by monotone convergence
`𝔼[U∞] ≤ (K + |a|)/(b − a) < ∞`, hence `U∞ < ∞` a.s. Oscillation forces
infinitely many upcrossings of some rational band,

  ```
  { liminf Xₙ < limsup Xₙ } = ⋃_{a<b ∈ ℚ} { liminf Xₙ < a < b < limsup Xₙ }
                            ⊆ ⋃_{a<b ∈ ℚ} { U∞[a,b] = ∞ }
  ```

a countable union of null sets. So `X∞ = lim Xₙ` exists in `[−∞, ∞]` a.s.

Sixth, the limit is finite. By Fatou,

  ```
  𝔼|X∞| = 𝔼[ liminf |Xₙ| ]
        ≤ liminf 𝔼|Xₙ|
        ≤ K
  ```

so `|X∞| < ∞` a.s.

Hence,

  ```
  supₙ 𝔼|Xₙ| < ∞  ⟹  Xₙ → X∞  a.s.,   𝔼|X∞| ≤ supₙ 𝔼|Xₙ|
  ```

Now let `(Mₙ)` be a martingale with `supₙ 𝔼[Mₙ²] < ∞`. It is
`L¹`-bounded by Cauchy–Schwarz, so `Mₙ → M∞` a.s. Seventh, for `n < m`
the past is orthogonal to the future increment,
`𝔼[ Mₙ (Mₘ − Mₙ) ] = 𝔼[ Mₙ ⋅ 𝔼[ Mₘ − Mₙ | ℱₙ ] ] = 0`, so,

  ```
  𝔼[Mₘ²] = 𝔼[ (Mₙ + (Mₘ − Mₙ))² ]
         = 𝔼[Mₙ²] + 2 𝔼[ Mₙ (Mₘ − Mₙ) ] + 𝔼[ (Mₘ − Mₙ)² ]
         = 𝔼[Mₙ²] + 𝔼[ (Mₘ − Mₙ)² ]
  ```

So `𝔼[Mₙ²]` is nondecreasing and bounded, with limit `S`. Eighth, pass
`m → ∞` with Fatou,

  ```
  𝔼[ (M∞ − Mₙ)² ] = 𝔼[ liminfₘ (Mₘ − Mₙ)² ]
                  ≤ liminfₘ 𝔼[ (Mₘ − Mₙ)² ]
                  = liminfₘ ( 𝔼[Mₘ²] − 𝔼[Mₙ²] )
                  = S − 𝔼[Mₙ²]
  ```

and the right side `→ 0` as `n → ∞`.

Hence,

  ```
  supₙ 𝔼[Mₙ²] < ∞  ⟹  Mₙ → M∞  a.s. and in L²
  ```

`L¹`-boundedness gives a.s. convergence but not `L¹` convergence: the
product martingale `Mₙ = Πₖ₌₁ⁿ ξₖ`, with iid `ξₖ ∈ {0, 2}` equally
likely, has `𝔼Mₙ = 1` but `Mₙ → 0` a.s. `L¹` convergence needs uniform
integrability, which `L²`-boundedness supplies.

Fine print: since `Y ≥ 0`, an upcrossing of `[0, b − a]` begins exactly
at a time with `Yₖ = 0`, i.e., `Xₖ ≤ a`, so `σᵢ, τᵢ` count every
upcrossing. Convergence in `L¹` under uniform integrability, and in `Lᵖ`
for `p > 1` via doob-maximal.md, are cited, not proved.

Regularity: `𝔼|Xₙ| < ∞` for every `n` and `supₙ 𝔼|Xₙ| < ∞`; conditional
Jensen for the convex `(· − a)⁺`; `𝔼[Mₙ²] < ∞` so `Mₙ (Mₘ − Mₙ)` is
integrable (Cauchy–Schwarz) in the orthogonality step.
