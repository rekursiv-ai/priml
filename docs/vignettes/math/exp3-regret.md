# EXP3 Regret

EXP3's regret bound follows from two facts:

1. the Hedge potential argument (hedge-regret.md) still works with
   nonnegative unbounded losses, at a cost of `η²/2` times a second moment,
   and,
2. importance weighting makes the loss estimates unbiased, with second moment
   at most `K` per round.

Let,

  ```
  ℓ_t ∈ [0,1]^K                 loss vector at round t, fixed in advance
  p_{t,i}  ≝ w_{t,i} / W_t      w_{t,i} ≝ exp(−η L̂_{t−1,i}),   W_t ≝ Σᵢ w_{t,i}
  A_t ∼ p_t                     the arm played; only ℓ_{t,A_t} is observed
  ℓ̂_{t,i}  ≝ ℓ_{t,i} 1{A_t = i} / p_{t,i}
  L̂_{t,i}  ≝ Σ_{u≤t} ℓ̂_{u,i},   L_{T,i} ≝ Σ_{t≤T} ℓ_{t,i}
  Regret_T ≝ 𝔼[ Σ_t ℓ_{t,A_t} ] − minᵢ L_{T,i}
  ```

Let `𝓕_{t−1}` be the history `A_1, …, A_{t−1}`; `p_t` is `𝓕_{t−1}`-measurable.
`Φ_t ≝ log W_t`, so `Φ_1 = log K`, and `p_{t,i} > 0` always.

First, the estimates are unbiased, with bounded second moment. Given
`𝓕_{t−1}`,

  ```
  𝔼[ ℓ̂_{t,i} | 𝓕_{t−1} ] = p_{t,i} ⋅ ℓ_{t,i} / p_{t,i}
                         = ℓ_{t,i}
  ```

and

  ```
  𝔼[ Σᵢ p_{t,i} ℓ̂²_{t,i} | 𝓕_{t−1} ] = Σᵢ p_{t,i} ⋅ p_{t,i} ⋅ ℓ²_{t,i} / p²_{t,i}
                                     = Σᵢ ℓ²_{t,i}
                                     ≤ K
  ```

Second, the per-round potential drop. For `x ≥ 0`, `e^{−x} ≤ 1 − x + x²/2`
(the difference is `0` at `x = 0` and has derivative `x − 1 + e^{−x} ≥ 0`).
Since `ℓ̂ ≥ 0`,

  ```
  Φ_{t+1} − Φ_t = log Σᵢ p_{t,i} e^{−η ℓ̂_{t,i}}
                ≤ Σᵢ p_{t,i} e^{−η ℓ̂_{t,i}} − 1
                ≤ Σᵢ p_{t,i} ( 1 − η ℓ̂_{t,i} + η² ℓ̂²_{t,i} / 2 ) − 1
                = −η ⟨p_t, ℓ̂_t⟩ + (η²/2) Σᵢ p_{t,i} ℓ̂²_{t,i}
  ```

where the second step is `log y ≤ y − 1`.

Third, telescope and lower bound by any arm `i`, as in hedge-regret.md,

  ```
  −η L̂_{T,i} − log K ≤ Φ_{T+1} − Φ_1
                     ≤ −η Σ_t ⟨p_t, ℓ̂_t⟩ + (η²/2) Σ_t Σⱼ p_{t,j} ℓ̂²_{t,j}
  ```

so, dividing by `η`,

  ```
  Σ_t ⟨p_t, ℓ̂_t⟩ − L̂_{T,i} ≤ log K / η + (η/2) Σ_t Σⱼ p_{t,j} ℓ̂²_{t,j}
  ```

Fourth, take expectations. By the tower rule and the first step,
`𝔼⟨p_t, ℓ̂_t⟩ = 𝔼⟨p_t, ℓ_t⟩ = 𝔼[ℓ_{t,A_t}]` and `𝔼 L̂_{T,i} = L_{T,i}`.
Let `i` be the best arm, fixed because `ℓ` is fixed in advance,

  ```
  Regret_T = 𝔼[ Σ_t ⟨p_t, ℓ̂_t⟩ − L̂_{T,i} ]
           ≤ log K / η + (η/2) Σ_t 𝔼[ Σⱼ p_{t,j} ℓ̂²_{t,j} ]
           ≤ log K / η + η K T / 2
  ```

Hence, with `η = √(2 log K / (K T))`,

  ```
  Regret_T ≤ log K / η + η K T / 2
           = √(K T log K / 2) + √(K T log K / 2)
           = √(2 T K log K)
  ```

The price of bandit feedback is the factor `K` in the second moment: with full
information, the same proof has `Σᵢ p_{t,i} ℓ²_{t,i} ≤ 1` in its place.
The lower bound for adversarial bandits is `Ω(√(KT))` (Auer et al. 2002b), so
the bound is tight up to `√log K`.

Fine print: the variance bound uses that `ℓ̂` only enters through
`p_{t,i} ℓ̂²_{t,i}`, whose `1/p_{t,i}` cancels one `p_{t,i}`; no explicit
exploration mixing is needed for losses. For rewards, EXP3 as in Auer et al.
mixes in uniform exploration, because `e^{+x}` has no such quadratic bound for
large `x`. Against an adaptive adversary (`ℓ_t` chosen from `𝓕_{t−1}`), the
same steps bound the pseudo-regret `maxᵢ 𝔼[Σ_t ℓ_{t,A_t} − L_{T,i}]`. The
regret bound holds in expectation only; EXP3.P and EXP3-IX give
high-probability bounds. Tsallis-INF removes the `√log K`.

Regularity: losses in `[0,1]`, fixed before play; `η > 0`; and `T` known to
set `η` (the doubling trick removes this at a constant-factor cost).
