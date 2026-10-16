# UCB Regret

UCB1's regret bound (Auer, Cesa-Bianchi & Fischer 2002) follows from two
facts:

1. a suboptimal arm `i` beats the optimal arm only if a confidence bound fails
   or arm `i` has fewer than `8 log T / Δᵢ²` pulls, and,
2. by Hoeffding, each confidence bound fails with probability at most `t⁻⁴`,
   which is summable.

Let,

  ```
  μᵢ ∈ [0,1]           mean of arm i ∈ {1..K};   μ* ≝ μ_{i*} = maxᵢ μᵢ;   Δᵢ ≝ μ* − μᵢ
  X_{i,1}, X_{i,2}, …  iid rewards of arm i in [0,1], independent across arms
  μ̂_{i,s}             mean of X_{i,1}, …, X_{i,s}
  Tᵢ(t)                pulls of arm i in rounds 1..t
  c_{t,s}              ≝ √(2 log t / s)
  A_t                  ≝ t for t ≤ K;   else A_t ∈ argmaxᵢ μ̂_{i,Tᵢ(t−1)} + c_{t,Tᵢ(t−1)}
  R_T                  ≝ T μ* − 𝔼[ Σ_{t≤T} X_{A_t, T_{A_t}(t)} ]
  ```

The `s`-th pull of arm `i` returns `X_{i,s}`, which is independent of the
history before that pull.

First, regret counts pulls. `A_t` is fixed by the history, and the reward it
returns has mean `μ_{A_t}` given that history, so

  ```
  R_T = T μ* − Σ_t 𝔼[ μ_{A_t} ]
      = Σᵢ 𝔼[Tᵢ(T)] μ* − Σᵢ 𝔼[Tᵢ(T)] μᵢ
      = Σᵢ Δᵢ 𝔼[Tᵢ(T)]
  ```

using `Σᵢ Tᵢ(T) = T`.

Second, split the pulls of a suboptimal arm `i` at `ℓ = ⌈8 log T / Δᵢ²⌉`. At
most `ℓ` pulls happen while `Tᵢ(t−1) < ℓ`. A later pull at round `t > K`
needs arm `i`'s index to reach `i*`'s. Taking a union over the unknown counts
`s = T_{i*}(t−1)` and `sᵢ = Tᵢ(t−1)`,

  ```
  Tᵢ(T) ≤ ℓ + Σ_{t=K+1}^{T} 1{ A_t = i, Tᵢ(t−1) ≥ ℓ }
        ≤ ℓ + Σ_{t=K+1}^{T} Σ_{s=1}^{t−1} Σ_{sᵢ=ℓ}^{t−1} 1{ μ̂_{i*,s} + c_{t,s} ≤ μ̂_{i,sᵢ} + c_{t,sᵢ} }
  ```

Third, that index comparison forces one of three events,

  ```
  (a)  μ̂_{i*,s} ≤ μ* − c_{t,s}
  (b)  μ̂_{i,sᵢ} ≥ μᵢ + c_{t,sᵢ}
  (c)  μ* < μᵢ + 2 c_{t,sᵢ}
  ```

If (a) and (b) both fail,

  ```
  μ* < μ̂_{i*,s} + c_{t,s}
     ≤ μ̂_{i,sᵢ} + c_{t,sᵢ}
     < μᵢ + 2 c_{t,sᵢ}
  ```

which is (c). And (c) is impossible for `sᵢ ≥ ℓ` and `t ≤ T`,

  ```
  2 c_{t,sᵢ} = 2 √(2 log t / sᵢ)
             ≤ 2 √(2 log T ⋅ Δᵢ² / (8 log T))
             = Δᵢ
  ```

Fourth, bound (a) and (b) by Hoeffding (hoeffding.md). With `s` fixed,
`μ̂_{i*,s}` is the mean of `s` iid draws in `[0,1]`, so

  ```
  P(a) ≤ exp( −2 s c_{t,s}² )
       = exp( −4 log t )
       = t⁻⁴
  ```

and `P(b) ≤ t⁻⁴` the same way. Taking expectations in the second step,

  ```
  𝔼[Tᵢ(T)] ≤ ℓ + Σ_{t=K+1}^{T} Σ_{s=1}^{t−1} Σ_{sᵢ=ℓ}^{t−1} 2 t⁻⁴
           ≤ ℓ + Σ_{t≥1} 2 t² ⋅ t⁻⁴
           = ℓ + π²/3
           ≤ 8 log T / Δᵢ² + 1 + π²/3
  ```

Hence, by the first step,

  ```
  R_T ≤ Σ_{i: Δᵢ>0} ( 8 log T / Δᵢ + (1 + π²/3) Δᵢ )
  ```

Fifth, the gap-free bound. Fix `δ > 0`. Arms with `Δᵢ ≤ δ` cost at most
`δ Σᵢ 𝔼[Tᵢ(T)] ≤ δ T`. Each arm with `Δᵢ > δ` costs at most
`8 log T / δ + 1 + π²/3`, since `Δᵢ ≤ 1`. So,

  ```
  R_T ≤ δ T + 8 K log T / δ + (1 + π²/3) K
      = 2 √(8 K T log T) + (1 + π²/3) K           (δ = √(8 K log T / T))
      = √(32 K T log T) + (1 + π²/3) K
  ```

The `log T / Δᵢ` rate is optimal up to constants. Lai & Robbins (1985) show
any consistent policy has `R_T ≳ Σᵢ Δᵢ log T / 𝕂(μᵢ, μ*)`. For Bernoulli
arms, `𝕂(μᵢ, μ*) ≤ Δᵢ² / (μ*(1 − μ*))` (KL is at most χ²), so the lower
bound is at least `Σᵢ μ*(1 − μ*) log T / Δᵢ`.

Fine print: the constant 8 is loose; a tighter analysis of UCB1 gives
`(2 + o(1)) log T / Δᵢ`, and KL-UCB matches the Lai–Robbins constant. The
minimax lower bound is `Ω(√(KT))` (Auer et al. 2002b); MOSS removes the
`√log T`. Ties in `argmax` may be broken arbitrarily, since the second step
only uses that `i`'s index is at least `i*`'s. The bound needs no knowledge of
`T`: the index uses `log t`, and `T` enters only the analysis.

Regularity: rewards in `[0,1]`, iid per arm and independent across arms;
`T ≥ K` and `T ≥ 2` so that `log T > 0`. Hoeffding needs only boundedness; a
range `[a,b]` scales `c_{t,s}` by `b − a`.
