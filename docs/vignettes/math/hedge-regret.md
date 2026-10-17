# Hedge Regret

Hedge's regret bound follows from two facts:

1. the log-potential `Φ_t ≝ log Σᵢ w_{t,i}` decreases each round by at most
   the learner's loss times `η`, plus `η²/8` by Hoeffding's lemma, and,
2. `Φ` can never drop below the log-weight of the best expert.

Let,

  ```
  n                          number of experts
  ℓ_t ∈ [0, 1]ⁿ              loss vector at round t, chosen adversarially
  L_{t,i}  ≝ Σ_{s ≤ t} ℓ_{s,i},    L_{0,i} ≝ 0
  w_{t,i}  ≝ exp(−η L_{t−1,i}),    η > 0
  W_t      ≝ Σᵢ w_{t,i},     Φ_t ≝ log W_t
  p_t      ≝ w_t / W_t       distribution played at round t
  Regret_T ≝ Σ_t ⟨p_t, ℓ_t⟩ − minᵢ L_{T,i}
  ```

Note `w_1 = 1`, so `Φ_1 = log n`, and `w_{t+1,i} = w_{t,i} ⋅ e^{−η ℓ_{t,i}}`.

First, bound one round's change in potential. Let `I ~ p_t`,

  ```
  Φ_{t+1} − Φ_t = log( Σᵢ w_{t,i} e^{−η ℓ_{t,i}} / W_t )
                = log Σᵢ p_{t,i} e^{−η ℓ_{t,i}}
                = log 𝔼_{I~p_t}[ e^{−η ℓ_{t,I}} ]
                ≤ −η ⋅ 𝔼_{I~p_t}[ ℓ_{t,I} ] + η² (1 − 0)² / 8
                = −η ⟨p_t, ℓ_t⟩ + η² / 8
  ```

where the fourth step is Hoeffding's lemma (see hoeffding.md),
`log 𝔼 e^{sX} ≤ s 𝔼X + s²(b − a)²/8` for `X ∈ [a, b]`, at `s = −η`.

Second, sum over `t = 1..T` and telescope,

  ```
  Φ_{T+1} − Φ_1 ≤ −η Σ_t ⟨p_t, ℓ_t⟩ + T η² / 8
  ```

Third, lower bound `Φ_{T+1}` by any single expert `i`,

  ```
  Φ_{T+1} = log Σⱼ w_{T+1,j}
          ≥ log w_{T+1,i}
          = −η L_{T,i}
  ```

where the second step drops nonnegative terms and uses that `log` is
increasing.

Combining with `Φ_1 = log n`,

  ```
  −η L_{T,i} − log n       ≤ −η Σ_t ⟨p_t, ℓ_t⟩ + T η² / 8
  Σ_t ⟨p_t, ℓ_t⟩ − L_{T,i} ≤ log n / η + η T / 8
  ```

where the second line adds `η Σ_t ⟨p_t, ℓ_t⟩ + log n` to both sides and
divides by `η > 0`.

Hence, taking `i` to be the best expert,

  ```
  Regret_T ≤ log n / η + η T / 8
  ```

Fourth, minimize over `η`. The derivative `−log n/η² + T/8` vanishes at
`η* = √(8 log n / T)`,

  ```
  log n / η* + η* T / 8 = √(T log n / 8) + √(T log n / 8)
                        = 2 ⋅ √(T log n / 8)
                        = √(T log n / 2)
  ```

So with `η = √(8 log n / T)`,

  ```
  Regret_T ≤ √(T log n / 2)
  ```

The slack is Hoeffding's lemma, which is tight to second order as `η → 0` for
losses in `{0, 1}` split evenly under `p_t`, and dropping the other experts'
weights. The bound `√(T log n / 2)` matches the minimax regret, including the
constant, as `n, T → ∞`.

Regularity: the proof needs `ℓ_{t,i} ∈ [0, 1]` (a range `[a, b]` replaces
`1/8` with `(b − a)²/8`), `η > 0`, and `T` known in advance to set `η*` (the
doubling trick removes this at a constant-factor cost). No assumption on how
`ℓ_t` is chosen is used, beyond `ℓ_t` not depending on the draw `I ~ p_t`.
