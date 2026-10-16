# Thompson Sampling

Thompson sampling's Bayesian regret bound (Russo & Van Roy 2014) follows from
two facts:

1. given the history, the played arm `A_t` and the optimal arm `A*` have the
   same law, so any history-measurable upper bound `U_t` scores them equally,
   and,
2. with `U_t` the UCB index, regret splits into confidence widths and failure
   probabilities, bounded as for UCB (ucb-regret.md).

Let,

  ```
  μ ∼ prior on [0,1]^K;   A* ≝ argmaxₐ μₐ      (fixed tie rule)
  X_{a,1}, X_{a,2}, …     given μ, iid rewards of arm a in [0,1] with mean μₐ
  H_{t−1}                 history A_1, r_1, …, A_{t−1}, r_{t−1}
  A_t                     drawn so that P(A_t = a | H_{t−1}) = P(A* = a | H_{t−1})
  Nₜ(a)                   pulls of a before round t;   μ̂ₜ(a) ≝ their mean
  BR_T                    ≝ 𝔼[ Σ_{t≤T} ( μ_{A*} − μ_{A_t} ) ]
  ```

Thompson sampling draws `μ̃ ∼ P(μ | H_{t−1})` and plays `argmaxₐ μ̃ₐ` with the
same tie rule, which is exactly the matching in the definition of `A_t`. The
expectation in `BR_T` is over the prior, the rewards, and the sampling.

First, posterior matching. Let `U_t : [K] → ℝ` be any `H_{t−1}`-measurable
function. Then

  ```
  𝔼[ U_t(A_t) | H_{t−1} ] = Σₐ P(A_t = a | H_{t−1}) U_t(a)
                          = Σₐ P(A* = a | H_{t−1}) U_t(a)
                          = 𝔼[ U_t(A*) | H_{t−1} ]
  ```

where the first and third steps hold because `U_t` is fixed given `H_{t−1}`.

Second, the reduction. Add and subtract `U_t`, and use the first step with
the tower rule,

  ```
  𝔼[ μ_{A*} − μ_{A_t} ] = 𝔼[ μ_{A*} − U_t(A*) ] + 𝔼[ U_t(A*) − U_t(A_t) ] + 𝔼[ U_t(A_t) − μ_{A_t} ]
                        = 𝔼[ μ_{A*} − U_t(A*) ] + 𝔼[ U_t(A_t) − μ_{A_t} ]
  ```

So TS pays what a UCB algorithm with index `U_t` would pay. TS never computes
`U_t`; it exists only in the analysis.

Third, choose the UCB index and its good event. Let

  ```
  U_t(a) ≝ min{ 1, μ̂ₜ(a) + √(2 log T / Nₜ(a)) }    (U_t(a) = 1 if Nₜ(a) = 0)
  E      ≝ { |μ̂_{a,s} − μₐ| < √(2 log T / s)  for all a ≤ K, s ≤ T }
  ```

where `μ̂_{a,s}` is the mean of `X_{a,1..s}`. Conditional on `μ`, Hoeffding
(hoeffding.md) bounds each of the `KT` events by `2 exp(−4 log T)`, so

  ```
  P(E^c) ≤ K T ⋅ 2 T⁻⁴
         = 2 K T⁻³
  ```

Fourth, bound both terms. On `E`, `U_t(a) ≥ μₐ` for all `a` (since `μₐ ≤ 1`),
so the first term is `≤ 0` on `E` and `≤ 1` off it. On `E` with
`Nₜ(A_t) ≥ 1`, `U_t(A_t) − μ_{A_t} < 2 √(2 log T / Nₜ(A_t))`; otherwise the
gap is at most `1`. Summing over `t`,

  ```
  BR_T ≤ T P(E^c) + T P(E^c) + 𝔼[ Σ_t 1{Nₜ(A_t) = 0} ] + 2 √(2 log T) 𝔼[ Σ_t 1{Nₜ(A_t) ≥ 1} / √Nₜ(A_t) ]
       ≤ 4 K T⁻² + K + 2 √(2 log T) 𝔼[ Σₐ Σ_{n=1}^{N_{T+1}(a)} n^{−1/2} ]
  ```

where the second step uses that each arm's first pull happens once, and that
the `n`-th repeat pull of `a` has `Nₜ(a) = n`.

Fifth, sum the widths. Use `Σ_{n≤m} n^{−1/2} ≤ 2√m` (compare with
`∫₀^m x^{−1/2} dx`), then Cauchy–Schwarz with `Σₐ N_{T+1}(a) = T`,

  ```
  Σₐ Σ_{n=1}^{N_{T+1}(a)} n^{−1/2} ≤ 2 Σₐ √N_{T+1}(a)
                                   ≤ 2 √( K Σₐ N_{T+1}(a) )
                                   = 2 √(K T)
  ```

Hence,

  ```
  BR_T ≤ 4 √(2 K T log T) + K + 4 K T⁻²
       = O( √(K T log T) )
  ```

The `√log T` comes from the union bound in the third step, not from TS; the
Bayesian minimax lower bound is `Ω(√(KT))`.

Fine print: the bound is Bayesian: it averages over the prior that TS itself
uses, so it says nothing about a fixed `μ` or a misspecified prior. The
argument works for any `U_t`, so any UCB analysis transfers (linear, GP, and
eluder-dimension bounds in Russo & Van Roy 2014). Bubeck & Liu (2013) sharpen
the Bayesian bound to `14 √(KT)`. Frequentist bounds for a fixed `μ` are
cited, not proved here: for Bernoulli rewards with Beta(1,1) priors, Agrawal &
Goyal (2012, 2017) show `O(Σ_{Δₐ>0} log T / Δₐ)` and `O(√(KT log T))`, and
Kaufmann, Korda & Munos (2012) show the Lai–Robbins constant is attained.

Regularity: rewards in `[0,1]` and iid per arm given `μ`; a prior on
`[0,1]^K` that TS samples from exactly; `A*` with a fixed tie rule shared by
TS; and `T ≥ 2` so `log T > 0`.
