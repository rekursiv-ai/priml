# Q-Learning

Tabular Q-learning converges w.p. 1, from two facts:

1. its expected update is the Bellman optimality operator, a `γ`-contraction,
   and,
2. its noise has zero conditional mean and conditionally bounded variance.

An asynchronous stochastic-approximation theorem turns these into
convergence; that theorem is cited, not proved here.

Let,

  ```
  |S|, |A| < ∞,   i = (s,a) indexes the table
  (H Q)(s,a)   ≝ r(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q(s',a')
  Q*           = H Q*
  r_t ∼ R(·|s_t,a_t),   𝔼[r_t | s_t,a_t] = r(s_t,a_t),   𝕍[r_t | s_t,a_t] ≤ σ_r²
  s_{t+1} ∼ P(·|s_t,a_t)
  Q_{t+1}(s,a) ≝ (1 − α_t(s,a)) Q_t(s,a) + α_t(s,a) ( r_t + γ max_{a'} Q_t(s_{t+1},a') )
  α_t(s,a)     = 0  unless (s,a) = (s_t,a_t)
  ```

and let `𝓕_t` be the history through `(s_t, a_t)`, with `α_t`
`𝓕_t`-measurable. Given `𝓕_t`, `(r_t, s_{t+1})` depends only on `(s_t,a_t)`.

First, `H` is a `γ`-contraction in `‖·‖_∞`,

  ```
  |(H Q)(s,a) − (H Q')(s,a)| = γ | Σ_{s'} P(s'|s,a) ( max_{a'} Q(s',a') − max_{a'} Q'(s',a') ) |
                             ≤ γ Σ_{s'} P(s'|s,a) max_{a'} |Q(s',a') − Q'(s',a')|
                             ≤ γ Σ_{s'} P(s'|s,a) ‖Q − Q'‖_∞
                             = γ ‖Q − Q'‖_∞
  ```

where the second step uses `|max f − max g| ≤ max |f − g|`
(bellman-contraction.md). By Banach (banach.md), `Q*` exists and is unique.

Second, rewrite as stochastic approximation. Let `Δ_t ≝ Q_t − Q*`. Subtracting
`Q*(s,a)` from both sides of the update,

  ```
  Δ_{t+1}(i) = (1 − α_t(i)) Δ_t(i) + α_t(i) F_t(i)
  F_t(s,a)   ≝ r_t + γ max_{a'} Q_t(s_{t+1},a') − Q*(s,a)       if (s,a) = (s_t,a_t)
  F_t(s,a)   ≝ (H Q_t)(s,a) − Q*(s,a)                           otherwise
  ```

where the second branch is arbitrary (its `α_t(i) = 0`) and chosen to satisfy
the conditions below.

Third, the conditional mean contracts. For the visited `(s,a)`,

  ```
  𝔼[ F_t(s,a) | 𝓕_t ] = r(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q_t(s',a') − Q*(s,a)
                      = (H Q_t)(s,a) − (H Q*)(s,a)
  ```

and the other entries equal this by definition. So, by the first step,
`‖𝔼[F_t | 𝓕_t]‖_∞ = ‖H Q_t − H Q*‖_∞ ≤ γ ‖Δ_t‖_∞`.

Fourth, the conditional variance grows at most quadratically. For the visited
`(s,a)`, let `M ≝ max_{a'} Q_t(s_{t+1},a')`,

  ```
  𝕍[ F_t(s,a) | 𝓕_t ] = 𝕍[ r_t + γ M | 𝓕_t ]
                      ≤ 2 𝕍[ r_t | 𝓕_t ] + 2 γ² 𝕍[ M | 𝓕_t ]
                      ≤ 2 σ_r² + 2 γ² ‖Q_t‖_∞²
                      ≤ 2 σ_r² + 2 γ² ( ‖Δ_t‖_∞ + ‖Q*‖_∞ )²
                      ≤ C ( 1 + ‖Δ_t‖_∞ )²
  ```

with `C ≝ 2 σ_r² + 2 γ² max(1, ‖Q*‖_∞)²`. The second step is
`𝕍(X + Y) ≤ 2 𝕍 X + 2 𝕍 Y`, the third uses `𝕍 M ≤ 𝔼 M² ≤ ‖Q_t‖_∞²`,
and the fourth the triangle inequality. Unvisited entries have zero variance.

Fifth, apply the cited theorem (Jaakkola, Jordan & Singh 1994, Thm. 1;
Tsitsiklis 1994, Thm. 3). Let `Δ_{t+1}(i) = (1 − α_t(i)) Δ_t(i) + α_t(i) F_t(i)`
on a finite index set, with `α_t`, `Δ_t` `𝓕_t`-measurable and `F_t`
`𝓕_{t+1}`-measurable. If, w.p. 1, for all `i`,

  ```
  0                  ≤ α_t(i) ≤ 1,   Σ_t α_t(i) = ∞,   Σ_t α_t(i)² < ∞
  ‖𝔼[ F_t | 𝓕_t ]‖_∞ ≤ κ ‖Δ_t‖_∞,   κ < 1
  𝕍[ F_t(i) | 𝓕_t ]  ≤ C ( 1 + ‖Δ_t‖_∞ )²
  ```

then `Δ_t → 0` w.p. 1. The third and fourth steps give the last two
conditions with `κ = γ`; the first is the step-size assumption. The cited
proof first shows `Q_t` bounded a.s., then sandwiches `Δ_t` between
deterministic contracting envelopes plus noise terms that vanish by a
Robbins–Monro argument (robbins-monro.md).

Hence,

  ```
  Q_t → Q*   w.p. 1
  ```

Since `α_t(s,a) = 0` off visits, `Σ_t α_t(s,a) = ∞` forces every `(s,a)` to be
visited infinitely often; this is the exploration requirement. Counts
`α_t(s,a) = 1/n_t(s,a)` satisfy the step conditions given infinite visits.
Once `‖Q_t − Q*‖_∞` is below half the smallest nonzero action gap of `Q*`, the
greedy policy w.r.t. `Q_t` is greedy w.r.t. `Q*`, hence optimal
(policy-improvement.md).

Regularity: the proof used finite `S` and `A`, `γ < 1`, bounded mean reward
`r(s,a)`, reward variance at most `σ_r²`, Markov sampling of
`(r_t, s_{t+1})` given `(s_t,a_t)`, and `𝓕_t`-measurable steps in `[0,1]`
with `Σ α = ∞`, `Σ α² < ∞` per entry.
