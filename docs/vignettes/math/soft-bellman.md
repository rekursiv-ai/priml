# Soft Bellman

The soft Bellman operator is a `γ`-contraction with a softmax optimal policy,
from two facts:

1. log-sum-exp is monotone and shifts with constants, so it is 1-Lipschitz in
   `‖·‖_∞`, and,
2. log-sum-exp is the maximum of `𝔼_p[q] + τ H(p)`, attained at
   `p ∝ exp(q/τ)` (gibbs-variational.md).

Let,

  ```
  τ            > 0,   S, A finite,   H(p) ≝ −Σ_a p(a) log p(a)
  L_τ(q)       ≝ τ log Σ_a exp(q(a)/τ)                        q ∈ ℝ^{|A|}
  V^π(s)       ≝ 𝔼^π[ Σ_t γ^t ( r(s_t,a_t) + τ H(π(·|s_t)) ) | s₀ = s ]
  Q^π(s,a)     ≝ r(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')
  (T Q)(s,a)   ≝ r(s,a) + γ Σ_{s'} P(s'|s,a) L_τ(Q(s',·))
  (T^π Q)(s,a) ≝ r(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') ( Q(s',a') − τ log π(a'|s') )
  J_τ(π)       ≝ 𝔼_{s∼ρ}[ V^π(s) ]
  ```

Splitting off the first step gives
`V^π(s) = Σ_a π(a|s) (Q^π(s,a) − τ log π(a|s))`, so `Q^π = T^π Q^π`.

First, `L_τ` is 1-Lipschitz. Let `c ≝ ‖q − q'‖_∞`, so `q ≤ q' + c` entrywise,

  ```
  L_τ(q) ≤ L_τ(q' + c)
         = τ log( e^{c/τ} Σ_a exp(q'(a)/τ) )
         = L_τ(q') + c
  ```

where the first step uses that `exp` and `log` are increasing. Swapping `q`
and `q'` gives `|L_τ(q) − L_τ(q')| ≤ ‖q − q'‖_∞`.

Second, `T` is a `γ`-contraction,

  ```
  |(T Q)(s,a) − (T Q')(s,a)| ≤ γ Σ_{s'} P(s'|s,a) |L_τ(Q(s',·)) − L_τ(Q'(s',·))|
                             ≤ γ Σ_{s'} P(s'|s,a) ‖Q − Q'‖_∞
                             = γ ‖Q − Q'‖_∞
  ```

By Banach (banach.md), `T` has a unique fixed point `Q*`, and value iteration
`Q_{k+1} = T Q_k` converges at rate `γ^k`. The same steps, with an average in
place of `L_τ`, make `T^π` a `γ`-contraction with fixed point `Q^π`.

Third, the variational form. Apply gibbs-variational.md with `Q` uniform on
`A` and `f = q/τ`. Since `𝕂(p‖uniform) = log |A| − H(p)`,

  ```
  log( Σ_a e^{q(a)/τ} / |A| ) = max_p { 𝔼_p[q]/τ − log |A| + H(p) }
  L_τ(q)                      = max_p { 𝔼_p[q] + τ H(p) }
  ```

with the unique maximizer `p(a) = exp(q(a)/τ) / Σ_b exp(q(b)/τ)`. So
`T^π Q ≤ T Q` pointwise, with equality iff `π(·|s') ∝ exp(Q(s',·)/τ)` at
every `s'`.

Fourth, the softmax policy is optimal. Let `π*(a|s) ∝ exp(Q*(s,a)/τ)`. Then
`T^{π*} Q* = T Q* = Q*`, so `Q^{π*} = Q*` by uniqueness. `T` is monotone,
since `L_τ` is. For any `π`,

  ```
  Q^π = T^π Q^π
      ≤ T Q^π
      ≤ T^k Q^π
      → Q*
  ```

where each step applies the monotone `T` to the previous inequality. Then at
each `s`,

  ```
  V^π(s) = Σ_a π(a|s) ( Q^π(s,a) − τ log π(a|s) )
         ≤ Σ_a π(a|s) ( Q*(s,a) − τ log π(a|s) )
         ≤ L_τ(Q*(s,·))
         = V^{π*}(s)
  ```

where the second step uses `Q^π ≤ Q*`, the third is the maximum from the
third step, and the last is its attainment at `π*(·|s)`.

Fifth, the hard limit. Let `T_0` be the hard operator with `max_a` in place
of `L_τ`, and `Q*_0` its fixed point. Bounding the sum below by its largest
term and above by `|A|` times it gives
`max q ≤ L_τ(q) ≤ max q + τ log |A|`, so

  ```
  ‖Q* − Q*_0‖_∞ ≤ ‖T Q* − T_0 Q*‖_∞ + ‖T_0 Q* − T_0 Q*_0‖_∞
                ≤ γ τ log |A| + γ ‖Q* − Q*_0‖_∞
  ```

and rearranging gives the last line below.

Hence,

  ```
  ‖T Q − T Q'‖_∞ ≤ γ ‖Q − Q'‖_∞
  Q*             = T Q*,      π*(a|s) ≝ exp( (Q*(s,a) − V*(s)) / τ ),      V*(s) ≝ L_τ(Q*(s,·))
  J_τ(π)         ≤ J_τ(π*)   for every π
  ‖Q* − Q*_0‖_∞  ≤ γ τ log |A| / (1 − γ)  → 0   as τ → 0
  ```

The optimal policy is unique, because the maximizer in the third step is
unique: equality `V^π = V^{π*}` at every `s` forces `π(·|s) = π*(·|s)`. The
`τ log |A|` gap in `L_τ` is attained when all actions tie.

Fine print: as `τ → 0`, `π*(·|s)` concentrates on `argmax_a Q*_0(s,a)`. An
action trailing by gap `g > 0` in `Q*_0` gets probability at most
`exp(−(g − 2γτ log |A| / (1 − γ)) / τ) → 0`. Within the argmax the limit
need not be uniform, since `Q*` can split hard ties at order `τ`. The
constant-shift property in the first step is what makes `L_τ` a "soft max":
besides monotonicity, it is the only property the contraction uses. Soft
Q-learning samples `T`. SAC alternates sampled `T^π` evaluation with the
softmax improvement of the third step.

Regularity: finite `S` and `A`, `|r| ≤ R_max`, `γ < 1`, and `τ > 0`.
Entropy is at most `log |A|`, so `V^π` is bounded and the series defining it
converges.
