# Policy Improvement

Policy improvement follows from two facts:

1. acting greedily for one step, then following `π`, does no worse than `π`,
   and,
2. `T^{π'}` is monotone, so that one-step gain compounds over every later step.

Let,

  ```
  Q^π(s,a) ≝ r(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')
  V^π(s)   ≝ Σ_a π(a|s) Q^π(s,a)
  π'(·|s)  is supported on argmax_a Q^π(s,a)
  ```

with `T^π`, `T`, and `V*` as in bellman-contraction.md, so `V^π = T^π V^π`,
`V* = T V*`, and `(T^{π'} V^π)(s) = Σ_a π'(a|s) Q^π(s,a)`.

First, the greedy step does no worse at every state,

  ```
  V^π(s) = Σ_a π(a|s) Q^π(s,a)
         ≤ max_a Q^π(s,a)
         = Σ_a π'(a|s) Q^π(s,a)
         = (T^{π'} V^π)(s)
         = (T V^π)(s)
  ```

where the inequality holds because an average is at most the max.

Second, unroll. `T^{π'}` is monotone: if `V ≤ W` pointwise then
`T^{π'} W − T^{π'} V = γ Σ_a π' Σ_{s'} P (W − V) ≥ 0`. Write
`𝔼^{π'}_s` for a trajectory from `s₀ = s` with `a_t ∼ π'(·|s_t)` and
`s_{t+1} ∼ P(·|s_t,a_t)`. Applying the first step at `s₁`, then `s₂`, ...,

  ```
  V^π(s) ≤ 𝔼^{π'}_s[ r(s₀,a₀) + γ V^π(s₁) ]
         ≤ 𝔼^{π'}_s[ r(s₀,a₀) + γ r(s₁,a₁) + γ² V^π(s₂) ]
         ≤ 𝔼^{π'}_s[ Σ_{t<n} γ^t r(s_t,a_t) + γ^n V^π(s_n) ]
         → V^{π'}(s)
  ```

where each `≤` replaces `V^π(s_k)` by the larger `(T^{π'} V^π)(s_k)` inside a
nonnegatively weighted average, and the third line is induction on `n`. The
limit holds because `|γ^n V^π(s_n)| ≤ γ^n R_max / (1 − γ) → 0` and the partial
sums converge to `V^{π'}(s)` by dominated convergence.

Third, `V*` is the optimal value, and it is attained. For any stationary `σ`,
`T^σ V ≤ T V` pointwise (average ≤ max), and `T` is monotone (a max of monotone
maps). Hence,

  ```
  V^σ = T^σ V^σ
      ≤ T V^σ
      ≤ T² V^σ
      ≤ T^k V^σ
      → V*
  ```

where each step applies the monotone `T` to the previous inequality, and the
limit is value iteration from bellman-contraction.md. Let `π*` be greedy
w.r.t. `r + γ P V*`. Then `T^{π*} V* = T V* = V*`, so `V^{π*} = V*` by
uniqueness of the fixed point of `T^{π*}`. So `V* = max_σ V^σ` pointwise.

Fourth, equality characterizes optimality. If `V^{π'} = V^π`, then

  ```
  V^π = V^{π'}
      = T^{π'} V^{π'}
      = T^{π'} V^π
      = T V^π
  ```

so `V^π` is the fixed point of `T`, i.e., `V^π = V*`, and `π` is optimal by the
third step. Conversely, if `π` is optimal, `V^π = V* ≥ V^{π'} ≥ V^π`.

Hence,

  ```
  V^{π'} ≥ V^π                                     (pointwise)
  V^{π'} = V^π  ⇔  V^π = T V^π  ⇔  π optimal
  ```

So if `π` is not optimal, `V^{π'} ≥ V^π` with strict inequality at some state.
The Bellman optimality equation `V^π = T V^π` is exactly the condition that no
greedy step can improve.

Regularity: the proof used finite `S` and `A`, `|r| ≤ R_max`, `γ < 1`, and
stationary (possibly stochastic) Markov policies. That no history-dependent
policy beats `V*` is the standard extension (Puterman 1994, Ch. 6), cited, not
proved here.
