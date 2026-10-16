# Policy Iteration

Policy iteration terminates at an optimal policy, from two facts:

1. every non-terminal step strictly improves the value, and,
2. a finite MDP has finitely many deterministic policies.

Let,

  ```
  |S|, |A| < ∞,   π_k : S → A deterministic
  P^π(s,s')   ≝ P(s'|s,π(s))
  r^π(s)      ≝ r(s,π(s))
  V^{π_k}     ≝ (I − γ P^{π_k})⁻¹ r^{π_k}               (evaluation)
  π_{k+1}(s) ∈ argmax_a Q^{π_k}(s,a)                    (improvement)
  ```

Break ties in favor of `π_k(s)` whenever `π_k(s)` is a maximizer. Stop when
`π_{k+1} = π_k`.

First, evaluation is well defined. For `v ≠ 0`, since `P^π` has rows summing
to one, `‖γ P^π v‖_∞ ≤ γ ‖v‖_∞`, so

  ```
  ‖(I − γ P^π) v‖_∞ ≥ ‖v‖_∞ − ‖γ P^π v‖_∞
                    ≥ (1 − γ) ‖v‖_∞
                    > 0
  ```

by the reverse triangle inequality. So `I − γ P^π` is invertible, and its
solution is `V^π`, which satisfies `V^π = r^π + γ P^π V^π`.

Second, a non-terminal step strictly improves. If `π_{k+1} ≠ π_k`, then by the
tie rule there is an `s` where `π_k(s)` is not a maximizer, so

  ```
  (T V^{π_k})(s) = max_a Q^{π_k}(s,a)
                 > Q^{π_k}(s, π_k(s))
                 = V^{π_k}(s)
  ```

Thus `V^{π_k} ≠ T V^{π_k}`, so `π_k` is not optimal. By policy-improvement.md,
`V^{π_{k+1}} ≥ V^{π_k}` pointwise, with strict inequality at some `s`.

Third, no policy repeats. For `j > k` with no termination before `j`, the
second step at each iteration gives, at the state `s` where step `k` was strict,

  ```
  V^{π_j}(s) ≥ V^{π_{j−1}}(s)
             ≥ V^{π_{k+1}}(s)
             > V^{π_k}(s)
  ```

So `V^{π_j} ≠ V^{π_k}`, hence `π_j ≠ π_k`. The iterates are distinct
deterministic policies, of which there are `|A|^{|S|}`. So the stop rule fires
at some `K ≤ |A|^{|S|} − 1`.

Fourth, the terminal policy is optimal. If `π_{K+1} = π_K`, then `π_K(s)` is a
maximizer at every `s`, so

  ```
  (T V^{π_K})(s) = max_a Q^{π_K}(s,a)
                 = Q^{π_K}(s, π_K(s))
                 = V^{π_K}(s)
  ```

Thus `V^{π_K} = T V^{π_K}`, so `V^{π_K} = V*` and `π_K` is optimal by
policy-improvement.md.

Hence,

  ```
  π_K optimal,   K ≤ |A|^{|S|} − 1
  ```

The tie rule matters. Without it, policy iteration can cycle among distinct
optimal policies with equal values; stopping on `V^{π_{k+1}} = V^{π_k}` instead
also terminates, by the same argument. The count `|A|^{|S|}` is loose; for
fixed `γ` the iteration count is strongly polynomial (Ye 2011; Hansen,
Miltersen & Zwick 2013), cited, not proved here.

Regularity: the proof used finite `S` and `A`, finite `r`, `γ < 1`, and exact
policy evaluation.
