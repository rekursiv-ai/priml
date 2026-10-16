# Performance Difference

The performance difference lemma follows from two facts:

1. `Σ_t γ^t (γ V(s_{t+1}) − V(s_t))` telescopes to `−V(s₀)`, and,
2. `r(s,a) + γ 𝔼[V^π(s')|s,a]` is `Q^π(s,a)` by definition.

Let,

  ```
  A^π(s,a)    ≝ Q^π(s,a) − V^π(s)
  J(π)        ≝ 𝔼_{s₀∼ρ}[ V^π(s₀) ]
  d^{π'}_ρ(s) ≝ (1 − γ) Σ_t γ^t Pr^{π'}(s_t = s)
  τ           ≝ (s₀, a₀, s₁, a₁, …),  s₀ ∼ ρ,  a_t ∼ π'(·|s_t),  s_{t+1} ∼ P(·|s_t,a_t)
  ```

with `|r| ≤ R_max`, so `|V^π| ≤ R_max / (1 − γ)`.

First, along any trajectory, for bounded `V`,

  ```
  Σ_{t<T} γ^t (γ V(s_{t+1}) − V(s_t)) = Σ_{t=1}^{T} γ^t V(s_t) − Σ_{t=0}^{T−1} γ^t V(s_t)
                                      = γ^T V(s_T) − V(s₀)
                                      → −V(s₀)
  ```

Second, roll out `π'` and add the telescoped sum,

  ```
  J(π') − J(π) = 𝔼_τ[ Σ_t γ^t r(s_t,a_t) ] − 𝔼_τ[ V^π(s₀) ]
               = 𝔼_τ[ Σ_t γ^t ( r(s_t,a_t) + γ V^π(s_{t+1}) − V^π(s_t) ) ]
               = Σ_t γ^t 𝔼_τ[ r(s_t,a_t) + γ 𝔼[ V^π(s_{t+1}) | s_t,a_t ] − V^π(s_t) ]
               = Σ_t γ^t 𝔼_τ[ Q^π(s_t,a_t) − V^π(s_t) ]
               = Σ_t γ^t 𝔼_τ[ A^π(s_t,a_t) ]
  ```

where the first step uses `s₀ ∼ ρ` under `τ`, the second the first step, the
third swaps `Σ_t` and `𝔼` (dominated by `Σ_t γ^t (R_max + 2 R_max/(1−γ))`) and
applies the tower rule, and the fourth uses `s_{t+1} ∼ P(·|s_t,a_t)`.

Third, group by state,

  ```
  Σ_t γ^t 𝔼_τ[ A^π(s_t,a_t) ] = Σ_t γ^t Σ_s Pr^{π'}(s_t = s) Σ_a π'(a|s) A^π(s,a)
                              = Σ_s ( Σ_t γ^t Pr^{π'}(s_t = s) ) Σ_a π'(a|s) A^π(s,a)
                              = (1/(1−γ)) Σ_s d^{π'}_ρ(s) Σ_a π'(a|s) A^π(s,a)
  ```

where the second step swaps the absolutely convergent sums.

Hence,

  ```
  J(π') − J(π) = (1/(1−γ)) 𝔼_{s∼d^{π'}_ρ, a∼π'(·|s)}[ A^π(s,a) ]
  ```

This is an identity, with no slack. Two checks: `Σ_a π(a|s) A^π(s,a) = 0`, so
`π' = π` gives zero; and if `Σ_a π'(a|s) A^π(s,a) ≥ 0` at every `s`, then
`J(π') ≥ J(π)` for every `ρ`, recovering policy-improvement.md. The states are
drawn from the new policy's `d^{π'}_ρ`. Replacing it by `d^π_ρ` gives the
surrogate used by conservative policy iteration and TRPO, which is exact only
to first order in `π' − π`.

Regularity: the proof used `|r| ≤ R_max` and `γ < 1` (so `V^π` is bounded,
the telescoping remainder `γ^T V(s_T)` vanishes, and Fubini applies), and
countable `S`, `A` (so `Pr(s_t = s)` sums; replace sums with integrals in
general spaces).
