# Policy Gradient

The policy gradient theorem follows from two facts:

1. differentiating the Bellman equation gives a Bellman equation for `∇V^π`,
   and,
2. the score has zero mean, `Σ_a ∇π(a|s) = ∇ 1 = 0`.

Let,

  ```
  π         ≝ π_θ,  θ ∈ ℝ^d,  ∇ ≝ ∇_θ
  P^π(s,s') ≝ Σ_a π(a|s) P(s'|s,a)
  g(s)      ≝ Σ_a ∇π(a|s) Q^π(s,a) = 𝔼_{a∼π(·|s)}[ Q^π(s,a) ∇ log π(a|s) ]
  J(θ)      ≝ Σ_s ρ(s) V^π(s)
  G_t       ≝ Σ_{k≥t} γ^{k−t} r(s_k,a_k)
  ```

with `S`, `A` finite and `π(a|s) > 0` differentiable in `θ`, so
`∇π = π ∇ log π`.

First, `V^π` is differentiable. `V^π = (I − γ P^π)⁻¹ r^π` with
`r^π(s) ≝ Σ_a π(a|s) r(s,a)`; the entries of `P^π`, `r^π` are differentiable,
and matrix inversion is smooth on invertible matrices (policy-iteration.md
shows invertibility). So `∇V^π` and `∇Q^π` exist.

Second, differentiate `V^π(s) = Σ_a π(a|s) Q^π(s,a)`,

  ```
  ∇V^π(s) = Σ_a ∇π(a|s) Q^π(s,a) + Σ_a π(a|s) ∇Q^π(s,a)
          = g(s) + Σ_a π(a|s) γ Σ_{s'} P(s'|s,a) ∇V^π(s')
          = g(s) + γ Σ_{s'} P^π(s,s') ∇V^π(s')
  ```

where the second step uses `Q^π = r + γ P V^π` with `r`, `P` independent of
`θ`.

Third, unroll. Stack `∇V^π(s)ᵀ` and `g(s)ᵀ` as rows of `|S| × d` matrices,

  ```
  ∇V^π = g + γ P^π ∇V^π
       = g + γ P^π g + γ² (P^π)² ∇V^π
       = Σ_{t<n} γ^t (P^π)^t g + γ^n (P^π)^n ∇V^π
       → Σ_t γ^t (P^π)^t g
  ```

where the remainder is bounded by `γ^n max_s ‖∇V^π(s)‖ → 0`.

Fourth, average over `ρ`, which does not depend on `θ`,

  ```
  ∇J(θ) = Σ_s ρ(s) Σ_t γ^t Σ_x Pr(s_t = x | s₀ = s) g(x)
        = Σ_x ( Σ_t γ^t Pr(s_t = x) ) g(x)
        = (1/(1−γ)) Σ_x d^π_ρ(x) g(x)
        = (1/(1−γ)) 𝔼_{s∼d^π_ρ, a∼π}[ Q^π(s,a) ∇ log π(a|s) ]
  ```

Fifth, the trajectory (REINFORCE) form. By the Markov property, the future
after `(s_t,a_t)` is a fresh rollout, so `𝔼[G_t | s_t,a_t] = Q^π(s_t,a_t)`.
Expanding `d^π_ρ` by its definition,

  ```
  ∇J(θ) = Σ_t γ^t 𝔼_τ[ Q^π(s_t,a_t) ∇ log π(a_t|s_t) ]
        = Σ_t γ^t 𝔼_τ[ 𝔼[ G_t | s_t,a_t ] ∇ log π(a_t|s_t) ]
        = 𝔼_τ[ Σ_t γ^t G_t ∇ log π(a_t|s_t) ]
  ```

where the last step is the tower rule plus Fubini; the summand is bounded by
`γ^t R_max/(1−γ) ⋅ max_{s,a} ‖∇ log π(a|s)‖`, finite for finite `S`, `A`.

Sixth, a baseline `b(s)` changes nothing in expectation,

  ```
  𝔼_{a∼π(·|s)}[ ∇ log π(a|s) ] = Σ_a π(a|s) ∇ log π(a|s)
                               = Σ_a ∇π(a|s)
                               = ∇ Σ_a π(a|s)
                               = ∇ 1
                               = 0
  ```

So `𝔼_{a∼π}[ (Q^π(s,a) − b(s)) ∇ log π(a|s) ] = g(s) − b(s) ⋅ 0 = g(s)`. In
the trajectory form, conditioning on `s_t` shows
`𝔼[ b(s_t) ∇ log π(a_t|s_t) ] = 0`, so `G_t − b(s_t)` is also unbiased.
`b = V^π` turns `Q^π − b` into `A^π`.

Seventh, the variance-optimal scalar baseline. Fix `s`, draw `a ∼ π(·|s)`, let
`ĝ_b ≝ (Q^π(s,a) − b) ∇ log π(a|s)` and `w(a) ≝ ‖∇ log π(a|s)‖²` with
`𝔼[w] > 0`. Since `𝔼 ĝ_b = g(s)` for all `b`,

  ```
  tr 𝕍(ĝ_b) = 𝔼‖ĝ_b‖² − ‖g(s)‖²
            = 𝔼[ (Q − b)² w ] − ‖g(s)‖²
            = 𝔼[w] b² − 2 𝔼[Q w] b + 𝔼[Q² w] − ‖g(s)‖²
            = 𝔼[w] (b − b*)² + 𝔼[Q² w] − 𝔼[Q w]² / 𝔼[w] − ‖g(s)‖²
  ```

with `b* ≝ 𝔼[Q w] / 𝔼[w]`, where the last step completes the square. Only the
first term depends on `b`, and it is zero iff `b = b*`.

Hence,

  ```
  ∇J(θ) = (1/(1−γ)) 𝔼_{s∼d^π_ρ, a∼π}[ (Q^π(s,a) − b(s)) ∇ log π(a|s) ]
        = 𝔼_τ[ Σ_t γ^t (G_t − b(s_t)) ∇ log π(a_t|s_t) ]
  b*(s) = 𝔼_{a∼π}[ Q^π(s,a) ‖∇ log π(a|s)‖² ] / 𝔼_{a∼π}[ ‖∇ log π(a|s)‖² ]
  ```

`b*(s) = V^π(s)` iff `Q^π(s,·)` is uncorrelated with `w` under `π(·|s)`
(e.g., `w` constant in `a`). With `G_t` in place of `Q^π`, `b*` is unchanged:
`𝔼[G_t w | s_t] = 𝔼[Q^π w | s_t]` by the tower rule, and `𝔼[G_t² w]` does not
depend on `b`.

Regularity: the proof used finite `S` and `A`, `π_θ(a|s) > 0` and
differentiable, `r`, `P`, `ρ` independent of `θ`, `|r| ≤ R_max`, `γ < 1`, and
`𝔼_π[‖∇ log π‖²] > 0` for `b*`. Dropping `γ^t` in the trajectory form, as is
common in practice, gives a biased estimator.
