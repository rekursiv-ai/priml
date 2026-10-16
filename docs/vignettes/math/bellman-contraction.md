# Bellman Contraction

The Bellman operators are `γ`-contractions, from two facts:

1. a max (or an average) moves by at most the largest change of its argument,
   and,
2. `P(·|s,a)` is a probability distribution, so averaging over it cannot
   enlarge `‖·‖_∞`.

Let,

  ```
  (T^π V)(s) ≝ Σ_a π(a|s) ⋅ [ r(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ]
  (T V)(s)   ≝ max_a [ r(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ]
  ‖V‖_∞      ≝ max_s |V(s)|
  ```

with `S` and `A` finite and `γ ∈ [0,1)`.

First, a max moves by at most the largest pointwise change. For `f, g` on a
finite set, let `a*` maximize `f`,

  ```
  max_a f(a) − max_a g(a) = f(a*) − max_a g(a)
                          ≤ f(a*) − g(a*)
                          ≤ max_a |f(a) − g(a)|
  ```

Swapping `f` and `g` bounds the negative side, so
`|max f − max g| ≤ max |f − g|`.

Second, `T` is a `γ`-contraction. Fix `s`,

  ```
  |(T V)(s) − (T W)(s)| ≤ max_a | γ Σ_{s'} P(s'|s,a) (V(s') − W(s')) |
                        ≤ γ max_a Σ_{s'} P(s'|s,a) |V(s') − W(s')|
                        ≤ γ max_a Σ_{s'} P(s'|s,a) ‖V − W‖_∞
                        = γ ‖V − W‖_∞
  ```

where the first step is the first fact (the `r(s,a)` terms cancel), the second
is the triangle inequality, and the last uses `Σ_{s'} P(s'|s,a) = 1`. Taking
`max_s` gives `‖T V − T W‖_∞ ≤ γ ‖V − W‖_∞`.

Third, `T^π` is a `γ`-contraction by the same steps with an average in place
of the max,

  ```
  |(T^π V)(s) − (T^π W)(s)| = | γ Σ_a π(a|s) Σ_{s'} P(s'|s,a) (V(s') − W(s')) |
                            ≤ γ Σ_a π(a|s) Σ_{s'} P(s'|s,a) |V(s') − W(s')|
                            ≤ γ Σ_a π(a|s) Σ_{s'} P(s'|s,a) ‖V − W‖_∞
                            = γ ‖V − W‖_∞
  ```

Fourth, apply Banach (banach.md). `(ℝ^{|S|}, ‖·‖_∞)` is complete, so `T` has a
unique fixed point `V*`, and `T^π` has a unique fixed point. That fixed point
is `V^π`, since conditioning on the first step gives `V^π = T^π V^π`. Value
iteration `V_{k+1} = T V_k` then satisfies

  ```
  ‖V_k − V*‖_∞ = ‖T V_{k−1} − T V*‖_∞
               ≤ γ ‖V_{k−1} − V*‖_∞
               ≤ γ^k ‖V_0 − V*‖_∞
  ```

where the last step is induction on `k`. For a computable stopping rule, use
the triangle inequality and one contraction step,

  ```
  ‖V_k − V*‖_∞ ≤ ‖V_k − V_{k+1}‖_∞ + ‖T V_k − T V*‖_∞
               ≤ ‖V_k − V_{k+1}‖_∞ + γ ‖V_k − V*‖_∞
  ```

and rearrange.

Hence,

  ```
  V*           = T V*,   V^π = T^π V^π            (each the unique fixed point)
  ‖V_k − V*‖_∞ ≤ γ^k ‖V_0 − V*‖_∞
  ‖V_k − V*‖_∞ ≤ ‖V_{k+1} − V_k‖_∞ / (1 − γ)
  ```

The rate `γ` is tight. With one state, one action, and a self-loop,
`T V = r + γ V`, so `V_k − V* = γ^k (V_0 − V*)` exactly. That `V*` is the
optimal value `max_π V^π` is shown in policy-improvement.md.

Regularity: the proof used finite `S` (so `ℝ^{|S|}` is complete and `max_s`
exists; for countable `S` use bounded functions with `sup`), finite `A` (so
`max_a` is attained), `P(·|s,a)` summing to one, finite `r`, and `γ < 1`.
