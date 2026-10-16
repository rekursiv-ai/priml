# Nesterov Acceleration

Nesterov's method reaches `O(1/k²)` because a weighted potential,
`t_k² ⋅ (f(x_k) − f*) + (L/2) ⋅ ‖u_k‖²`, never increases, and the weights
grow as `t_k ≥ (k+1)/2`.

Let,

  ```
  f : ℝⁿ → ℝ convex and L-smooth,   x* a minimizer,   f* ≝ f(x*)
  y₁      ≝ x₀,   t₁ ≝ 1
  x_k     ≝ y_k − ∇f(y_k)/L
  t_{k+1} ≝ (1 + √(1 + 4t_k²)) / 2
  y_{k+1} ≝ x_k + ((t_k − 1)/t_{k+1}) ⋅ (x_k − x_{k−1})
  v_k     ≝ f(x_k) − f*
  u_k     ≝ t_k x_k − (t_k − 1) x_{k−1} − x*
  ```

First, one gradient step `x⁺ = y − ∇f(y)/L` satisfies, for any `z`,

  ```
  f(x⁺) ≤ f(y) − ‖∇f(y)‖²/(2L)
        ≤ f(z) + ⟨∇f(y), y − z⟩ − ‖∇f(y)‖²/(2L)
        = f(z) + L ⋅ ⟨y − x⁺, y − z⟩ − (L/2) ⋅ ‖y − x⁺‖²
  ```

where the first step is the descent lemma (descent-lemma.md), the second is
convexity, `f(y) ≤ f(z) + ⟨∇f(y), y − z⟩`, and the third substitutes
`∇f(y) = L ⋅ (y − x⁺)`.

Second, apply the first step at `y = y_{k+1}`, `x⁺ = x_{k+1}`, once with
`z = x_k` and once with `z = x*`. Write `y ≝ y_{k+1}` and `x⁺ ≝ x_{k+1}`,

  ```
  v_{k+1} − v_k ≤ L ⋅ ⟨y − x⁺, y − x_k⟩ − (L/2) ⋅ ‖y − x⁺‖²
  v_{k+1}       ≤ L ⋅ ⟨y − x⁺, y − x*⟩  − (L/2) ⋅ ‖y − x⁺‖²
  ```

Multiply the first by `t_{k+1} − 1 ≥ 0` and add the second,

  ```
  t_{k+1} v_{k+1} − (t_{k+1} − 1) v_k
      ≤ L ⋅ ⟨y − x⁺, t_{k+1} y − (t_{k+1} − 1) x_k − x*⟩ − (L/2) ⋅ t_{k+1} ⋅ ‖y − x⁺‖²
  ```

Third, multiply by `t_{k+1}`. The recursion makes `t_{k+1}` the positive root
of `t² − t − t_k² = 0`, so `t_{k+1}² − t_{k+1} = t_k²`. With
`a ≝ t_{k+1} ⋅ (y − x⁺)` and `b ≝ t_{k+1} y − (t_{k+1} − 1) x_k − x*`,

  ```
  t_{k+1}² v_{k+1} − t_k² v_k ≤ L ⋅ ⟨a, b⟩ − (L/2) ⋅ ‖a‖²
                              = (L/2) ⋅ ( ‖b‖² − ‖b − a‖² )
  ```

where the last step expands `‖b − a‖² = ‖b‖² − 2⟨a, b⟩ + ‖a‖²`.

Fourth, identify `b` and `b − a`. From the momentum step,
`t_{k+1} y_{k+1} = t_{k+1} x_k + (t_k − 1) ⋅ (x_k − x_{k−1})`, so

  ```
  b     = t_{k+1} x_k + (t_k − 1)(x_k − x_{k−1}) − (t_{k+1} − 1) x_k − x*
        = t_k x_k − (t_k − 1) x_{k−1} − x*
        = u_k
  b − a = t_{k+1} x_{k+1} − (t_{k+1} − 1) x_k − x*
        = u_{k+1}
  ```

So the potential `Φ_k ≝ t_k² v_k + (L/2) ⋅ ‖u_k‖²` is nonincreasing,

  ```
  Φ_{k+1} − Φ_k = t_{k+1}² v_{k+1} − t_k² v_k + (L/2) ⋅ ( ‖u_{k+1}‖² − ‖u_k‖² )
                ≤ 0
  ```

Fifth, bound `Φ₁`. Since `t₁ = 1`, `u₁ = x₁ − x*`. Apply the first step at
`y = x₀`, `x⁺ = x₁`, `z = x*`, and complete the square as in the third step
with `a = x₀ − x₁`, `b = x₀ − x*`, `b − a = x₁ − x*`,

  ```
  v₁ ≤ L ⋅ ⟨x₀ − x₁, x₀ − x*⟩ − (L/2) ⋅ ‖x₀ − x₁‖²
     = (L/2) ⋅ ( ‖x₀ − x*‖² − ‖x₁ − x*‖² )
  ```

so `Φ₁ = v₁ + (L/2) ⋅ ‖x₁ − x*‖² ≤ (L/2) ⋅ ‖x₀ − x*‖²`.

Sixth, the weights grow linearly. By induction from `t₁ = 1`,

  ```
  t_{k+1} = (1 + √(1 + 4t_k²)) / 2
          ≥ (1 + 2t_k) / 2
          = t_k + 1/2
          ≥ (k + 2) / 2
  ```

so `t_k ≥ (k+1)/2` for all `k ≥ 1`, and in particular `t_k ≥ 1`, which the
second step used.

Hence, chaining `t_k² v_k ≤ Φ_k ≤ Φ₁`,

  ```
  f(x_k) − f* ≤ Φ₁ / t_k²
              ≤ (L/2) ⋅ ‖x₀ − x*‖² ⋅ 4/(k+1)²
              = 2L ⋅ ‖x₀ − x*‖² / (k+1)²
  ```

The order is optimal: for `k ≤ (n−1)/2`, Nesterov's lower bound gives a
convex L-smooth `f` on which every method whose iterates lie in the span of
past gradients has `f(x_k) − f* ≥ 3L ⋅ ‖x₀ − x*‖² / (32(k+1)²)`. The
momentum weight `(t_k − 1)/t_{k+1}` behaves like `(k − 1)/(k + 2)`.

Regularity: the proof needs `f` convex (first step) and L-smooth with step
exactly `1/L` (descent lemma), a minimizer `x*`, and the specific recursion
for `t_k`, used only through `t₁ = 1`, `t_{k+1}² − t_{k+1} = t_k²`, and
`t_k ≥ 1`. Since `t₁ = 1`, `y₂ = x₁` and `x₀`'s role in `u₁` vanishes.
