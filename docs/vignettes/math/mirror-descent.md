# Mirror Descent

Mirror descent's regret bound follows from three facts:

1. the three-point identity of the Bregman divergence,
2. the first-order optimality condition of the update, and,
3. σ-strong convexity of `ψ`, which makes `D_ψ` dominate `(σ/2) ⋅ ‖·‖²`.

Let,

  ```
  X ⊆ ℝⁿ convex,   ‖·‖ a norm,   ‖g‖* ≝ sup_{‖v‖≤1} ⟨g, v⟩
  ψ : X → ℝ differentiable, σ-strongly convex w.r.t. ‖·‖
  D_ψ(y, x) ≝ ψ(y) − ψ(x) − ⟨∇ψ(x), y − x⟩ ≥ (σ/2) ⋅ ‖y − x‖²
  x_{t+1}   ≝ argmin_{y∈X} { ⟨g_t, y⟩ + (1/η) ⋅ D_ψ(y, x_t) }
  ```

The inequality on `D_ψ` is the definition of σ-strong convexity. With
`ψ = ½‖·‖₂²`, `D_ψ(y, x) = ½‖y − x‖₂²` and the update is projected GD.

First, the three-point identity. Expanding the three divergences, every `ψ`
value cancels,

  ```
  D_ψ(u, x) − D_ψ(u, y) − D_ψ(y, x)
      = −⟨∇ψ(x), u − x⟩ + ⟨∇ψ(y), u − y⟩ + ⟨∇ψ(x), y − x⟩
      = −⟨∇ψ(x), u − y⟩ + ⟨∇ψ(y), u − y⟩
      = ⟨∇ψ(y) − ∇ψ(x), u − y⟩
  ```

Second, the update minimizes a convex differentiable function over `X`,
whose gradient in `y` is `g_t + (∇ψ(y) − ∇ψ(x_t))/η`. First-order optimality
at `x_{t+1}` says, for every `u ∈ X`,

  ```
  ⟨η g_t + ∇ψ(x_{t+1}) − ∇ψ(x_t), u − x_{t+1}⟩ ≥ 0
  ```

Rearranging, then applying the three-point identity with `y = x_{t+1}`,
`x = x_t`,

  ```
  η ⋅ ⟨g_t, x_{t+1} − u⟩ ≤ ⟨∇ψ(x_{t+1}) − ∇ψ(x_t), u − x_{t+1}⟩
                         = D_ψ(u, x_t) − D_ψ(u, x_{t+1}) − D_ψ(x_{t+1}, x_t)
  ```

Third, add the term `η ⋅ ⟨g_t, x_t − x_{t+1}⟩`. Write
`b ≝ ‖x_t − x_{t+1}‖`,

  ```
  η ⋅ ⟨g_t, x_t − u⟩ = η ⋅ ⟨g_t, x_t − x_{t+1}⟩ + η ⋅ ⟨g_t, x_{t+1} − u⟩
                     ≤ η ‖g_t‖* b + D_ψ(u, x_t) − D_ψ(u, x_{t+1}) − D_ψ(x_{t+1}, x_t)
                     ≤ η ‖g_t‖* b − (σ/2) b² + D_ψ(u, x_t) − D_ψ(u, x_{t+1})
                     ≤ η² ‖g_t‖*² / (2σ) + D_ψ(u, x_t) − D_ψ(u, x_{t+1})
  ```

where the second step uses the dual-norm (Hölder) inequality
`⟨g, v⟩ ≤ ‖g‖* ⋅ ‖v‖`, the third uses strong convexity, and the fourth uses
`a b − (σ/2) b² ≤ a²/(2σ)`, the maximum over `b` of the left side.

Fourth, sum over `t = 1, …, T` and divide by `η`. The divergences telescope,
and `D_ψ(u, x_{T+1}) ≥ 0` is dropped.

Hence, for every `u ∈ X`,

  ```
  Σ_{t=1}^T ⟨g_t, x_t − u⟩ ≤ D_ψ(u, x₁)/η + η Σ_{t=1}^T ‖g_t‖*² / (2σ)
  ```

If `g_t ∈ ∂f(x_t)` for convex `f`, then `f(x_t) − f(u) ≤ ⟨g_t, x_t − u⟩`, and
Jensen gives `f(x̄_T) − f(u) ≤ (1/T) Σ_t ⟨g_t, x_t − u⟩` for the average
`x̄_T`. With `D_ψ(u, x₁) ≤ B²`, `‖g_t‖* ≤ G`, and `η = (B/G) ⋅ √(2σ/T)`,
the two terms are equal,

  ```
  Σ_t ⟨g_t, x_t − u⟩ ≤ B G √T / √(2σ) + B G √T / √(2σ)
                     = B G ⋅ √(2T/σ)
  ```

Fifth, the negative-entropy case. Take `X = Δₙ`, the probability simplex,
and `ψ(x) = Σᵢ xᵢ log xᵢ`, so `∇ψ(x)ᵢ = log xᵢ + 1`. For `x, y ∈ Δₙ`,

  ```
  D_ψ(y, x) = Σᵢ yᵢ log yᵢ − Σᵢ xᵢ log xᵢ − Σᵢ (log xᵢ + 1)(yᵢ − xᵢ)
            = Σᵢ yᵢ log yᵢ − Σᵢ yᵢ log xᵢ − Σᵢ (yᵢ − xᵢ)
            = Σᵢ yᵢ log(yᵢ / xᵢ)
            = 𝕂(y ‖ x)
  ```

where the last-but-one step uses `Σᵢ yᵢ = Σᵢ xᵢ = 1`. Pinsker's inequality
(pinsker.md) gives `𝕂(y ‖ x) ≥ ½ ‖y − x‖₁²`, which is exactly 1-strong
convexity of `ψ` w.r.t. `‖·‖₁`. So `σ = 1` and `‖·‖* = ‖·‖∞`.

Sixth, the update is exponentiated gradient. Minimize
`⟨g, y⟩ + (1/η) ⋅ 𝕂(y ‖ x)` subject to `Σᵢ yᵢ = 1` with multiplier `λ`,

  ```
  0  = gᵢ + (1/η) ⋅ (log(yᵢ / xᵢ) + 1) + λ
  yᵢ = xᵢ ⋅ exp(−η gᵢ) ⋅ exp(−1 − ηλ)
  yᵢ = xᵢ ⋅ exp(−η gᵢ) / Σⱼ xⱼ ⋅ exp(−η gⱼ)
  ```

where the last step picks `λ` so that `Σᵢ yᵢ = 1`. The solution has
`yᵢ > 0`, so the constraint `yᵢ ≥ 0` is inactive, and the KKT point of a
convex problem is its minimizer. From uniform `x₁`,
`D_ψ(u, x₁) = log n − H(u) ≤ log n`.

Hence, with `‖g_t‖∞ ≤ G` and `η = √(2 log n / T) / G`,

  ```
  x_{t+1,i} ∝ x_{t,i} ⋅ exp(−η g_{t,i})
  Σ_t ⟨g_t, x_t − u⟩ ≤ log n / η + η T G² / 2
                     = G ⋅ √(2T log n)
  ```

The dependence on dimension is `√(log n)`, versus `√n` for Euclidean
projected GD with `‖g‖∞ ≤ G` on the simplex. Equality in the regret bound
requires every inequality of the third step to be tight and
`D_ψ(u, x_{T+1}) = 0`.

Regularity: the proof needs `X` convex, `ψ` differentiable and σ-strongly
convex on `X` with each `x_{t+1}` a point where `∇ψ` exists (for negative
entropy, iterates stay in the relative interior of `Δₙ` when `x₁` does),
`u ∈ X` with `D_ψ(u, x₁) < ∞`, and convex `f` for the optimization
corollary.
