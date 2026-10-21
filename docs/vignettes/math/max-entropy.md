# Maximum Entropy

The maximum-entropy distribution follows from one fact: the entropy gap
between the exponential family and any feasible `p` is a KL, which is
nonnegative (gibbs-inequality.md).

Let `μ` be a base measure on `𝒴` (counting measure for discrete `Y`,
Lebesgue for continuous), and,

  ```
  T : 𝒴 → ℝᵈ                         sufficient statistic
  τ ∈ ℝᵈ                              target moment
  H(p)   ≝ −∫ p(y) ⋅ log p(y) dμ(y)   entropy relative to μ
  p_η(y) ≝ exp( ηᵀT(y) − A(η) )
  A(η)   ≝ log ∫ exp( ηᵀT(y) ) dμ(y)
  ```

and suppose some `η` with `A(η) < ∞` satisfies `𝔼_{p_η}[T] = τ`. The problem
is `max_p H(p)` over densities `p` with respect to `μ`, subject to
`∫ p dμ = 1` and `𝔼_p[T] = τ`.

First, stationarity identifies the candidate. The Lagrangian with
multipliers `η` (moments) and `λ` (normalization) is
`H(p) + ηᵀ(𝔼_p[T] − τ) + λ(∫ p dμ − 1)`. Setting its variation in `p(y)` to
zero,

  ```
  0    = −log p(y) − 1 + ηᵀT(y) + λ
  p(y) = exp( ηᵀT(y) + λ − 1 )
       ∝ exp( ηᵀT(y) )
  ```

so normalization forces `p = p_η`. Stationarity alone does not prove a
maximum; the next two steps do.

Second, the entropy of `p_η` is affine in the moment,

  ```
  H(p_η) = −∫ p_η ⋅ ( ηᵀT − A(η) ) dμ
         = −ηᵀ 𝔼_{p_η}[T] + A(η)
         = −ηᵀ τ + A(η)
  ```

Third, for any feasible `p`, the cross-entropy against `p_η` depends on `p`
only through `𝔼_p[T] = τ`,

  ```
  H(p_η) − H(p) = −ηᵀτ + A(η) + ∫ p ⋅ log p dμ
                = −∫ p ⋅ ( ηᵀT − A(η) ) dμ + ∫ p ⋅ log p dμ
                = −∫ p ⋅ log p_η dμ + ∫ p ⋅ log p dμ
                = ∫ p ⋅ log( p / p_η ) dμ
                = 𝕂(p‖p_η)
                ≥ 0
  ```

where the second step uses `∫ p ⋅ ηᵀT dμ = ηᵀτ` and `∫ p dμ = 1`, and the
last step is Gibbs.

Hence,

  ```
  H(p) ≤ H(p_η) = A(η) − ηᵀτ    for every p with 𝔼_p[T] = τ
  ```

Equality holds iff `𝕂(p‖p_η) = 0`, i.e., `p = p_η` `μ`-almost everywhere. So
the maximizer is unique, and `−log p_η = A(η) − ηᵀT` is the loss the
constraint implies.

Examples. Each picks `μ` and `T`; the theorem does the rest.

- Softmax. `𝒴 = {1, …, K}`, counting measure, `T(y) = f(y) ∈ ℝ` a score per
  class. With `η` the inverse temperature,

    ```
    p_η(y) = exp( η f(y) ) / Σ_k exp( η f(k) )
    ```

  so `softmax(η f)` is the maximum-entropy distribution with expected score
  `τ`, and cross-entropy is its loss. With `T(y) = e_y` (one-hot),
  `p_η = softmax(η)` and `𝔼[T] = τ` pins `p = τ` directly.

- Gaussian. `𝒴 = ℝ`, Lebesgue, `T(y) = (y, y²)`, `τ = (m, m² + σ²)`. Then
  `p_η ∝ exp( η₁ y + η₂ y² )` normalizes only for `η₂ < 0`, and matching the
  moments gives `η₁ = m/σ²`, `η₂ = −1/(2σ²)`,

    ```
    p_η(y) = 𝒩(y | m, σ²)
    H(p)   ≤ ½ log(2πe σ²)    for every density with mean m, variance σ²
    ```

  so squared loss, `−log p_η = ½ z² + log σ + ½ log 2π` with
  `z ≝ (y − m)/σ`, is the maximum-entropy loss given two moments.

- Laplace and exponential. `𝒴 = ℝ`, Lebesgue, `T(y) = |y − m|`, `τ = b`
  gives `p_η ∝ exp(−|y − m|/b)` and absolute loss. On `𝒴 = [0, ∞)` with
  `T(y) = y` it gives the exponential `p_η ∝ exp(−y/τ)`.

More generally, `T(y) = ρ(y)` for any `ρ` with `∫ exp(−λρ) dμ < ∞` gives
`p_η ∝ exp(−λρ)`: maximum entropy reads a loss off a constraint. It does not
say which constraint to impose; least-favorable-density.md picks one by
robustness, and its Huber `ρ_c` is maximum-entropy only for the constraint
`𝔼[ρ_c] = τ`, which assumes the answer.

Regularity: `A(η) < ∞` at the maximizing `η`; `τ` in the range of
`η ↦ 𝔼_{p_η}[T]`, i.e., in the relative interior of the convex hull of
`T(𝒴)` (for finite `𝒴`) or of the support of `T` under `μ`. If `τ` is on the
boundary, the maximizer exists but lies at a limit `η → ∞` and has smaller
support. Entropy is relative to `μ`: changing `μ` changes the maximizer (a
uniform `μ` on `[a, b]` with no `T` gives the uniform density). Since
`p_η > 0` `μ`-a.e., `𝕂(p‖p_η)` is finite for every feasible `p` with finite
entropy.
