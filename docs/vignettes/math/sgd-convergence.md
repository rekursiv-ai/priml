# SGD Convergence

SGD on a convex function converges because the squared distance to the
optimum drops by the expected linearized gap, minus a noise term
`η² ⋅ G²`.

Let,

  ```
  f : ℝⁿ → ℝ convex,   x* a minimizer,   f* ≝ f(x*)
  x_{k+1}           ≝ x_k − ηₖ ⋅ g_k        ηₖ > 0 deterministic
  𝔼[ g_k | x_k ]    = ∇f(x_k)               (unbiased)
  𝔼[ ‖g_k‖² | x_k ] ≤ G²                    (bounded second moment)
  x̄_K               ≝ Σ_{k=0}^{K−1} ηₖ x_k / Σ_{k=0}^{K−1} ηₖ
  ```

First, expand the squared distance after one step,

  ```
  ‖x_{k+1} − x*‖² = ‖x_k − x* − ηₖ g_k‖²
                  = ‖x_k − x*‖² − 2ηₖ ⋅ ⟨g_k, x_k − x*⟩ + ηₖ² ⋅ ‖g_k‖²
  ```

Second, take the conditional expectation given `x_k`, then use convexity,
`⟨∇f(x_k), x_k − x*⟩ ≥ f(x_k) − f*`,

  ```
  𝔼[ ⟨g_k, x_k − x*⟩ ] = 𝔼[ ⟨𝔼[g_k | x_k], x_k − x*⟩ ]
                       = 𝔼[ ⟨∇f(x_k), x_k − x*⟩ ]
                       ≥ 𝔼[ f(x_k) − f* ]
  ```

where the first step is the tower property, since `x_k − x*` is fixed given
`x_k`. Taking expectations in the first step and substituting,

  ```
  2ηₖ ⋅ 𝔼[ f(x_k) − f* ] ≤ 𝔼‖x_k − x*‖² − 𝔼‖x_{k+1} − x*‖² + ηₖ² ⋅ G²
  ```

Third, sum over `k = 0, …, K−1`. The distances telescope,

  ```
  2 Σₖ ηₖ ⋅ 𝔼[ f(x_k) − f* ] ≤ ‖x₀ − x*‖² − 𝔼‖x_K − x*‖² + G² Σₖ ηₖ²
                             ≤ ‖x₀ − x*‖² + G² Σₖ ηₖ²
  ```

Fourth, `x̄_K` is a convex combination of the iterates with weights
`ηₖ / Σⱼ ηⱼ`, so Jensen's inequality for convex `f` gives,

  ```
  𝔼[ f(x̄_K) ] − f* ≤ Σₖ ηₖ ⋅ 𝔼[ f(x_k) − f* ] / Σₖ ηₖ
                   ≤ ( ‖x₀ − x*‖² + G² Σₖ ηₖ² ) / (2 Σₖ ηₖ)
  ```

Fifth, take a constant step `ηₖ = R / (G√K)` with `R ≥ ‖x₀ − x*‖`. Then
`Σₖ ηₖ = R√K / G` and `Σₖ ηₖ² = R² / G²`,

  ```
  𝔼[ f(x̄_K) ] − f* ≤ ( R² + G² ⋅ R²/G² ) / (2R√K / G)
                   = 2R² ⋅ G / (2R√K)
                   = R ⋅ G / √K
  ```

Hence,

  ```
  𝔼[ f(x̄_K) ] − f* ≤ ( ‖x₀ − x*‖² + G² Σₖ ηₖ² ) / (2 Σₖ ηₖ)
                   ≤ R ⋅ G / √K          for ηₖ = R / (G√K)
  ```

The step `R / (G√K)` minimizes `(R² + K G² η²) / (2Kη)` over constant `η`.
The order `RG/√K` matches the minimax lower bound for stochastic first-order
methods on G-Lipschitz convex functions (Nemirovski and Yudin), so it cannot
be improved without further structure such as smoothness or strong convexity.

Regularity: the proof needs `f` convex with a minimizer `x*`, `g_k`
conditionally unbiased given `x_k` (subgradients suffice in place of
`∇f`), `𝔼[‖g_k‖² | x_k] ≤ G²`, and steps `ηₖ` fixed in advance. With a
projection onto a closed convex set containing `x*`, the first step holds
with `≤` because projection is nonexpansive, and the bound is unchanged.
