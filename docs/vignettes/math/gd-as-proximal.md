# Gradient Descent as a Proximal Step

A gradient step minimizes a linear model of `f` plus a quadratic penalty on
moving. Keeping a nonsmooth `g` exact in that model gives the proximal
gradient method, which keeps GD's `O(1/k)` rate.

Let,

  ```
  f : ℝⁿ → ℝ convex and L-smooth
  g : ℝⁿ → ℝ ∪ {+∞} convex, closed, proper
  F            ≝ f + g,   x* a minimizer of F,   F* ≝ F(x*)
  prox_{ηg}(v) ≝ argmin_y { g(y) + (1/2η) ⋅ ‖y − v‖² }
  ```

`prox_{ηg}` is single-valued: its objective is closed and 1/η-strongly
convex, so it has exactly one minimizer.

First, the gradient step is the minimizer of a regularized linear model.
Completing the square,

  ```
  f(x) + ⟨∇f(x), y − x⟩ + (1/2η) ⋅ ‖y − x‖²
      = f(x) + (1/2η) ⋅ ‖y − x + η∇f(x)‖² − (η/2) ⋅ ‖∇f(x)‖²
  ```

which follows from expanding
`‖d + η∇f‖² = ‖d‖² + 2η⟨d, ∇f⟩ + η²‖∇f‖²` with `d = y − x`. The only term
depending on `y` is the square, so

  ```
  argmin_y { f(x) + ⟨∇f(x), y − x⟩ + (1/2η) ⋅ ‖y − x‖² } = x − η∇f(x)
  ```

With `η = 1/L` the model upper-bounds `f` (descent-lemma.md), so GD is
majorize–minimize.

Second, add `g` to the model without linearizing it. The same square appears,
and terms constant in `y` drop out,

  ```
  x⁺ ≝ argmin_y { f(x) + ⟨∇f(x), y − x⟩ + g(y) + (1/2η) ⋅ ‖y − x‖² }
     = argmin_y { g(y) + (1/2η) ⋅ ‖y − (x − η∇f(x))‖² }
     = prox_{ηg}( x − η∇f(x) )
  ```

This is proximal gradient descent (ISTA when `g = λ‖·‖₁`).

Third, three choices of `g` recover known methods,

  ```
  g = 0          ⇒  prox_{ηg}(v) = v                              (GD)
  g = ι_C        ⇒  prox_{ηg}(v) = argmin_{y∈C} ‖y − v‖ = Π_C(v)  (projected GD)
  g = λ‖·‖₁      ⇒  prox_{ηg}(v)ᵢ = sign(vᵢ) ⋅ max(|vᵢ| − ηλ, 0)   (soft-threshold)
  ```

where `ι_C` is 0 on a closed convex `C` and `+∞` off it. For `λ‖·‖₁` the
problem separates by coordinate. The optimality condition
`0 ∈ ηλ ⋅ ∂|y| + (y − v)` gives `y = v − ηλ` if `v > ηλ`, `y = v + ηλ` if
`v < −ηλ`, and `y = 0` if `|v| ≤ ηλ`, since `∂|0| = [−1, 1]`.

Fourth, a one-step inequality. Take `η = 1/L`. The optimality condition of
the prox gives `L ⋅ (x − x⁺) − ∇f(x) ∈ ∂g(x⁺)`, so for every `z`,

  ```
  g(x⁺) ≤ g(z) − ⟨L ⋅ (x − x⁺) − ∇f(x), z − x⁺⟩
  ```

Add the descent lemma `f(x⁺) ≤ f(x) + ⟨∇f(x), x⁺ − x⟩ + (L/2) ⋅ ‖x⁺ − x‖²`
and convexity `f(x) ≤ f(z) + ⟨∇f(x), x − z⟩`. The `∇f(x)` terms cancel:
`⟨∇f(x), x − z + x⁺ − x + z − x⁺⟩ = 0`. With `d = x − x⁺`,

  ```
  F(x⁺) ≤ F(z) − L ⋅ ⟨d, z − x⁺⟩ + (L/2) ⋅ ‖d‖²
        = F(z) + L ⋅ ⟨d, x − z⟩ − L ⋅ ‖d‖² + (L/2) ⋅ ‖d‖²
        = F(z) + L ⋅ ⟨d, x − z⟩ − (L/2) ⋅ ‖d‖²
        = F(z) + (L/2) ⋅ ( ‖x − z‖² − ‖x − z − d‖² )
        = F(z) + (L/2) ⋅ ( ‖x − z‖² − ‖x⁺ − z‖² )
  ```

where the second step writes `x⁺ − z = (x − z) − d`, and the fourth expands
`‖x − z − d‖²`.

Fifth, iterate `x_{k+1} ≝ prox_{g/L}(x_k − ∇f(x_k)/L)`. Setting `z = x_k`
gives `F(x_{k+1}) ≤ F(x_k) − (L/2) ⋅ ‖x_{k+1} − x_k‖²`, so `F(x_k)` is
nonincreasing. Setting `z = x*` and summing over `k = 0, …, K−1`, the
distances telescope,

  ```
  K ⋅ (F(x_K) − F*) ≤ Σ_{k=0}^{K−1} (F(x_{k+1}) − F*)
                    ≤ (L/2) ⋅ ( ‖x₀ − x*‖² − ‖x_K − x*‖² )
                    ≤ (L/2) ⋅ ‖x₀ − x*‖²
  ```

Hence,

  ```
  x_{k+1}     = prox_{g/L}( x_k − ∇f(x_k)/L )
  F(x_K) − F* ≤ L ⋅ ‖x₀ − x*‖² / (2K)
  ```

With `g = 0` this is exactly the convex GD rate (gd-convergence.md). The
nonsmooth `g` costs nothing because it is never linearized; only `f` pays
the `L/2` curvature term. Adding momentum as in nesterov.md (FISTA) gives
`O(1/K²)`.

Regularity: the proof needs `f` convex and L-smooth on `dom g`, `g` convex,
closed, and proper (so the prox exists, is unique, and the subgradient
optimality condition holds), a minimizer `x*` of `F`, and step exactly
`η = 1/L` (any `η ≤ 1/L` works with `L` replaced by `1/η`).
