# Frank–Wolfe

Frank–Wolfe converges at `O(1/k)` because of two facts:

1. the linear-minimization gap upper-bounds the suboptimality, and,
2. the descent lemma along `s_k − x_k` costs at most `(L/2) γ² D²`.

Let,

  ```
  C ⊆ ℝⁿ compact and convex,   D ≝ max_{x,y∈C} ‖x − y‖
  f : ℝⁿ → ℝ convex and L-smooth,   x* ∈ argmin_C f,   f* ≝ f(x*)
  s_k     ≝ argmin_{s∈C} ⟨∇f(x_k), s⟩
  x_{k+1} ≝ x_k + γ_k ⋅ (s_k − x_k),   γ_k ≝ 2/(k+2),   x₀ ∈ C
  gap_k   ≝ ⟨∇f(x_k), x_k − s_k⟩
  h_k     ≝ f(x_k) − f*
  ```

Each `x_{k+1}` is a convex combination of points of `C`, so it lies in `C`.

First, the gap certifies suboptimality. Since `x* ∈ C`, `s_k` does at least
as well as `x*` on the linear objective,

  ```
  gap_k = ⟨∇f(x_k), x_k − s_k⟩
        ≥ ⟨∇f(x_k), x_k − x*⟩
        ≥ f(x_k) − f*
        = h_k
  ```

where the second step is convexity, `f(x*) ≥ f(x_k) + ⟨∇f(x_k), x* − x_k⟩`.

Second, one step decreases `h` by `γ_k ⋅ gap_k`, up to a curvature term.
Apply the descent lemma (descent-lemma.md) with `y − x = γ_k ⋅ (s_k − x_k)`,

  ```
  f(x_{k+1}) ≤ f(x_k) + γ_k ⋅ ⟨∇f(x_k), s_k − x_k⟩ + (L/2) ⋅ γ_k² ⋅ ‖s_k − x_k‖²
             = f(x_k) − γ_k ⋅ gap_k + (L/2) ⋅ γ_k² ⋅ ‖s_k − x_k‖²
             ≤ f(x_k) − γ_k ⋅ h_k + (L/2) ⋅ γ_k² ⋅ D²
  ```

where the last step uses the first step and `s_k, x_k ∈ C`. Subtracting
`f*`,

  ```
  h_{k+1} ≤ (1 − γ_k) ⋅ h_k + γ_k² ⋅ LD²/2
  ```

Third, the base case `k = 1`. Since `γ₀ = 1`, the first term vanishes,

  ```
  h₁ ≤ 0 ⋅ h₀ + LD²/2
     ≤ 2LD²/3
  ```

Fourth, the induction step. Assume `h_k ≤ 2LD²/(k+2)` for some `k ≥ 1`.
Substitute `γ_k = 2/(k+2)` into the second step,

  ```
  h_{k+1} ≤ (k/(k+2)) ⋅ 2LD²/(k+2) + (4/(k+2)²) ⋅ LD²/2
          = 2LD² ⋅ (k + 1) / (k+2)²
          ≤ 2LD² / (k+3)
  ```

where the last step uses `(k+1)(k+3) = (k+2)² − 1 ≤ (k+2)²`.

Hence, for all `k ≥ 1`,

  ```
  f(x_k) − f* ≤ gap_k
  f(x_k) − f* ≤ 2LD² / (k+2)
  ```

The first line makes `gap_k` a computable stopping certificate: it costs
only the linear minimization the method already performs, and needs neither
`f*` nor `x*`. The `O(1/k)` order cannot be improved for any method that
adds one vertex per step. On `C = Δₙ` with `f(x) = ‖x‖₂²`, an iterate with
at most `k+1` nonzeros has `f(x_k) ≥ 1/(k+1)` by Cauchy–Schwarz, while
`f* = 1/n` (Jaggi, 2013).

Fine print: `h₀` is unbounded by `LD²` in general, which is why the bound
starts at `k = 1`; the `γ₀ = 1` step erases `x₀`. Line search,
`γ_k = argmin_{γ∈[0,1]} f(x_k + γ(s_k − x_k))`, satisfies the same
recursion, since its decrease is at least that of `γ = 2/(k+2)`. The gap
itself also converges at `O(1/k)` in its running minimum (Jaggi, 2013). The
method is projection-free: on the simplex or a nuclear-norm ball, `s_k` is a
vertex or a rank-one matrix, so iterates are sparse or low rank.

Regularity: the proof needs `C` compact and convex (so `s_k` exists,
`x_k ∈ C`, and `D < ∞`), `f` convex (first step) and L-smooth on `C` in the
same norm used for `D` (second step), and the step `γ_k = 2/(k+2)` with
`γ₀ = 1`. Ties in `argmin` may be broken arbitrarily.
