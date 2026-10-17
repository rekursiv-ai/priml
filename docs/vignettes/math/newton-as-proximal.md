# Newton's Method as a Proximal Step

Newton's step is gradient descent with the penalty `½‖·‖²` replaced by
`½‖·‖²_H`, where `H` is the Hessian. Its local quadratic rate follows from
the fundamental theorem of calculus and a Lipschitz Hessian.

Let,

  ```
  f : ℝⁿ → ℝ twice differentiable
  ∇²f(x)            ⪰ μ I                    for all x   (μ-strongly convex)
  ‖∇²f(x) − ∇²f(y)‖ ≤ M ⋅ ‖x − y‖            for all x, y (operator norm)
  g                 ≝ ∇f(x),   H ≝ ∇²f(x),   ‖d‖_H ≝ √(dᵀHd)
  x⁺                ≝ x − H⁻¹g
  ```

Strong convexity makes `H` invertible with `‖H⁻¹‖ ≤ 1/μ`, and gives a unique
minimizer `x*` with `∇f(x*) = 0`.

First, the Newton step minimizes the local quadratic model. With
`d ≝ y − x`, complete the square,

  ```
  f(x) + ⟨g, d⟩ + ½ dᵀHd = f(x) − ½ gᵀH⁻¹g + ½ (d + H⁻¹g)ᵀ H (d + H⁻¹g)
  ```

which follows from expanding
`½ (d + H⁻¹g)ᵀH(d + H⁻¹g) = ½ dᵀHd + ⟨g, d⟩ + ½ gᵀH⁻¹g`. Since `H ≻ 0`, the
square is minimized, uniquely, at `d = −H⁻¹g`, i.e.,

  ```
  argmin_y { f(x) + ⟨g, y − x⟩ + ½ ‖y − x‖²_H } = x − H⁻¹g = x⁺
  ```

This is the proximal form of GD (gd-as-proximal.md) with `(1/2η) ⋅ ‖·‖²`
replaced by `½ ‖·‖²_H`.

Second, the Newton direction is steepest descent in `‖·‖_H`. Substitute
`w ≝ H^{1/2} d`, so `‖d‖_H = ‖w‖`, and apply Cauchy–Schwarz,

  ```
  min_{‖d‖_H ≤ 1} ⟨g, d⟩ = min_{‖w‖ ≤ 1} ⟨H^{−1/2} g, w⟩
                         = −‖H^{−1/2} g‖
                         = −√(gᵀH⁻¹g)
  ```

attained at `w = −H^{−1/2}g / ‖H^{−1/2}g‖`, i.e.,
`d = −H⁻¹g / √(gᵀH⁻¹g)`, a positive multiple of the Newton step.

Third, express the error after one step. Using `∇f(x*) = 0` and the
fundamental theorem of calculus along `z(t) = x* + t ⋅ (x − x*)`,

  ```
  x⁺ − x* = x − x* − H⁻¹ ⋅ (∇f(x) − ∇f(x*))
          = H⁻¹ ⋅ [ H ⋅ (x − x*) − ∫₀¹ ∇²f(z(t)) ⋅ (x − x*) dt ]
          = H⁻¹ ⋅ ∫₀¹ [ ∇²f(x) − ∇²f(z(t)) ] ⋅ (x − x*) dt
  ```

Fourth, bound its norm with `‖H⁻¹‖ ≤ 1/μ` and the Lipschitz Hessian,

  ```
  ‖x⁺ − x*‖ ≤ ‖H⁻¹‖ ⋅ ∫₀¹ ‖∇²f(x) − ∇²f(z(t))‖ dt ⋅ ‖x − x*‖
            ≤ (1/μ) ⋅ ∫₀¹ M ⋅ ‖x − z(t)‖ dt ⋅ ‖x − x*‖
            = (1/μ) ⋅ ∫₀¹ M ⋅ (1 − t) dt ⋅ ‖x − x*‖²
            = (M / (2μ)) ⋅ ‖x − x*‖²
  ```

where the third step uses `x − z(t) = (1 − t) ⋅ (x − x*)`.

Fifth, iterate. Let `e_k ≝ (M/(2μ)) ⋅ ‖x_k − x*‖`. The fourth step reads
`e_{k+1} ≤ e_k²`, so by induction,

  ```
  e_k ≤ e_{k−1}²
      ≤ e_{k−2}⁴
      ≤ e₀^{2^k}
  ```

Hence, if `‖x₀ − x*‖ < 2μ/M`, so that `q ≝ e₀ < 1`,

  ```
  ‖x_{k+1} − x*‖ ≤ (M / (2μ)) ⋅ ‖x_k − x*‖²
  ‖x_k − x*‖     ≤ (2μ / M) ⋅ q^{2^k}
  ```

The number of correct digits doubles each step. For quadratic `f`, `M = 0`
and the fourth step gives `x⁺ = x*`: Newton is exact in one step. Outside
the radius `2μ/M` the pure step can diverge; damping or a line search is
then needed.

Regularity: the proof needs `f` twice differentiable with `∇²f` M-Lipschitz
on the segment `[x*, x]` (for the integral and its bound), `∇²f(x) ⪰ μ I`
at each iterate (so `H⁻¹` exists and `‖H⁻¹‖ ≤ 1/μ`), a minimizer `x*` with
`∇f(x*) = 0`, `M > 0` for the radius, and the start within `2μ/M` of `x*`.
Since `e_k` decreases, every iterate stays inside that radius.
