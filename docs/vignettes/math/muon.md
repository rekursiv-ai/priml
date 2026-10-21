# Muon

Muon's update follows from one fact: the dual of the spectral norm is the
nuclear norm, so steepest descent on a matrix under the spectral norm moves
in the direction `−U Vᵀ`, the gradient with every singular value set to one.

Let `G ∈ ℝ^{m×n}` be a gradient (or momentum) with thin SVD
`G = U Σ Vᵀ`, `Σ = diag(σ₁, …, σ_r)`, `σᵢ > 0`, and,

  ```
  ⟨A, B⟩ ≝ tr(AᵀB)
  ‖D‖₂   ≝ σ_max(D)              spectral norm
  ‖G‖_*  ≝ Σᵢ σᵢ(G)               nuclear norm
  polar(G) ≝ U Vᵀ
  ```

First, a lower bound on the linear model over the spectral ball. For any `D`
with `‖D‖₂ ≤ 1`,

  ```
  ⟨G, D⟩ = tr( V Σ Uᵀ D )
         = Σᵢ σᵢ ⋅ uᵢᵀ D vᵢ
         ≥ −Σᵢ σᵢ ⋅ |uᵢᵀ D vᵢ|
         ≥ −Σᵢ σᵢ
  ```

where the last step uses `|uᵀDv| ≤ ‖u‖ ‖D‖₂ ‖v‖ ≤ 1` for unit `u, v`.

Second, the bound is attained. Take `D = −U Vᵀ`, which has every singular
value `1`, so `‖D‖₂ = 1`, and `uᵢᵀ(−U Vᵀ)vᵢ = −1`,

  ```
  ⟨G, −U Vᵀ⟩ = −Σᵢ σᵢ = −‖G‖_*
  ```

Hence,

  ```
  argmin_{‖D‖₂ ≤ 1} ⟨G, D⟩ = −polar(G),    min = −‖G‖_*
  ```

unique when `G` has full rank `r = min(m, n)`. Muon steps
`W ← W − η ⋅ polar(M)` with `M` a Nesterov momentum of the gradient, scaled
by `√(m/n)` or similar so update size is comparable across shapes.

Third, computing `polar(M)` without an SVD. The Newton–Schulz iteration

  ```
  X₀ = M / ‖M‖_F,    X_{k+1} = ½ X_k (3I − X_kᵀ X_k)
  ```

acts on each singular value by `σ ↦ ½ σ (3 − σ²)`, whose fixed point on
`(0, √3)` is `1`, so `X_k → U Vᵀ`. Muon uses a quintic odd polynomial tuned
to push small singular values up quickly, needing about five matrix
multiplications and no decomposition.

Examples. Each is steepest descent, `argmin_{‖d‖≤1} ⟨g, d⟩`, in a different
norm:

  ```
  ‖·‖₂ (vector)      d = −g / ‖g‖₂                 normalized GD
  ‖·‖∞ (vector)      d = −sign(g)                  sign descent (Adam, β → 0)
  ‖·‖_H              d ∝ −H⁻¹ g                    Newton (newton-as-proximal.md)
  ‖·‖₂ (spectral)    d = −U Vᵀ                     Muon
  ```

Adam and sign descent equalize coordinates; Muon equalizes directions in the
row and column spaces, so a gradient dominated by a few large singular
directions is not allowed to spend the whole step on them.

Regularity: the derivation is exact for any matrix `G`; for rank-deficient
`G` the minimizer is `−U Vᵀ` plus any `D` acting on the null spaces with
`‖D‖₂ ≤ 1`. Newton–Schulz converges for singular values of `X₀` in
`(0, √3)`, guaranteed by the Frobenius normalization. Muon is applied to 2-D
weight matrices; embeddings, gains, and biases are typically left to Adam.
Convergence guarantees for non-convex training are not proved here.
