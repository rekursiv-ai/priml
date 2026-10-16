# LQR and the Riccati Equation

The LQR value function is quadratic by induction, from two facts:

1. a quadratic cost-to-go composed with linear dynamics stays quadratic, and,
2. completing the square in `u` gives the minimizer and the minimum in closed
   form.

Let,

  ```
  x_{t+1} = A x_t + B u_t,          x ∈ ℝ^n,  u ∈ ℝ^m
  c(x,u)  ≝ xᵀ Q x + uᵀ R u,        Q ⪰ 0,  R ≻ 0
  V_N(x)  ≝ xᵀ Q_f x,               Q_f ⪰ 0
  V_t(x)  ≝ min_u [ c(x,u) + V_{t+1}(A x + B u) ]
  ```

where the last line is dynamic programming: after choosing `u_t`, the rest is
the same problem from `x_{t+1}`. Claim: `V_t(x) = xᵀ P_t x` with
`P_t = P_tᵀ ⪰ 0`. The base case is `P_N = Q_f`.

First, assume `V_{t+1}(y) = yᵀ P y` with `P = P_{t+1}` symmetric `⪰ 0`. Let
`S ≝ R + BᵀPB ≻ 0` and `K ≝ S⁻¹ BᵀPA`, so `S K = BᵀPA`. Then

  ```
  c(x,u) + V_{t+1}(Ax + Bu) = xᵀQx + uᵀRu + (Ax + Bu)ᵀ P (Ax + Bu)
                            = xᵀ(Q + AᵀPA)x + 2 uᵀBᵀPAx + uᵀ(R + BᵀPB)u
                            = xᵀ(Q + AᵀPA)x + 2 uᵀSKx + uᵀSu
                            = xᵀ(Q + AᵀPA)x + (u + Kx)ᵀ S (u + Kx) − xᵀKᵀSKx
  ```

where the second step uses `P = Pᵀ`, and the last expands
`(u + Kx)ᵀ S (u + Kx) = uᵀSu + 2 uᵀSKx + xᵀKᵀSKx` with `S = Sᵀ`.

Second, minimize over `u`. `S ≻ 0`, so the square is `≥ 0` with equality iff
`u = −Kx`,

  ```
  V_t(x) = xᵀ(Q + AᵀPA − KᵀSK)x
         = xᵀ(Q + AᵀPA − AᵀPB S⁻¹ BᵀPA)x
  ```

using `KᵀSK = AᵀPB S⁻¹ S S⁻¹ BᵀPA`.

Third, `P_t` is symmetric and `⪰ 0`, which closes the induction. Using
`KᵀBᵀPA = KᵀSK = AᵀPBK`,

  ```
  (A − BK)ᵀ P (A − BK) + KᵀRK = AᵀPA − AᵀPBK − KᵀBᵀPA + Kᵀ(BᵀPB + R)K
                              = AᵀPA − AᵀPBK − KᵀSK + KᵀSK
                              = AᵀPA − AᵀPB S⁻¹ BᵀPA
  ```

so `P_t = Q + (A − BK)ᵀ P (A − BK) + KᵀRK`, a sum of symmetric `⪰ 0` terms.

Hence,

  ```
  V_t(x) = xᵀ P_t x,      u_t* = −K_t x_t
  K_t    ≝ (R + BᵀP_{t+1}B)⁻¹ BᵀP_{t+1}A
  P_t    ≝ Q + AᵀP_{t+1}A − AᵀP_{t+1}B (R + BᵀP_{t+1}B)⁻¹ BᵀP_{t+1}A,    P_N ≝ Q_f
  ```

The minimizer is unique because `S ≻ 0`. With additive noise
`x_{t+1} = Ax + Bu + w`, `𝔼w = 0`, `𝕍 w = W`, the cross term vanishes in
expectation, so `K_t` is unchanged and `V_t(x) = xᵀP_t x + c_t` with
`c_t = c_{t+1} + tr(P_{t+1} W)` (certainty equivalence).

Fourth, the infinite horizon. A fixed point of the recursion solves the
algebraic Riccati equation (ARE),

  ```
  P = Q + AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA
  ```

Suppose `P ⪰ 0` solves it and `A − BK` is Schur stable, with
`K = (R + BᵀPB)⁻¹ BᵀPA`. The first step, with the ARE replacing
`Q + AᵀPA − KᵀSK` by `P`, gives along any trajectory

  ```
  c(x_t,u_t) + x_{t+1}ᵀ P x_{t+1} − x_tᵀ P x_t = (u_t + Kx_t)ᵀ S (u_t + Kx_t)
  ```

Summing over `t < N` telescopes. For any controls with `x_t → 0`,

  ```
  Σ_{t<N} c(x_t,u_t) = x₀ᵀPx₀ − x_NᵀPx_N + Σ_{t<N} (u_t + Kx_t)ᵀ S (u_t + Kx_t)
                     → x₀ᵀPx₀ + Σ_t (u_t + Kx_t)ᵀ S (u_t + Kx_t)
                     ≥ x₀ᵀPx₀
  ```

with equality iff `u_t = −Kx_t`, which drives `x_t → 0` because `A − BK` is
Schur. So `u = −Kx` is optimal among controls with `x_t → 0`, at cost
`x₀ᵀPx₀`.

Existence is cited, not proved here (Kučera 1972; Anderson & Moore 1990): if
`(A, B)` is stabilizable and `(A, Q^{1/2})` is detectable, the ARE has a
unique solution `P ⪰ 0`, it makes `A − BK` Schur, every finite-cost control
has `x_t → 0`, and the recursion converges to `P` from any `Q_f ⪰ 0`.
Without stabilizability the cost can be infinite; without detectability an
unstable mode can be invisible to the cost.

Regularity: the proof used `R ≻ 0` (so `S ≻ 0` for any `P ⪰ 0`), `Q ⪰ 0`,
`Q_f ⪰ 0`, known `A`, `B`, unconstrained `u`, and, for the infinite horizon, a
stabilizing ARE solution.
