# Kalman Filter

The Kalman filter is exact Bayesian filtering, from two facts:

1. affine maps of independent Gaussians are jointly Gaussian, and,
2. conditioning a Gaussian is a Schur complement (gaussian-conditioning.md).

Let,

  ```
  x_{t+1} = A x_t + w_t,     w_t ∼ N(0, W)
  y_t     = H x_t + v_t,     v_t ∼ N(0, R),   R ≻ 0
  x_1 ∼ N(m_1⁻, Σ_1⁻);   x_1, (w_t), (v_t) mutually independent
  x_t | y_{1:t−1} ∼ N(m_t⁻, Σ_t⁻)                (predicted)
  x_t | y_{1:t}   ∼ N(m_t, Σ_t)                  (filtered)
  ```

Every `x_t`, `y_t` is affine in `(x_1, w, v)`, so all are jointly Gaussian.

First, Gaussian conditioning. Let `(x, y)` be jointly Gaussian with means
`m_x`, `m_y`, covariance blocks `Σ_xx`, `Σ_xy`, `Σ_yy ≻ 0`. Let
`G ≝ Σ_xy Σ_yy⁻¹` and `z ≝ x − G y`. Then

  ```
  𝕍(z, y) = Σ_xy − G Σ_yy
          = 0
  𝕍(z)    = Σ_xx − G Σ_yx − Σ_xy Gᵀ + G Σ_yy Gᵀ
          = Σ_xx − Σ_xy Σ_yy⁻¹ Σ_yx
  ```

where each of the three `G` terms equals `Σ_xy Σ_yy⁻¹ Σ_yx`. `(z, y)` is
jointly Gaussian and uncorrelated, hence independent. So given `y`,
`x = z + G y` is Gaussian with mean `m_x − G m_y + G y` and covariance
`Σ_xx − Σ_xy Σ_yy⁻¹ Σ_yx`, the Schur complement of `Σ_yy`. Given
`y_{1:t−1}`, `(x_t, y_t)` is again jointly Gaussian. Conditioning that law on
`y_t` gives the law given `y_{1:t}`.

Second, the update. Given `y_{1:t−1}`, `x_t ∼ N(m_t⁻, Σ_t⁻)` and `v_t` is
independent of `(x_t, y_{1:t−1})`, so

  ```
  𝔼[ y_t | y_{1:t−1} ]    = H m_t⁻
  𝕍(x_t, y_t | y_{1:t−1})   = 𝕍(x_t, H x_t + v_t) = Σ_t⁻ Hᵀ
  𝕍(y_t | y_{1:t−1})        = H Σ_t⁻ Hᵀ + R
  ```

and `H Σ_t⁻ Hᵀ + R ≻ 0` since `R ≻ 0`. The first step with `G = K` gives

  ```
  K   ≝ Σ_t⁻ Hᵀ (H Σ_t⁻ Hᵀ + R)⁻¹
  m_t = m_t⁻ + K (y_t − H m_t⁻)
  Σ_t = Σ_t⁻ − Σ_t⁻ Hᵀ (H Σ_t⁻ Hᵀ + R)⁻¹ H Σ_t⁻
      = (I − K H) Σ_t⁻
  ```

Third, the prediction. Given `y_{1:t}`, `x_t ∼ N(m_t, Σ_t)` and `w_t` is
independent of `(x_t, y_{1:t})`, so `A x_t + w_t` is Gaussian with

  ```
  m_{t+1}⁻ = 𝔼[ A x_t + w_t | y_{1:t} ]   = A m_t
  Σ_{t+1}⁻ = 𝕍( A x_t + w_t | y_{1:t} ) = A Σ_t Aᵀ + W
  ```

Hence,

  ```
  predict:  m_{t+1}⁻ = A m_t,              Σ_{t+1}⁻ = A Σ_t Aᵀ + W
  update:   K = Σ_t⁻ Hᵀ (H Σ_t⁻ Hᵀ + R)⁻¹
            m_t = m_t⁻ + K (y_t − H m_t⁻),  Σ_t = (I − K H) Σ_t⁻
  ```

These are the exact posteriors, so `m_t` is the minimum-mean-square-error
estimate of `x_t`. If the noise is non-Gaussian with the same first two
moments, the same recursion is the best affine estimator, but no longer the
posterior.

Fourth, duality with LQR. Substituting the update into the prediction,

  ```
  Σ_{t+1}⁻ = A Σ_t⁻ Aᵀ + W − A Σ_t⁻ Hᵀ (H Σ_t⁻ Hᵀ + R)⁻¹ H Σ_t⁻ Aᵀ
  ```

The LQR recursion (lqr-riccati.md) is

  ```
  P_t = Q + Aᵀ P_{t+1} A − Aᵀ P_{t+1} B (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A
  ```

Substituting `A → Aᵀ`, `B → Hᵀ`, `Q → W`, `R → R`, `P_{t+1} → Σ_t⁻`,
`P_t → Σ_{t+1}⁻` maps the LQR recursion onto the filter's, with time
reversed. The LQR gain maps to `(H Σ⁻ Hᵀ + R)⁻¹ H Σ⁻ Aᵀ = (A K)ᵀ`.
Controllability maps to observability. So a steady-state filter exists when
`(A, H)` is detectable and `(A, W^{1/2})` is stabilizable, the dual of the
LQR conditions.

Regularity: the proof used linear dynamics and observations, Gaussian
`x_1`, `w_t`, `v_t`, mutual independence of `x_1` and all noise terms, known
`A`, `H`, `W`, `R`, and `R ≻ 0` so the innovation covariance is invertible.
