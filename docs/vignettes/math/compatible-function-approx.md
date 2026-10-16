# Compatible Function Approximation

A compatible critic gives the exact policy gradient (Sutton et al. 2000), from
two facts:

1. at a least-squares fit, the error `Q^π − f_w` is orthogonal to `∇_w f_w`,
   and,
2. compatibility makes `∇_w f_w` equal the score, the only direction the
   policy gradient sees.

Let,

  ```
  ψ(s,a)       ≝ ∇_θ log π(a|s)                               (score)
  ∇_w f_w(s,a) = ψ(s,a)                                       (compatibility)
  ε(w)         ≝ 𝔼_{s∼d^π_ρ, a∼π}[ (Q^π(s,a) − f_w(s,a))² ]
  F            ≝ 𝔼_{s∼d^π_ρ, a∼π}[ ψ(s,a) ψ(s,a)ᵀ ]
  ```

and let `w` be a stationary point of `ε`, i.e., `∇_w ε(w) = 0`. All
expectations below are over `s ∼ d^π_ρ, a ∼ π(·|s)`.

First, stationarity makes the error orthogonal to the score,

  ```
  0 = −½ ∇_w ε(w)
    = 𝔼[ (Q^π(s,a) − f_w(s,a)) ∇_w f_w(s,a) ]
    = 𝔼[ (Q^π(s,a) − f_w(s,a)) ψ(s,a) ]
  ```

where the second step moves `∇_w` inside a finite sum, and the third is
compatibility.

Second, substitute into the policy gradient theorem (policy-gradient.md),

  ```
  ∇J(θ) = (1/(1−γ)) 𝔼[ Q^π(s,a) ψ(s,a) ]
        = (1/(1−γ)) 𝔼[ f_w(s,a) ψ(s,a) ] + (1/(1−γ)) 𝔼[ (Q^π(s,a) − f_w(s,a)) ψ(s,a) ]
        = (1/(1−γ)) 𝔼[ f_w(s,a) ψ(s,a) ]
  ```

where the last step uses the first step.

Third, `f_w` effectively fits the advantage. Compatibility for all `w` forces
`f_w = wᵀψ + c(s,a)`; take `c = 0`. Then, since `𝔼_{a∼π}[ψ(s,a)] = 0`
(policy-gradient.md),

  ```
  Σ_a π(a|s) f_w(s,a) = wᵀ Σ_a π(a|s) ψ(s,a)
                      = 0
  ```

so `f_w` has zero mean under `π`, like `A^π`. Fitting `A^π` instead of `Q^π`
gives the same stationary `w`, because the targets differ by `V^π(s)` and
`𝔼[ V^π(s) ψ(s,a) ] = 𝔼_s[ V^π(s) 𝔼_{a∼π}[ψ(s,a)] ] = 0`.

Fourth, the natural gradient. With `f_w = wᵀψ`,

  ```
  𝔼[ f_w(s,a) ψ(s,a) ] = 𝔼[ ψ(s,a) ψ(s,a)ᵀ w ]
                       = F w
  ```

so the stationarity condition is the normal equation `F w = 𝔼[ Q^π ψ ]`, and
`ε` is a convex quadratic, so stationary ⇔ global minimum. Combining with the
second step, `∇J(θ) = (1/(1−γ)) F w`.

Hence,

  ```
  ∇J(θ) = (1/(1−γ)) 𝔼_{s∼d^π_ρ, a∼π}[ f_w(s,a) ∇ log π(a|s) ]
  w     = (1−γ) F⁻¹ ∇J(θ)                                     (F ≻ 0)
  ```

`F` is the Fisher information of `π(·|s)` averaged over `d^π_ρ`, so `w` is the
natural gradient direction (Kakade 2001). The factor `1 − γ` comes from
normalizing `d^π_ρ`; with the unnormalized weights `Σ_t γ^t Pr(s_t = s)` in
`F`, `w = F⁻¹ ∇J(θ)` exactly. The gradient is exact for any `f_w` with
`Q^π − f_w` orthogonal to the span of the score; `f_w` itself can be far from
`Q^π`.

Regularity: the proof used the conditions of policy-gradient.md, `f_w`
differentiable in `w`, `w` stationary for `ε` (for nonlinear `f_w`, a local
stationary point suffices), and `F` invertible for the natural-gradient form.
