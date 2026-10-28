# Two-Timescale Tracking

Two-timescale stochastic approximation runs a fast iterate and a slow
iterate together. The fast one tracks its equilibrium for the current slow
value, because the slow value moves by `O(α)` per step while the fast one
contracts by `O(β)` per step. This vignette proves the tracking bound for a
strongly convex quadratic fast problem; the general ODE result is Borkar
(Systems & Control Letters 1997, cited).

Let,

  ```
  b_k ∈ ℝᵐ                                      fast iterate
  w_k                                           slow iterate
  b*(w)                                         fast equilibrium for slow value w
  A ∈ ℝ^{m×m},  μI ⪯ A ⪯ ΛI,  μ > 0
  b_{k+1} = b_k − β ⋅ ( A (b_k − b*(w_k)) + ζ_k )
  d_k     ≝ b*(w_{k+1}) − b*(w_k)
  e_k     ≝ b_k − b*(w_k)
  E_k     ≝ 𝔼‖e_k‖²
  ```

Assume `0 < β ≤ 1/Λ`, `𝔼[ζ_k | F_k] = 0` and `𝔼[‖ζ_k‖² | F_k] ≤ σ_b²`,
`𝔼[⟨d_k, ζ_k⟩] = 0` (slow and fast noise uncorrelated), and
`𝔼‖d_k‖² ≤ (ℓ α U)²`: `b*` is `ℓ`-Lipschitz and the slow step has size
`α ⋅ u_k` with `𝔼‖u_k‖² ≤ U²`.

First, the error recursion,

  ```
  e_{k+1} = b_{k+1} − b*(w_{k+1})
          = (I − βA) e_k − β ζ_k − d_k
  ```

Second, take the conditional second moment. The `ζ_k` cross terms vanish
(unbiased, and uncorrelated with `d_k`),

  ```
  E_{k+1} = 𝔼‖(I − βA) e_k − d_k‖² + β² 𝔼‖ζ_k‖²
          ≤ 𝔼‖(I − βA) e_k − d_k‖² + β² σ_b²
  ```

Third, contract. `‖I − βA‖ ≤ 1 − βμ` since `β ≤ 1/Λ`. With
`(a + b)² ≤ (1 + t) a² + (1 + 1/t) b²` at `t = βμ ≤ 1`,

  ```
  𝔼‖(I − βA) e_k − d_k‖² ≤ (1 + βμ)(1 − βμ)² E_k + (1 + 1/(βμ)) 𝔼‖d_k‖²
                         ≤ (1 − βμ) E_k + (2/(βμ)) (ℓ α U)²
  ```

where the second step uses `(1 + t)(1 − t)² = (1 − t²)(1 − t) ≤ 1 − t` and
`1 + 1/t ≤ 2/t`.

Fourth, unroll the recursion `E_{k+1} ≤ (1 − βμ) E_k + D` with
`D = 2(ℓαU)²/(βμ) + β²σ_b²`, summing the geometric series,

  ```
  E_k ≤ (1 − βμ)ᵏ E₀ + D / (βμ)
      = (1 − βμ)ᵏ E₀ + 2 (ℓU/μ)² (α/β)² + β σ_b² / μ
  ```

Hence,

  ```
  limsup_k 𝔼‖b_k − b*(w_k)‖² ≤ 2 (ℓU/μ)² ⋅ (α/β)² + σ_b²/μ ⋅ β
  ```

The first term is lag (the slow variable moves while the fast one chases),
the second is noise. Both vanish when `α/β → 0` and `β → 0`: the slow
iterate sees the fast one at equilibrium. With constant steps the bound is
attained up to constants: with no noise and `w` drifting linearly, `e_k`
settles at lag `≈ ‖d‖/(βμ)`.

Fine print: Borkar's theorem drops the quadratic and Lipschitz-drift
assumptions in exchange for asymptotics: with steps `αₖ, βₖ` satisfying the
Robbins–Monro conditions (`robbins-monro.md`) and `αₖ/βₖ → 0`, the pair
converges almost surely to the equilibria of the slow ODE evaluated at
`b*(w)`, under stability conditions. Actor-critic is the standard instance:
the critic is fast, the actor slow.

Regularity: the proof needs `A` symmetric with `μI ⪯ A ⪯ ΛI`, `β ≤ 1/Λ`,
martingale-difference fast noise with bounded conditional second moment,
fast noise uncorrelated with the slow step's effect `d_k`, and the drift
bound `𝔼‖d_k‖² ≤ (ℓαU)²`.
