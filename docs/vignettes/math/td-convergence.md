# TD(0) Convergence

Linear TD(0) converges (Tsitsiklis & Van Roy 1997), from two facts:

1. the expected update `Aθ + b` has `A` negative definite, and,
2. its unique root solves the projected Bellman equation.

Let,

  ```
  P(s,s') ≝ Σ_a π(a|s) P(s'|s,a),     r^π(s) ≝ Σ_a π(a|s) r(s,a)
  μᵀ P     = μᵀ,  μ(s) > 0               (stationary distribution)
  D        ≝ diag(μ),   ‖v‖_D² ≝ vᵀ D v,   ⟨u, v⟩_D ≝ uᵀ D v
  Φ ∈ ℝ^{|S|×k} full column rank,   φ(s)ᵀ ≝ row s of Φ
  T v      ≝ r^π + γ P v
  Π        ≝ Φ (ΦᵀDΦ)⁻¹ Φᵀ D             (D-orthogonal projection onto range Φ)
  δ_t      ≝ r(s_t,a_t) + γ φ(s_{t+1})ᵀθ_t − φ(s_t)ᵀθ_t
  θ_{t+1} ≝ θ_t + α_t δ_t φ(s_t)
  ```

First, the expected update. Fix `θ_t = θ`, draw `s_t ∼ μ`, `a_t ∼ π(·|s_t)`,
`s_{t+1} ∼ P(·|s_t,a_t)`,

  ```
  𝔼[ δ_t φ(s_t) ] = Σ_s μ(s) φ(s) ( r^π(s) + γ Σ_{s'} P(s,s') φ(s')ᵀθ − φ(s)ᵀθ )
                  = Φᵀ D r^π + Φᵀ D (γ P Φ − Φ) θ
                  = b + A θ
  ```

with `b ≝ Φᵀ D r^π` and `A ≝ Φᵀ D (γ P Φ − Φ)`.

Second, the key lemma: `P` is non-expansive in `‖·‖_D`,

  ```
  ‖P v‖_D² = Σ_s μ(s) ( Σ_{s'} P(s,s') v(s') )²
           ≤ Σ_s μ(s) Σ_{s'} P(s,s') v(s')²
           = Σ_{s'} ( Σ_s μ(s) P(s,s') ) v(s')²
           = Σ_{s'} μ(s') v(s')²
           = ‖v‖_D²
  ```

where the second step is Jensen (`P(s,·)` is a distribution, `x²` is
convex) and the fourth is stationarity.

Third, `A` is negative definite. Let `v ≝ Φθ`,

  ```
  θᵀ A θ = γ vᵀ D P v − vᵀ D v
         ≤ γ ‖v‖_D ‖P v‖_D − ‖v‖_D²
         ≤ γ ‖v‖_D² − ‖v‖_D²
         = −(1 − γ) ‖Φθ‖_D²
         ≤ −(1 − γ) λ_min(ΦᵀDΦ) ‖θ‖²
  ```

where the second step is Cauchy–Schwarz in `⟨·,·⟩_D`, the third is the lemma,
and `ΦᵀDΦ ≻ 0` because `Φ` has full column rank and `μ > 0`. So
`θᵀAθ < 0` for `θ ≠ 0`, and `A` is nonsingular (`Aθ = 0` ⇒ `θᵀAθ = 0` ⇒
`θ = 0`). `A` need not be symmetric.

Fourth, the root `θ* ≝ −A⁻¹ b` solves the projected Bellman equation,

  ```
  Aθ* + b = 0  ⇔  Φᵀ D ( T Φθ* − Φθ* ) = 0
               ⇔  ΦᵀDΦ θ* = Φᵀ D T Φθ*
               ⇔  θ* = (ΦᵀDΦ)⁻¹ Φᵀ D T Φθ*
               ⇔  Φθ* = Π T Φθ*
  ```

where the last `⇐` uses that `Φ` is injective. With `V ≝ V^π = T V`,
`ΠV − V` is `D`-orthogonal to range `Φ`, so by Pythagoras,

  ```
  ‖Φθ* − V‖_D² = ‖Φθ* − ΠV‖_D² + ‖ΠV − V‖_D²
               = ‖Π T Φθ* − Π T V‖_D² + ‖ΠV − V‖_D²
               ≤ γ² ‖Φθ* − V‖_D² + ‖ΠV − V‖_D²
  ```

using `‖Π‖_D ≤ 1` and `‖T u − T w‖_D = γ ‖P (u − w)‖_D ≤ γ ‖u − w‖_D`.

Fifth, convergence. With i.i.d. samples `s_t ∼ μ`, TD(0) is Robbins–Monro
(robbins-monro.md) with `h(θ) = −(Aθ + b) = −A(θ − θ*)`. Its conditions hold:
`(θ − θ*)ᵀ h(θ) ≥ (1 − γ) λ_min(ΦᵀDΦ) ‖θ − θ*‖²` by the third step,
`‖h(θ)‖ ≤ ‖A‖ ‖θ − θ*‖`, and the noise `ξ_t = −(δ_t φ(s_t) − Aθ_t − b)` has
zero conditional mean and, with `‖φ‖ ≤ φ_max`, `|r| ≤ R_max`,

  ```
  𝔼[ ‖ξ_t‖² | 𝓕_t ] ≤ 𝔼[ ‖δ_t φ(s_t)‖² | 𝓕_t ]
                    ≤ φ_max² ( R_max + (1 + γ) φ_max ‖θ_t‖ )²
                    ≤ 2 φ_max² ( R_max + 2 φ_max ‖θ*‖ )² + 8 φ_max⁴ ‖θ_t − θ*‖²
  ```

using `(x + y)² ≤ 2x² + 2y²` and `‖θ_t‖ ≤ ‖θ*‖ + ‖θ_t − θ*‖`. For a single
Markov trajectory the noise is not a martingale difference. Tsitsiklis & Van
Roy (1997, Thm. 1) prove `θ_t → θ*` a.s. for an irreducible aperiodic finite
chain, full-rank `Φ`, and positive, nonincreasing, deterministic `α_t` with
`Σ α_t = ∞`, `Σ α_t² < ∞`; this step is cited, not proved here. Its ODE is
`θ̇ = Aθ + b`, globally asymptotically stable because
`d/dt ‖θ − θ*‖² = 2 (θ − θ*)ᵀ A (θ − θ*) < 0` for `θ ≠ θ*`.

Hence,

  ```
  θ_t → θ* = −A⁻¹ b   a.s.
  Φθ*           = Π T Φθ*
  ‖Φθ* − V^π‖_D ≤ ‖Π V^π − V^π‖_D / √(1 − γ²)
  ```

The bound is tight in the sense that `Φθ* = V^π` when `V^π ∈ range Φ`.

Regularity: the proof used finite `S`, `|r| ≤ R_max`, `γ < 1`, full column
rank `Φ`, `μ > 0`, and the on-policy weighting `D = diag(μ)` with `μ`
stationary for `P`. The lemma fails for other weightings; off-policy TD can
diverge (Baird 1995).
