# Minimum Fisher Information

The minimum-Fisher-information density follows from two facts:

1. with `u ≝ √p` as the variable, location Fisher information is a Dirichlet
   energy, `I = 4 ∫ (u′)²`, and,
2. minimizing a Dirichlet energy subject to `∫ u² = 1` and moment constraints
   is a Schrödinger eigenproblem, whose ground state is the minimizer.

So where maximum entropy puts the constraint in the exponent, minimum Fisher
information puts it in a potential.

Let `Y` have density `p` on an interval `𝒴 ⊆ ℝ`, and,

  ```
  I(p) ≝ ∫ p′(y)² / p(y) dy          Fisher information for location
  T    : 𝒴 → ℝᵈ                      constraint statistic
  τ    ∈ ℝᵈ                           target moment
  ```

`I(p)` is the information in one draw about a shift `θ` of `p(y − θ)`, and
`I(p)⁻¹` is the Cramér–Rao floor for estimating it (cramer-rao.md). The
problem is `min_p I(p)` subject to `∫ p = 1` and `𝔼_p[T] = τ`: the density,
among those meeting the constraints, under which the location is hardest to
learn.

First, the substitution. Write `p = u²` with `u ≥ 0`. Then `p′ = 2u u′`, and

  ```
  I(p) = ∫ (2u u′)² / u² dy
       = 4 ∫ (u′)² dy
  ```

The constraints become `∫ u² = 1` and `∫ T u² = τ`. The objective is convex
in `u` and the constraints are quadratic, so the problem is a Rayleigh
quotient.

Second, stationarity. The Lagrangian with multipliers `η` (moments) and `λ`
(normalization) is

  ```
  ℒ(u) = 4 ∫ (u′)² + ∫ (ηᵀT) u² − λ ( ∫ u² − 1 )
  ```

Its variation in `u`, after integrating `∫ u′ δu′` by parts (with `u u′ → 0`
at the ends of `𝒴`), vanishes iff

  ```
  −4 u″ + (ηᵀT) u = λ u
  ```

a Schrödinger equation with potential `V ≝ ηᵀT` and energy `λ`.

Third, the minimum is the ground state. Multiply the equation by `u` and
integrate, using `∫ u² = 1`,

  ```
  4 ∫ (u′)² + ∫ V u² = λ
  I(p)               = λ − ηᵀτ
  ```

so for fixed `η` the smallest `I` comes from the smallest eigenvalue `λ₀`.
Its eigenfunction `u₀` has no zeros in the interior (the ground state of a
one-dimensional Schrödinger operator is nodeless), so `p = u₀²` is a
positive density. `η` is then tuned so that `∫ T u₀² = τ`.

Hence,

  ```
  p⋆ = u₀²,    −4 u₀″ + (ηᵀT) u₀ = λ₀ u₀,    I(p⋆) = λ₀ − ηᵀτ
  ```

and `I(p) ≥ I(p⋆)` for every feasible `p`.

Examples.

- Gaussian. `𝒴 = ℝ`, `T(y) = y²`, `τ = σ²`. The potential `η y²` is the
  harmonic oscillator; its ground state is `u₀ ∝ exp(−y²/(4σ²))`, so

    ```
    p⋆ = 𝒩(0, σ²),    I(p⋆) = 1/σ²
    ```

  the same answer as maximum entropy (max-entropy.md), and Cramér–Rao for
  location: `𝕍(Y) ⋅ I(p) ≥ 1` with equality exactly here.
- Bounded support. `𝒴 = [0, L]`, no `T`. The potential is zero with
  `u(0) = u(L) = 0`; the ground state is `u₀ ∝ sin(πy/L)`, so

    ```
    p⋆(y) = (2/L) sin²(πy/L),    I(p⋆) = 4π²/L²
    ```

  Maximum entropy on the same support gives the uniform density, which has
  `I = ∞` (its edges jump). Minimum Fisher information prefers densities
  that fade to zero.
- Contamination. Minimize over `{ (1 − ε) f + ε h }` instead of a moment
  set. The constraint `u² ≥ (1 − ε) f` binds in a core and leaves a free
  region where `u″ = (λ/4) u`, giving exponential tails: Huber's least
  favorable density `p_c` and its clipped score
  (least-favorable-density.md). Maximum entropy over the same set is
  unbounded for continuous `Y`.

Compared with maximum entropy,

  ```
  maximum entropy:            p ∝ exp( ηᵀT )                    Gibbs form
  minimum Fisher information: √p = ground state of −4∂² + ηᵀT    Schrödinger form
  ```

The two agree when `T` is quadratic, where both give the Gaussian, and
differ otherwise: for `T(y) = |y|` maximum entropy gives the Laplace density,
which has a kink, while minimum Fisher information gives the square of an
Airy function, which is smooth. Every smooth positive `p` is the
minimum-Fisher density of some potential, `V = 4 (√p)″ / √p` up to a
constant, so the principle can reproduce any standard density, but only by
reading the potential off the answer; it derives closed forms where the
Schrödinger equation is solvable.

Regularity: `p` absolutely continuous with `√p ∈ H¹(𝒴)` (so `I(p) < ∞`),
boundary terms `u u′` vanishing at the ends of `𝒴`, and `V` bounded below
with `V → ∞` at infinite ends (so the spectrum is discrete and a ground state
exists). `τ` must lie in the range of `η ↦ 𝔼_{u₀²}[T]`. For several
location parameters, `I` becomes the Dirichlet energy `4∫‖∇u‖²` and the
same argument goes through in `ℝᵈ`.
