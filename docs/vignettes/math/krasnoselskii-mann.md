# Krasnoselskii–Mann

Krasnoselskii–Mann follows from one identity for convex combinations in a
Hilbert space,

  ```
  ‖(1 − α) a + α b‖² = (1 − α) ‖a‖² + α ‖b‖² − α(1 − α) ‖a − b‖²
  ```

which turns nonexpansiveness into a strict decrease of distance to any fixed
point, paid for by the residual.

Let,

  ```
  H Hilbert space
  T : H → H,   ‖Tx − Ty‖ ≤ ‖x − y‖,   Fix T ≝ { x : Tx = x } ≠ ∅
  α ∈ (0, 1),   xₖ₊₁ ≝ (1 − α) xₖ + α T xₖ
  rₖ ≝ xₖ − T xₖ
  x* ∈ Fix T
  ```

The identity holds by expanding both sides,

  ```
  (1 − α)‖a‖² + α‖b‖² − α(1 − α)(‖a‖² − 2⟨a,b⟩ + ‖b‖²)
    = (1 − α)²‖a‖² + 2α(1 − α)⟨a,b⟩ + α²‖b‖²
    = ‖(1 − α) a + α b‖²
  ```

First, Fejér monotonicity. Put `a ≝ xₖ − x*` and `b ≝ T xₖ − x*`. Since
`x* = T x*`, `xₖ₊₁ − x* = (1 − α) a + α b` and `a − b = rₖ`,

  ```
  ‖xₖ₊₁ − x*‖² = (1 − α)‖xₖ − x*‖² + α‖T xₖ − T x*‖² − α(1 − α)‖rₖ‖²
               ≤ (1 − α)‖xₖ − x*‖² + α‖xₖ − x*‖² − α(1 − α)‖rₖ‖²
               = ‖xₖ − x*‖² − α(1 − α)‖rₖ‖²
  ```

where the second step uses nonexpansiveness. So `‖xₖ − x*‖` is
nonincreasing for every `x* ∈ Fix T`.

Second, the residuals are square-summable. Telescope the first step over
`k = 0, …, K`,

  ```
  α(1 − α) Σ_{k=0}^{K} ‖rₖ‖² ≤ ‖x₀ − x*‖² − ‖x_{K+1} − x*‖²
                             ≤ ‖x₀ − x*‖²
  ```

Third, the residual is nonincreasing. Since `xₖ₊₁ − xₖ = −α rₖ`,

  ```
  rₖ₊₁   = (1 − α) xₖ + α T xₖ − T xₖ₊₁
         = (1 − α) rₖ + (T xₖ − T xₖ₊₁)
  ‖rₖ₊₁‖ ≤ (1 − α)‖rₖ‖ + ‖xₖ − xₖ₊₁‖
         = (1 − α)‖rₖ‖ + α‖rₖ‖
         = ‖rₖ‖
  ```

by the triangle inequality and nonexpansiveness.

Fourth, the rate. The last term of a nonincreasing sum is its smallest,

  ```
  (K + 1)‖r_K‖² ≤ Σ_{k=0}^{K} ‖rₖ‖²
                ≤ ‖x₀ − x*‖² / (α(1 − α))
  ```

For `o(1/√k)`: the block `j = ⌈k/2⌉, …, k` has at least `k/2` terms, each
at least `‖rₖ‖²`, so

  ```
  (k/2) ‖rₖ‖² ≤ Σ_{j=⌈k/2⌉}^{k} ‖rⱼ‖²
              → 0
  ```

as the tail of a convergent series. So `k ‖rₖ‖² → 0`.

Hence,

  ```
  ‖xₖ − T xₖ‖² ≤ ‖x₀ − x*‖² / (α(1 − α)(k + 1))
  ‖xₖ − T xₖ‖  = o(1/√k)
  ```

The constant is best at `α = ½`. The Fejér step is tight when `T` is an
isometry fixing `x*`, e.g., `T = −I`, `x* = 0`, where
`‖xₖ₊₁‖² = (1 − 2α)²‖xₖ‖²` exactly. That example also shows why `α < 1` is
needed: with `α = 1`, `xₖ = (−1)ᵏ x₀` never converges.

Weak convergence (sketch; one fact cited). The cited fact: bounded
sequences in a Hilbert space have weakly convergent subsequences. The
iterates are bounded by Fejér monotonicity.

- Cluster points are fixed (demiclosedness). Let `x_{kⱼ} ⇀ x̄`. Since
  `‖x_{kⱼ} − T x̄‖ ≤ ‖r_{kⱼ}‖ + ‖x_{kⱼ} − x̄‖` and `rₖ → 0`, expanding gives
  `limsup ‖x_{kⱼ} − x̄‖² + ‖x̄ − T x̄‖² ≤ limsup ‖x_{kⱼ} − x̄‖²`, because the
  cross term `2⟨x_{kⱼ} − x̄, x̄ − T x̄⟩ → 0`. So `T x̄ = x̄`.
- The cluster point is unique (Opial). For `z ∈ Fix T`, `‖xₖ − z‖`
  converges by Fejér. For two cluster points `z₁, z₂ ∈ Fix T`,
  `‖xₖ − z₁‖² − ‖xₖ − z₂‖² = 2⟨xₖ, z₂ − z₁⟩ + ‖z₁‖² − ‖z₂‖²` converges.
  Evaluating its limit along subsequences converging weakly to `z₁` and to
  `z₂` and subtracting gives `‖z₁ − z₂‖² = 0`.

So `xₖ ⇀ x̄ ∈ Fix T`. Strong convergence can fail in infinite dimensions;
in `ℝⁿ` weak and strong coincide.

Regularity: the proof needs `H` a Hilbert space (for the identity and weak
compactness), `T` nonexpansive on all of `H`, `Fix T ≠ ∅`, and a constant
`α ∈ (0, 1)`.
