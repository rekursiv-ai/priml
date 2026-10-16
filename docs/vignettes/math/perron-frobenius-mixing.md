# Perron–Frobenius Mixing

A reversible chain mixes at rate `λ*ᵗ`. This follows from three facts:

1. irreducibility forces harmonic functions to be constant (maximum
   principle), so `π` is unique and the eigenvalue `1` is simple,
2. reversibility makes `P` self-adjoint in `L²(π)`, so it has an orthonormal
   eigenbasis, and,
3. total variation is bounded by the `L²(π)` distance, which decays as `λ*ᵗ`.

Let,

  ```
  Ω finite,  |Ω| = n,  P a stochastic matrix on Ω
  irreducible:  ∀ x, y  ∃ t  Pᵗ(x, y) > 0
  aperiodic:    gcd{ t : Pᵗ(x, x) > 0 } = 1  for all x
  πP       = π,  π ≥ 0,  Σₓ π(x) = 1
  reversible:   π(x) P(x, y) = π(y) P(y, x)
  ⟨f, g⟩_π ≝ Σₓ π(x) f(x) g(x)
  ```

where a stationary distribution `π` exists (see Fine print).

First, harmonic functions are constant. Let `Ph = h` and let `x*` maximize
`h`. Then

  ```
  0 = h(x*) − (Ph)(x*)
    = Σ_y P(x*, y) (h(x*) − h(y))
  ```

Every term is nonnegative, so `h(y) = h(x*)` whenever `P(x*, y) > 0`. Each
such `y` is again a maximizer, so the argument repeats along every path from
`x*`. By irreducibility every state is reached, so `h` is constant.

Second, `π` is unique and positive. The first step gives
`dim ker(P − I) = 1`. Row and column rank agree, so
`dim ker(Pᵀ − I) = 1`: every stationary vector is a multiple of `π`, and
normalization fixes the multiple. Positivity: some
`π(x) > 0`, and for any `y` pick `t` with `Pᵗ(x, y) > 0`,

  ```
  π(y) = Σ_z π(z) Pᵗ(z, y)
       ≥ π(x) Pᵗ(x, y)
       > 0
  ```

Third, the spectral decomposition. Let `D ≝ diag(π)` and
`S = D^{1/2} P D^{−1/2}`. Reversibility makes `S` symmetric,

  ```
  S(x, y) = √π(x) P(x, y) / √π(y)
          = π(x) P(x, y) / √(π(x) π(y))
          = π(y) P(y, x) / √(π(x) π(y))
          = S(y, x)
  ```

By spectral-theorem.md, `S = Σᵢ λᵢ φᵢ φᵢᵀ` with orthonormal `φᵢ`. Set
`fᵢ = D^{−1/2} φᵢ`. Then `Pfᵢ = D^{−1/2} S φᵢ = λᵢ fᵢ` and
`⟨fᵢ, fⱼ⟩_π = φᵢᵀφⱼ = δᵢⱼ`. Since `P𝟙 = 𝟙` and `‖𝟙‖_π = 1`, take `f₁ = 𝟙`
with `λ₁ = 1`. By the first step no other `fᵢ` has eigenvalue `1`.

Fourth, the spectral gap. The `λᵢ` are real since `S` is symmetric, and
`|Pf(x)| ≤ maxᵧ |f(y)|` gives `|λᵢ| ≤ 1`. It remains to exclude `−1`.
Suppose `Ph = −h` with `h ≠ 0`. Let `x*` maximize `|h|`, with
`h(x*) = M > 0` after a sign flip. Aperiodicity gives an odd `t` with
`Pᵗ(x*, x*) > 0`; otherwise all return times are even and their gcd is at
least 2. Then

  ```
  −M = (Pᵗh)(x*)
     = Σ_y Pᵗ(x*, y) h(y)
  ```

is an average of values `h(y) ≥ −M`, so `h(y) = −M` wherever
`Pᵗ(x*, y) > 0`. Taking `y = x*` gives `M = −M`, a contradiction. Hence

  ```
  λ* ≝ max_{i≥2} |λᵢ| < 1
  ```

Fifth, expand the transition probabilities. Let `gᵧ ≝ 1[· = y] / π(y)`, so
`⟨gᵧ, fᵢ⟩_π = fᵢ(y)` and `gᵧ = Σᵢ fᵢ(y) fᵢ`. Applying `Pᵗ`,

  ```
  Pᵗ(x, y) / π(y) = (Pᵗ gᵧ)(x)
                  = Σᵢ λᵢᵗ fᵢ(x) fᵢ(y)
                  = 1 + Σ_{i≥2} λᵢᵗ fᵢ(x) fᵢ(y)
  ```

Sixth, the `L²(π)` distance. Let `hₜ(y) ≝ Pᵗ(x, y)/π(y) − 1`. By
orthonormality of the `fᵢ` in `L²(π)`,

  ```
  ‖hₜ‖²_π = Σ_{i≥2} λᵢ^{2t} fᵢ(x)²
          ≤ λ*^{2t} Σ_{i≥2} fᵢ(x)²
          = λ*^{2t} (‖gₓ‖²_π − f₁(x)²)
          = λ*^{2t} (1/π(x) − 1)
  ```

where the third step is Parseval for `gₓ`, and the last uses
`‖gₓ‖²_π = π(x) / π(x)² = 1/π(x)` and `f₁ = 𝟙`.

Seventh, total variation. By Cauchy–Schwarz under `π`,

  ```
  ‖Pᵗ(x, ·) − π‖_TV = ½ Σ_y |Pᵗ(x, y) − π(y)|
                    = ½ Σ_y π(y) |hₜ(y)|
                    ≤ ½ (Σ_y π(y) hₜ(y)²)^{1/2}
                    = ½ ‖hₜ‖_π
  ```

Hence,

  ```
  ‖Pᵗ(x, ·) − π‖_TV ≤ ½ √((1 − π(x)) / π(x)) ⋅ λ*ᵗ
  ```

The rate is sharp: if `fᵢ(x) ≠ 0` for some `i` with `|λᵢ| = λ*`, then
`‖hₜ‖_π` decays exactly like `λ*ᵗ`, and `‖·‖_TV ≥ ½ |⟨hₜ, fᵢ⟩_π| / max|fᵢ|`
decays at the same rate.

Fine print: existence of `π` is a Cesàro limit. The averages
`μ_T ≝ (1/T) Σ_{s<T} δₓPˢ` lie in the compact simplex, and any limit point is
stationary since `μ_T P − μ_T = (δₓP^T − δₓ)/T → 0`. Without reversibility,
`Pᵗ(x, ·) → π` still holds for irreducible aperiodic chains by the
Perron–Frobenius theorem (or coupling), at a rate set by the second-largest
eigenvalue modulus, but the clean constant above needs self-adjointness.
Uniqueness of `π` (first and second steps) uses irreducibility only;
aperiodicity is needed only for `λ* < 1`. A lazy chain `(I + P)/2` has
`λᵢ ≥ 0`, so `λ* = λ₂`.

Regularity: the proof needs `Ω` finite, `P` stochastic, irreducible,
aperiodic, and reversible with respect to `π`.
