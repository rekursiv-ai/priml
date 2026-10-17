# Stein's Lemma

Stein's lemma follows from one fact: the standard Gaussian density satisfies
`φ'(z) = −z ⋅ φ(z)`, so `z φ(z)` is a derivative and integration by parts
moves it onto `g`.

Let,

  ```
  Z ~ N(0, 1),   φ(z) ≝ (2π)^{−1/2} ⋅ exp(−z²/2)
  X    ≝ μ + σZ ~ N(μ, σ²),   σ > 0
  g : ℝ → ℝ absolutely continuous, with 𝔼|g'(X)| < ∞
  h(z) ≝ g(μ + σz),   so h'(z) = σ ⋅ g'(μ + σz)
  ```

First, the density identity. Differentiating `φ` and integrating back,

  ```
  φ'(z) = −z ⋅ φ(z)
  φ(u)  = ∫_u^∞ z φ(z) dz        for u ≥ 0
  φ(u)  = −∫_{−∞}^u z φ(z) dz    for u < 0
  ```

where both integrals use `φ(±∞) = 0`.

Second, integrate by parts on `[0, ∞)`. Rather than handle a boundary term,
substitute the density identity and swap the order of integration,

  ```
  ∫_0^∞ h'(u) φ(u) du = ∫_0^∞ h'(u) ∫_u^∞ z φ(z) dz du
                      = ∫_0^∞ z φ(z) ∫_0^z h'(u) du dz
                      = ∫_0^∞ z φ(z) (h(z) − h(0)) dz
  ```

where the second step is Fubini on `{0 ≤ u ≤ z}`, and the third is the
fundamental theorem of calculus for absolutely continuous `h`.

Third, the same on `(−∞, 0)`,

  ```
  ∫_{−∞}^0 h'(u) φ(u) du = −∫_{−∞}^0 h'(u) ∫_{−∞}^u z φ(z) dz du
                         = −∫_{−∞}^0 z φ(z) ∫_z^0 h'(u) du dz
                         = −∫_{−∞}^0 z φ(z) (h(0) − h(z)) dz
                         = ∫_{−∞}^0 z φ(z) (h(z) − h(0)) dz
  ```

where the second step is Fubini on `{z ≤ u ≤ 0}`.

Fourth, add the two halves,

  ```
  𝔼[ h'(Z) ] = ∫ z φ(z) (h(z) − h(0)) dz
             = 𝔼[ Z h(Z) ] − h(0) ⋅ 𝔼[Z]
             = 𝔼[ Z h(Z) ]
  ```

Fubini is valid because the same computation with `|h'|` in place of `h'`
(Tonelli) gives `∫∫ |h'(u)| |z| φ(z) = 𝔼|h'(Z)| < ∞`. This also shows
`𝔼|Z (h(Z) − h(0))| < ∞`, so `𝔼|Z h(Z)| < ∞` follows rather than being
assumed.

Fifth, rescale to `X = μ + σZ`,

  ```
  𝔼[ g(X)(X − μ) ] = σ ⋅ 𝔼[ h(Z) Z ]
                   = σ ⋅ 𝔼[ h'(Z) ]
                   = σ ⋅ 𝔼[ σ g'(X) ]
                   = σ² 𝔼[ g'(X) ]
  ```

Sixth, the multivariate form. Let `X = μ + AW ~ N(μ, Σ)` in `ℝᵈ` with
`AAᵀ = Σ`, `W ~ N(0, I)`, and `g : ℝᵈ → ℝ` with `𝔼‖∇g(X)‖ < ∞`. Let
`k(w) = g(μ + Aw)`, so `∇k(w) = Aᵀ ∇g(μ + Aw)`. Coordinate `Wⱼ` is
independent of the rest, so the scalar lemma applies conditionally on
`W₋ⱼ`, then the tower property removes the conditioning,

  ```
  𝔼[ k(W) Wⱼ ] = 𝔼[ 𝔼[ k(W) Wⱼ | W₋ⱼ ] ]
               = 𝔼[ 𝔼[ ∂ⱼk(W) | W₋ⱼ ] ]
               = 𝔼[ ∂ⱼk(W) ]
  ```

Stacking over `j`,

  ```
  𝔼[ g(X)(X − μ) ] = A ⋅ 𝔼[ k(W) W ]
                   = A ⋅ 𝔼[ ∇k(W) ]
                   = A Aᵀ 𝔼[ ∇g(X) ]
                   = Σ 𝔼[ ∇g(X) ]
  ```

Hence,

  ```
  𝔼[ g(X)(X − μ) ] = σ² 𝔼[ g'(X) ]     X ~ N(μ, σ²)
  𝔼[ g(X)(X − μ) ] = Σ 𝔼[ ∇g(X) ]      X ~ N(μ, Σ)
  ```

The identity characterizes the Gaussian: if it holds for all bounded `C¹`
`g` with bounded `g'`, then `X ~ N(μ, σ²)` (Stein's characterization, cited,
not proved). Examples: `g(x) = x` gives `𝕍(X) = σ²`, and `g(x) = x²` gives
`𝔼[X²(X − μ)] = 2σ²μ`. The multivariate form also holds for singular `Σ`.

Regularity: the proof needs `X` Gaussian, `g` absolutely continuous (along
each coordinate of `w` a.e., for the multivariate form), and `𝔼|g'(X)| < ∞`
(or `𝔼‖∇g(X)‖ < ∞`), which justifies Fubini.
