# Gaussian Concentration for Lipschitz Functions

The Maurey–Pisier bound follows from three facts:

1. interpolating between two independent Gaussians along a quarter circle
   keeps the pair `(point, velocity)` independent standard Gaussian,
2. Jensen moves the exponential inside the path integral, and,
3. a linear form `⟨v, Y⟩` with `Y ~ N(0, Iₙ)` has MGF `e^{λ²‖v‖²/2}`.

Let,

  ```
  X, Y ~ N(0, Iₙ)                    independent
  f : ℝⁿ → ℝ,  C¹,  ‖∇f‖ ≤ L         (so f is L-Lipschitz)
  X_θ  ≝ X sin θ + Y cos θ
  X'_θ ≝ X cos θ − Y sin θ           = d X_θ / dθ
  λ    > 0,   t > 0
  ```

First, symmetrize. Jensen (`jensen.md`) on `Y` gives
`𝔼_Y[e^{−λf(Y)}] ≥ e^{−λ𝔼f}`, so,

  ```
  𝔼[ e^{λ(f(X) − 𝔼f)} ] ≤ 𝔼[ e^{λf(X)} ] ⋅ 𝔼[ e^{−λf(Y)} ]
                        = 𝔼[ e^{λ(f(X) − f(Y))} ]
  ```

Second, write the gap as a path integral. `X_{π/2} = X` and `X₀ = Y`, so by
the fundamental theorem of calculus,

  ```
  f(X) − f(Y) = ∫₀^{π/2} (d/dθ) f(X_θ) dθ
              = ∫₀^{π/2} ⟨∇f(X_θ), X'_θ⟩ dθ
              = 𝔼_θ[ (π/2) ⟨∇f(X_θ), X'_θ⟩ ]
  ```

where `θ` is uniform on `[0, π/2]`, independent of `X, Y`.

Third, apply Jensen to the convex `e^{λ⋅}` over `θ`, then swap expectations,

  ```
  𝔼[ e^{λ(f(X) − f(Y))} ] ≤ 𝔼_θ 𝔼[ exp( λ(π/2) ⟨∇f(X_θ), X'_θ⟩ ) ]
                          = 𝔼[ exp( λ(π/2) ⟨∇f(X), Y⟩ ) ]
  ```

where the second step uses that `(X_θ, X'_θ)` is the orthogonal map
`⎡sin θ  cos θ; cos θ  −sin θ⎤` applied to `(X, Y)` in each coordinate, so
it has the same law `N(0, I₂ₙ)` for every `θ`.

Fourth, integrate out `Y` with `X` fixed. `⟨∇f(X), Y⟩ ~ N(0, ‖∇f(X)‖²)`,

  ```
  𝔼[ exp( λ(π/2) ⟨∇f(X), Y⟩ ) ] = 𝔼_X[ exp( λ²π² ‖∇f(X)‖² / 8 ) ]
                                ≤ exp( λ²π²L² / 8 )
  ```

Fifth, Chernoff (`markov-chebyshev-chernoff.md`) at `λ = 4t / (π²L²)`,

  ```
  P( f(X) − 𝔼f ≥ t ) ≤ exp( −λt + λ²π²L²/8 )
                     = exp( −4t²/(π²L²) + 2t²/(π²L²) )
                     = exp( −2t² / (π²L²) )
  ```

Hence,

  ```
  P( f(X) − 𝔼f(X) ≥ t ) ≤ exp( −2t² / (π²L²) )
  ```

The rate is dimension-free. The constant is not sharp: `f(x) = L x₁` has
tail `P(LX₁ ≥ t) ≈ e^{−t²/(2L²)}`, and `2/π² ≈ 0.20 < 1/2`.

Fine print: the sharp bound `P(f(X) − 𝔼f ≥ t) ≤ e^{−t²/(2L²)}` is due to
Tsirelson–Ibragimov–Sudakov (1976), and also follows from the Gaussian
log-Sobolev inequality via the Herbst argument (cited, not proved). For a
merely Lipschitz `f`, smooth it: `f_δ ≝ 𝔼[f(x + δZ)]` is `C^∞`, `L`-Lipschitz,
and `|f_δ − f| ≤ Lδ𝔼‖Z‖`, so the bound passes to `f` as `δ → 0`. The same
Jensen step with any convex `φ` gives the Pisier comparison
`𝔼φ(f(X) − f(Y)) ≤ 𝔼φ((π/2)⟨∇f(X), Y⟩)`. Applying the result to `−f` gives
the lower tail.

Regularity: the proof needs `X` standard Gaussian (rotation invariance of
`(X, Y)` and the Gaussian MGF), `f` with `‖∇f‖ ≤ L` everywhere (or
Lipschitz, via smoothing), and `𝔼|f(X)| < ∞`, which Lipschitz implies.
Fubini justifies swapping `𝔼_θ` and `𝔼` for the nonnegative integrand.
