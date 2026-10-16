# Universal Approximation

Universal approximation by one-hidden-layer ReLU networks follows from three
facts:

1. a continuous `f` on `[0, 1]` is uniformly close to its piecewise-linear
   interpolant on a fine grid,
2. a piecewise-linear function is exactly a sum of shifted ReLUs, one per
   kink, and,
3. sums of exponentials `exp(wᵀx)` are dense in `C([0, 1]ᵈ)`
   (Stone–Weierstrass, cited), and each is a function of the scalar `wᵀx`.

Let,

  ```
  ReLU(u) ≝ max(u, 0)
  f ∈ C([0, 1]),  ε > 0
  δ       > 0 with |s − t| ≤ δ ⇒ |f(s) − f(t)| ≤ ε   uniform continuity
  N       ≥ 1/δ,   tₖ ≝ k/N,  k = 0..N
  sₖ      ≝ N (f(t_{k+1}) − f(tₖ))                   slope on [tₖ, t_{k+1}]
  ```

First, interpolate. Let `g` be the piecewise-linear interpolant with
`g(tₖ) = f(tₖ)`. For `x ∈ [tₖ, t_{k+1}]`, `g(x) = (1 − θ) f(tₖ) + θ f(t_{k+1})`
with `θ ∈ [0, 1]`, and both grid points are within `1/N ≤ δ` of `x`,

  ```
  |g(x) − f(x)| = |(1 − θ)(f(tₖ) − f(x)) + θ (f(t_{k+1}) − f(x))|
                ≤ (1 − θ) ε + θ ε
                = ε
  ```

Second, write `g` as a ReLU network. Set `c₀ ≝ s₀` and `cₖ ≝ sₖ − s_{k−1}` for
`k ≥ 1`, and

  ```
  ĝ(x) ≝ f(0) + Σ_{k=0}^{N−1} cₖ ⋅ ReLU(x − tₖ)
  ```

On `[tₖ, t_{k+1}]`, `ReLU(x − tⱼ) = x − tⱼ` for `j ≤ k` and `0` for `j > k`, so

  ```
  ĝ′(x) = Σ_{j=0}^{k} cⱼ
        = s₀ + Σ_{j=1}^{k} (sⱼ − s_{j−1})
        = sₖ
  ```

`ĝ` and `g` are continuous, piecewise linear on the same grid, have the same
slope on every piece, and agree at `0`. So `ĝ = g` on `[0, 1]`.

Hence, on `[0, 1]`, a network with `N` hidden ReLUs and an output bias
satisfies,

  ```
  sup_{x ∈ [0,1]} |f(x) − ĝ(x)| ≤ ε
  ```

The same construction works on any `[a, b]` with `tₖ = a + k(b − a)/N`.

Third, extend to `[0, 1]ᵈ`. The span of `{exp(wᵀx) : w ∈ ℝᵈ}` is an algebra,
since `exp(wᵀx) exp(vᵀx) = exp((w + v)ᵀx)`. It contains constants (`w = 0`)
and separates points (`w = x − x′` gives `wᵀx ≠ wᵀx′` for `x ≠ x′`).
Stone–Weierstrass (cited) gives `p(x) = Σ_{m=1}^{M} aₘ exp(wₘᵀx)` with
`sup |f − p| ≤ ε/2`. On `[0, 1]ᵈ`, `wₘᵀx` lies in
`Iₘ ≝ [−‖wₘ‖₁, ‖wₘ‖₁]`. With `A ≝ Σₘ |aₘ|`, the one-dimensional result on
`Iₘ` gives a ReLU network `gₘ` with `sup_{Iₘ} |exp − gₘ| ≤ ε/(2A)`. Then,

  ```
  |f(x) − Σₘ aₘ gₘ(wₘᵀx)| ≤ |f(x) − p(x)| + Σₘ |aₘ| ⋅ |exp(wₘᵀx) − gₘ(wₘᵀx)|
                          ≤ ε/2 + A ⋅ ε/(2A)
                          = ε
  ```

and `Σₘ aₘ gₘ(wₘᵀx) = b + Σ_{m,k} aₘ cₘₖ ReLU(wₘᵀx − τₘₖ)` is again a
one-hidden-layer ReLU network.

Hence, one-hidden-layer ReLU networks are dense in `C([0, 1]ᵈ)` under the
sup norm.

The width is not tight in general. For `L`-Lipschitz `f` on `[0, 1]`,
`δ = ε/L` gives `N = ⌈L/ε⌉` units. In `d` dimensions the construction says
nothing about rates; widths can grow like `ε^{−d}`.

Fine print: the only cited step is Stone–Weierstrass, for the extension to
`d > 1`; the one-dimensional case is proved in full. The output bias `f(0)`
can be absorbed into a hidden unit, since `f(0) = f(0) ⋅ ReLU(x + 1)` minus
`f(0) ⋅ ReLU(x)` on `[0, 1]`. For a sigmoid `σ` with limits `0` and `1`, the
same 1-D argument works with steps: `σ(w(x − tₖ))` approaches
`1[x > tₖ]` uniformly away from `tₖ` as `w → ∞`. The step sum approximates
the piecewise-constant interpolant, and continuity of `σ` controls the
window near each `tₖ`. The general activation results are cited, not
proved. Cybenko (1989) shows that a continuous sigmoidal `σ` gives a dense
span: Hahn–Banach and Riesz representation reduce density to "no nonzero
signed measure annihilates every `σ(wᵀx + b)`". Leshno et al. (1993) show
that for a locally bounded, a.e.-continuous `σ`, density holds iff `σ` is
not a polynomial.

Regularity: the proof needs `f` continuous on a compact set (`[0, 1]` or
`[0, 1]ᵈ`), which gives uniform continuity and the Stone–Weierstrass
hypotheses, and an output bias (or the two-unit replacement above). The
approximation is uniform on the compact set only, not on `ℝᵈ`.
