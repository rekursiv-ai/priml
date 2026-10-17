# Bregman Divergences Are Minimized by the Mean

The mean minimizes every expected Bregman divergence. This follows from one
fact: `D_φ(y, c)` is affine in `y` apart from `φ(y)`, so averaging over `Y`
splits off an exact remainder `D_φ(𝔼Y, c)`.

Let,

  ```
  φ : ℝᵈ → ℝ   strictly convex, differentiable
  D_φ(y, c) ≝ φ(y) − φ(c) − ⟨∇φ(c), y − c⟩
  μ         ≝ 𝔼[Y]
  ```

First, `D_φ ≥ 0`, with equality iff `y = c`. This is the first-order
characterization of strict convexity: `φ(y) ≥ φ(c) + ⟨∇φ(c), y − c⟩`, with
equality only at `y = c`.

Second, expand the gap between predicting `c` and predicting `μ`,

  ```
  𝔼[D_φ(Y, c)] − 𝔼[D_φ(Y, μ)] = 𝔼[ φ(Y) − φ(c) − ⟨∇φ(c), Y − c⟩ ]
                                  − 𝔼[ φ(Y) − φ(μ) − ⟨∇φ(μ), Y − μ⟩ ]
                              = −φ(c) − ⟨∇φ(c), μ − c⟩ + φ(μ) + ⟨∇φ(μ), μ − μ⟩
                              = φ(μ) − φ(c) − ⟨∇φ(c), μ − c⟩
                              = D_φ(μ, c)
  ```

where the second step cancels `𝔼[φ(Y)]` and uses linearity,
`𝔼⟨a, Y − c⟩ = ⟨a, μ − c⟩` for fixed `a`.

Third, condition on `X`. Let `m(X) ≝ 𝔼[Y|X]` and `f` be any predictor.
Given `X`, `f(X)` is a fixed point. So the second step applies under
`p(Y|X)`, and the tower property finishes,

  ```
  𝔼[D_φ(Y, f(X))] = 𝔼[ 𝔼[ D_φ(Y, f(X)) | X ] ]
                  = 𝔼[ 𝔼[ D_φ(Y, m(X)) | X ] + D_φ(m(X), f(X)) ]
                  = 𝔼[D_φ(Y, m(X))] + 𝔼[D_φ(m(X), f(X))]
  ```

Hence,

  ```
  𝔼[D_φ(Y, c)]             = 𝔼[D_φ(Y, 𝔼Y)] + D_φ(𝔼Y, c)
  argmin_c 𝔼[D_φ(Y, c)]    = 𝔼[Y]
  argmin_f 𝔼[D_φ(Y, f(X))] = 𝔼[Y|X]   almost surely
  ```

The minimizer is unique since `D_φ(μ, c) = 0` iff `c = μ` (first step). The
excess risk is exactly `D_φ(μ, c)`, a Pythagorean identity.

Examples:

  ```
  φ(y) = ‖y‖²             D_φ(y, c) = ‖y − c‖²
  φ(y) = Σₖ yₖ log yₖ     D_φ(y, c) = Σₖ yₖ log(yₖ/cₖ) − Σₖ yₖ + Σₖ cₖ
  φ(y) = −log y  (y > 0)  D_φ(y, c) = y/c − log(y/c) − 1
  ```

The second, on the simplex, is `𝕂(y‖c)`. The third is the Itakura–Saito
distance. In each, the mean is optimal for the second argument. The order
matters: `argmin_c 𝔼[D_φ(c, Y)]` is generally not `𝔼[Y]`.

Fine print: the converse is cited, not proved (Banerjee, Guo, Wang 2005).
Take a nonnegative loss `F(y, c)` with `F(y, y) = 0` and continuous second
derivatives. If `argmin_c 𝔼[F(Y, c)] = 𝔼[Y]` for every distribution of `Y`,
then `F = D_φ` for some strictly convex `φ`. So "the mean is the optimal
point predictor" characterizes Bregman losses. When `φ` is defined only on
an open convex set `U` (as for `−log`, or entropy on the open simplex), `Y`
must lie in `U`. Then `μ ∈ U` too, since the mean of a law on a convex set
lies in it. proper-scoring.md applies this identity to the one-hot outcome
`δ_Y` in place of `Y`.

Regularity: the proof needs `φ` strictly convex and differentiable on a
convex set containing `Y` and `μ`, `𝔼‖Y‖ < ∞`, and `𝔼|φ(Y)| < ∞`, so that
every expectation above is finite. For the conditional form,
`𝔼[D_φ(Y, f(X))] < ∞` for the competitor `f`.
