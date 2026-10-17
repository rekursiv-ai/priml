# Change of Variables

The density transform follows from one fact: `Y = g(X)` lands in `B` iff `X`
lands in `g⁻¹(B)`, and integration by substitution rewrites the integral over
`g⁻¹(B)` as one over `B`.

Let,

  ```
  X       a random vector on open U ⊆ ℝᵈ with density p_X
  g       U → V a bijection, C¹ with C¹ inverse (a diffeomorphism)
  Y          ≝ g(X)
  J_{g⁻¹}(y) ≝ ∂g⁻¹/∂y,   the d × d Jacobian of the inverse
  ```

First, the 1-D case via the CDF. Let `d = 1`, `U, V` intervals, and `g`
strictly increasing, so `(g⁻¹)' > 0`. For `y ∈ V`,

  ```
  F_Y(y) = P( g(X) ≤ y )
         = P( X ≤ g⁻¹(y) )
         = ∫_{inf U}^{g⁻¹(y)} p_X(x) dx
         = ∫_{inf V}^{y} p_X(g⁻¹(u)) ⋅ (g⁻¹)'(u) du
  ```

where the second step uses that `g` is increasing, and the last substitutes
`x = g⁻¹(u)`. A function whose integral up to `y` is the CDF is a density, so
`p_Y(y) = p_X(g⁻¹(y)) ⋅ (g⁻¹)'(y)`.

Second, the decreasing 1-D case. If `g` is strictly decreasing, then
`(g⁻¹)' < 0` and,

  ```
  F_Y(y) = P( X ≥ g⁻¹(y) )
         = ∫_{g⁻¹(y)}^{sup U} p_X(x) dx
         = ∫_{y}^{inf V} p_X(g⁻¹(u)) ⋅ (g⁻¹)'(u) du
         = ∫_{inf V}^{y} p_X(g⁻¹(u)) ⋅ |(g⁻¹)'(u)| du
  ```

where the third step substitutes `x = g⁻¹(u)` (which maps `u = inf V` to
`x = sup U`), and the last flips the limits and the sign. Both cases give
`p_Y(y) = p_X(g⁻¹(y)) ⋅ |(g⁻¹)'(y)|`.

Third, the multi-D case. For any Borel `B ⊆ V`,

  ```
  P( Y ∈ B ) = P( X ∈ g⁻¹(B) )
             = ∫_{g⁻¹(B)} p_X(x) dx
             = ∫_{B} p_X(g⁻¹(y)) ⋅ |det J_{g⁻¹}(y)| dy
  ```

where the last step is the multivariate substitution theorem with
`x = g⁻¹(y)` (cited, not proved). Since this holds for every `B`, the
integrand is a density of `Y`.

Fourth, invert the Jacobian. Differentiating `g(g⁻¹(y)) = y` gives
`J_g(x) J_{g⁻¹}(y) = I` at `x = g⁻¹(y)`, so,

  ```
  |det J_{g⁻¹}(y)| = |det J_g(x)⁻¹|
                   = 1 / |det J_g(x)|
  ```

Hence,

  ```
  p_Y(y)     = p_X(g⁻¹(y)) ⋅ |det ∂g⁻¹/∂y|
             = p_X(x) / |det ∂g/∂x|,              x = g⁻¹(y)
  log p_Y(y) = log p_X(x) − log |det ∂g/∂x|
  ```

Normalizing flows compose `g = g_K ∘ … ∘ g_1`. With `x₀ = x` and
`xₖ = gₖ(xₖ₋₁)`, the chain rule and `det(AB) = det A ⋅ det B` give,

  ```
  log p_Y(x_K) = log p_X(x₀) − Σₖ₌₁ᴷ log |det J_{gₖ}(xₖ₋₁)|
  ```

Flows choose `gₖ` with triangular Jacobians, whose log-det is the sum of
the log-absolute diagonal entries, an `O(d)` cost instead of `O(d³)`.

If `g` is not injective, the formula sums over preimages:
`p_Y(y) = Σ_{x : g(x) = y} p_X(x) / |det J_g(x)|`, valid where `g` is
locally a diffeomorphism.

Regularity: the proof needs `X` to have a density, `g` a bijection between
open sets with `g` and `g⁻¹` continuously differentiable (so
`det J_g ≠ 0`), and, in 1-D, `g` strictly monotone.
