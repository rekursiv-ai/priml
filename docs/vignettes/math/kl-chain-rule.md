# KL Chain Rule

The KL chain rule follows from one fact: a joint density factors as a
marginal times a conditional, and `log` turns that product into a sum.

Let,

  ```
  p(x,y)           = p(x) ⋅ p(y|x)
  q(x,y)           = q(x) ⋅ q(y|x)
  𝕂(p(y|x)‖q(y|x)) ≝ ∫ p(y|x) ⋅ log( p(y|x) / q(y|x) ) dy    (for fixed x)
  ```

First, the log-ratio of the joints splits pointwise wherever `p(x,y) > 0`,

  ```
  log( p(x,y) / q(x,y) ) = log( p(x) p(y|x) / ( q(x) q(y|x) ) )
                         = log( p(x) / q(x) ) + log( p(y|x) / q(y|x) )
  ```

Second, take `𝔼_{p(x,y)}` of both sides and integrate each term,

  ```
  𝕂(p(x,y)‖q(x,y)) = 𝔼_{p(x,y)}[ log( p(X) / q(X) ) ] + 𝔼_{p(x,y)}[ log( p(Y|X) / q(Y|X) ) ]
                   = ∫ p(x) ⋅ log( p(x) / q(x) ) ⋅ ( ∫ p(y|x) dy ) dx
                      + ∫ p(x) ⋅ ( ∫ p(y|x) ⋅ log( p(y|x) / q(y|x) ) dy ) dx
                   = ∫ p(x) ⋅ log( p(x) / q(x) ) dx + ∫ p(x) ⋅ 𝕂(p(y|x)‖q(y|x)) dx
                   = 𝕂(p(x)‖q(x)) + 𝔼_{p(x)}[ 𝕂(p(y|x)‖q(y|x)) ]
  ```

where the second step uses Tonelli/Fubini to write `p(x,y) = p(x) p(y|x)` and
iterate the integral, and the third uses `∫ p(y|x) dy = 1`.

Hence,

  ```
  𝕂(p(x,y)‖q(x,y)) = 𝕂(p(x)‖q(x)) + 𝔼_{p(x)}[ 𝕂(p(y|x)‖q(y|x)) ]
  ```

By symmetry of the argument, the same holds with the roles of `x` and `y`
swapped. Since each term is nonnegative by Gibbs (gibbs-inequality.md), the
chain rule gives `𝕂(p(x,y)‖q(x,y)) ≥ 𝕂(p(x)‖q(x))`, with equality iff
`p(y|x) = q(y|x)` for `p(x)`-almost every `x`.

Regularity: the identity holds in `[0, ∞]`, including when either side is
`∞`. Splitting the expectation in the second step is valid because each
piece has an integrable negative part: by `log u ≤ u − 1`,
`p(x) ⋅ log( q(x)/p(x) )⁺ ≤ q(x)` and
`p(x,y) ⋅ log( q(y|x)/p(y|x) )⁺ ≤ p(x) q(y|x)`, both integrable. Densities
are w.r.t. a product base measure, and conditionals exist as regular
conditional densities.
