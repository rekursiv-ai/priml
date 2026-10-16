# Evidence Lower Bound

The ELBO follows from two facts:

1. Bayes' rule, `p(x,z) = p(x) ⋅ p(z|x)`, and,
2. KL is nonnegative, zero iff its arguments agree (gibbs-inequality.md).

Let,

  ```
  p(x,z)  ≝ p(z) ⋅ p(x|z)                     model
  p(x)    ≝ ∫ p(x,z) dz                       evidence, 0 < p(x) < ∞
  p(z|x)  ≝ p(x,z) / p(x)                     posterior
  q(z)    = any density with q ≪ p(z|x)
  ELBO(q) ≝ 𝔼_{q(z)}[ log p(x,z) − log q(z) ]
  ```

First, Bayes' rule gives, for every `z` with `q(z) > 0`,

  ```
  log p(x) = log p(x,z) − log p(z|x)
           = ( log p(x,z) − log q(z) ) + ( log q(z) − log p(z|x) )
  ```

Second, take `𝔼_{q(z)}`. The left side does not depend on `z`,

  ```
  log p(x) = 𝔼_{q(z)}[ log p(x,z) − log q(z) ] + 𝔼_{q(z)}[ log q(z) − log p(z|x) ]
           = ELBO(q) + 𝕂( q(z) ‖ p(z|x) )
  ```

Third, drop the nonnegative KL,

  ```
  log p(x) = ELBO(q) + 𝕂( q(z) ‖ p(z|x) )
           ≥ ELBO(q)
  ```

Hence,

  ```
  log p(x) ≥ ELBO(q) = 𝔼_{q(z)}[ log p(x|z) ] − 𝕂( q(z) ‖ p(z) )
  ```

where the right form splits `log p(x,z) = log p(x|z) + log p(z)`. Equality
holds iff `𝕂( q(z) ‖ p(z|x) ) = 0`, i.e., `q(z) = p(z|x)` almost
everywhere. Since `log p(x)` does not depend on `q`, maximizing the ELBO
over `q` is the same as minimizing `𝕂( q(z) ‖ p(z|x) )`.

Alternatively, via Jensen (jensen.md). Fourth, importance-weight by `q` and
apply Jensen to the concave `log`,

  ```
  log p(x) = log ∫ p(x,z) dz
           = log 𝔼_{q(z)}[ p(x,z) / q(z) ]
           ≥ 𝔼_{q(z)}[ log( p(x,z) / q(z) ) ]
           = ELBO(q)
  ```

where the second step needs `q(z) > 0` wherever `p(x,z) > 0`. By strict
concavity, equality holds iff `p(x,z)/q(z)` is `q`-a.s. constant, i.e.,
`q(z) ∝ p(x,z)`, which is again `q(z) = p(z|x)`.

Regularity: `0 < p(x) < ∞`; `q ≪ p(z|x)` for the decomposition (otherwise
`ELBO = −∞` and the bound holds trivially); `p(z|x) ≪ q` for the
importance-weighting step of the Jensen derivation (otherwise that step
reads `≥` and the bound still holds); and
`𝔼_q[ |log p(x,z) − log q(z)| ] < ∞` so that the split in the second step
is not `∞ − ∞`.
