# Mutual Information

The identities for mutual information follow from two facts:

1. `p(x,y) / ( p(x) p(y) ) = p(x|y) / p(x)`, and,
2. KL is nonnegative, zero iff its arguments agree (gibbs-inequality.md).

Let `X, Y` be discrete with joint `p(x,y)` and marginals `p(x)`, `p(y)`, and,

  ```
  I(X;Y) ≝ 𝕂( p(x,y) ‖ p(x)p(y) ) = Σ_{x,y} p(x,y) ⋅ log( p(x,y) / ( p(x)p(y) ) )
  H(X)   ≝ −Σₓ p(x) ⋅ log p(x)
  H(X|Y) ≝ −Σ_{x,y} p(x,y) ⋅ log p(x|y)
  ```

with sums over pairs where `p(x,y) > 0`.

First, rewrite the log-ratio with `p(x,y) = p(y) p(x|y)` and split the sum,

  ```
  I(X;Y) = Σ_{x,y} p(x,y) ⋅ log( p(y) p(x|y) / ( p(x) p(y) ) )
         = Σ_{x,y} p(x,y) ⋅ log( p(x|y) / p(x) )
         = −Σ_{x,y} p(x,y) ⋅ log p(x) + Σ_{x,y} p(x,y) ⋅ log p(x|y)
         = −Σₓ p(x) ⋅ log p(x) − H(X|Y)
         = H(X) − H(X|Y)
  ```

where the fourth step marginalizes `Σ_y p(x,y) = p(x)`. By symmetry,
`I(X;Y) = H(Y) − H(Y|X)`.

Second, nonnegativity is Gibbs applied to the definition,

  ```
  I(X;Y) = 𝕂( p(x,y) ‖ p(x)p(y) )
         ≥ 0
  ```

Third, `I(X;Y) = 0` iff `p(x,y) = p(x)p(y)` for all `x, y`, by the equality
case of Gibbs. That is the definition of independence.

Hence,

  ```
  I(X;Y) = 𝕂( p(x,y) ‖ p(x)p(y) ) = H(X) − H(X|Y) ≥ 0
  ```

with equality iff `X` and `Y` are independent. In words, conditioning never
increases entropy on average: `H(X|Y) ≤ H(X)`.

For general (e.g., continuous) `X, Y`, the KL definition and the first step
carry over with densities in place of mass functions. `H` then becomes
differential entropy, and `H(X) − H(X|Y)` is valid whenever both terms are
finite. The KL form is always well defined in `[0, ∞]`.

Regularity: discrete `X`, `Y` with `H(X) < ∞`, so that the split in the
first step is not `∞ − ∞`. Nonnegativity and the independence condition
need no finiteness.
