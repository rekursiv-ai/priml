# Data Processing Inequality

The data processing inequality follows from two facts:

1. the KL chain rule (kl-chain-rule.md) expands a joint KL two ways, and,
2. KL is nonnegative (gibbs-inequality.md).

Let,

  ```
  K(y|x) = a Markov kernel, the same for both inputs
  p(x,y) ≝ p(x) ⋅ K(y|x),    p(y) ≝ ∫ p(x) K(y|x) dx
  q(x,y) ≝ q(x) ⋅ K(y|x),    q(y) ≝ ∫ q(x) K(y|x) dx
  ```

First, expand the joint KL by conditioning on `x`. The conditionals of `y`
given `x` are both `K`,

  ```
  𝕂(p(x,y)‖q(x,y)) = 𝕂(p(x)‖q(x)) + 𝔼_{p(x)}[ 𝕂(K(y|x)‖K(y|x)) ]
                   = 𝕂(p(x)‖q(x)) + 0
                   = 𝕂(p(x)‖q(x))
  ```

Second, expand the same joint KL by conditioning on `y`, then drop the
nonnegative conditional term,

  ```
  𝕂(p(x,y)‖q(x,y)) = 𝕂(p(y)‖q(y)) + 𝔼_{p(y)}[ 𝕂(p(x|y)‖q(x|y)) ]
                   ≥ 𝕂(p(y)‖q(y))
  ```

Hence,

  ```
  𝕂(p(y)‖q(y)) ≤ 𝕂(p(x)‖q(x))
  ```

Equality holds iff the posteriors agree, `p(x|y) = q(x|y)` for
`p(y)`-almost every `y`. In words, `y` is sufficient for distinguishing `p`
from `q`. A deterministic map `y = g(x)` is the special case
`K(y|x) = δ(y − g(x))`.

For mutual information, let `X → Y → Z` be a Markov chain, i.e.,
`p(x,y,z) = p(x) p(y|x) p(z|y)`. Write `I(X;Y) ≝ 𝕂(p(x,y)‖p(x)p(y))`
(mutual-information.md). Third, expand `I(X;(Y,Z))` by the chain rule in two
orders,

  ```
  I(X;(Y,Z)) = 𝕂(p(x,y,z)‖p(x)p(y,z))
             = 𝕂(p(x,y)‖p(x)p(y)) + 𝔼_{p(x,y)}[ 𝕂(p(z|x,y)‖p(z|y)) ]
             = I(X;Y) + 0
  I(X;(Y,Z)) = 𝕂(p(x,z)‖p(x)p(z)) + 𝔼_{p(x,z)}[ 𝕂(p(y|x,z)‖p(y|z)) ]
             ≥ I(X;Z)
  ```

where the first expansion conditions on `(x,y)` against the reference
`p(x)p(y,z) = p(x)p(y)p(z|y)`, and the zero uses the Markov property
`p(z|x,y) = p(z|y)`. The second conditions on `(x,z)` against
`p(x)p(y,z) = p(x)p(z)p(y|z)`.

Hence,

  ```
  I(X;Z) ≤ I(X;Y)
  ```

with equality iff `p(y|x,z) = p(y|z)` almost surely, i.e., `X → Z → Y` is
also Markov.

Regularity: all KLs take values in `[0, ∞]`, the chain rule holds in that
extended sense (kl-chain-rule.md), and regular conditional densities exist
w.r.t. product base measures. The inequality holds trivially if
`𝕂(p(x)‖q(x)) = ∞`.
