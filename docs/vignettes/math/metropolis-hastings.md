# Metropolis–Hastings

Metropolis–Hastings samples `π` because of two facts:

1. the acceptance rule makes the probability flow `x → y` symmetric, and,
2. symmetric flow (detailed balance) implies stationarity.

Let,

  ```
  π(x)    > 0 a target on a countable space
  q(y|x) a proposal kernel
  α(x, y) ≝ min(1, π(y) q(x|y) / (π(x) q(y|x)))
  K(x, y) ≝ q(y|x) α(x, y),                    y ≠ x
  K(x, x) ≝ 1 − Σ_{y≠x} q(y|x) α(x, y)
  ```

so `K` proposes `y ~ q(·|x)`, accepts with probability `α(x, y)`, and
otherwise stays at `x`.

First, detailed balance. For `y ≠ x`,

  ```
  π(x) K(x, y) = π(x) q(y|x) min(1, π(y) q(x|y) / (π(x) q(y|x)))
               = min(π(x) q(y|x), π(y) q(x|y))
               = π(y) q(x|y) min(π(x) q(y|x) / (π(y) q(x|y)), 1)
               = π(y) K(y, x)
  ```

where the second and third steps move a positive factor inside `min`. For
`y = x` the identity is trivial.

Second, detailed balance implies stationarity. For every `y`,

  ```
  Σₓ π(x) K(x, y) = Σₓ π(y) K(y, x)
                  = π(y) Σₓ K(y, x)
                  = π(y)
  ```

since each row of `K` sums to one.

Third, Gibbs sampling is MH with acceptance one. Write `x ≝ (xᵢ, x₋ᵢ)` and
propose a new `i`-th coordinate from its full conditional,
`q(y|x) ≝ π(yᵢ | x₋ᵢ) 1[y₋ᵢ = x₋ᵢ]`. For `y₋ᵢ = x₋ᵢ`, factor
`π(x) = π(xᵢ | x₋ᵢ) π(x₋ᵢ)` and likewise for `y`,

  ```
  π(y) q(x|y) / (π(x) q(y|x))
      = π(yᵢ|x₋ᵢ) π(x₋ᵢ) π(xᵢ|x₋ᵢ) / (π(xᵢ|x₋ᵢ) π(x₋ᵢ) π(yᵢ|x₋ᵢ))
      = 1
  ```

so `α = 1` and every proposal is accepted.

Hence,

  ```
  π(x) K(x, y) = π(y) K(y, x)   and   π K = π
  ```

for every proposal `q`, and each Gibbs coordinate update leaves `π` invariant.

The MH rule accepts as often as detailed balance allows. Any `α′ ≤ 1` with
`π(x) q(y|x) α′(x, y) = π(y) q(x|y) α′(y, x)` has flow at most
`min(π(x) q(y|x), π(y) q(x|y))`, which MH attains.

Fine print: `π` enters only through the ratio `π(y)/π(x)`, so an unnormalized
target suffices. If `q(y|x) > 0` but `q(x|y) = 0`, then `α(x, y) = 0` and the
move is never made; pairs with `q(y|x) = 0` never arise. On continuous spaces
replace `Σ` by `∫` against densities; `K(x, ·)` then has an atom at `x`. A
systematic scan over coordinates is a product of `π`-invariant kernels, so it
is `π`-invariant but generally not reversible. Stationarity alone does not
give convergence; that needs irreducibility and aperiodicity (see
perron-frobenius-mixing.md).

Regularity: the proof needs `π > 0` wherever the chain can go, a proposal
`q(·|x)` that is a probability distribution, and the rows of `K` summing to
one. Detailed balance is sufficient for stationarity, not necessary.
