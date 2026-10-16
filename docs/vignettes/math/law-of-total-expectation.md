# Law of Total Expectation

The tower property follows from one fact: summing a joint distribution over
one variable leaves the marginal of the other.

Let,

  ```
  p(x, y)   joint pmf of discrete X, Y
  p(y)      ≝ Σₓ p(x, y)
  p(x|y)    ≝ p(x, y) / p(y)            for p(y) > 0
  𝔼[X|Y=y] ≝ Σₓ x ⋅ p(x|y)
  𝔼[X|Y]    ≝ m(Y), where m(y) ≝ 𝔼[X|Y=y]
  ```

and assume `𝔼|X| < ∞`.

First, average the conditional mean over `Y`,

  ```
  𝔼[ 𝔼[X|Y] ] = Σ_y p(y) ⋅ Σₓ x ⋅ p(x|y)
              = Σ_y Σₓ x ⋅ p(x, y)
              = Σₓ x ⋅ Σ_y p(x, y)
              = Σₓ x ⋅ p(x)
              = 𝔼[X]
  ```

where the third step swaps sums, which is valid because
`Σ_y Σₓ |x| ⋅ p(x, y) = 𝔼|X| < ∞` (Fubini–Tonelli).

Second, nest the conditioning. Let `Z` be another discrete variable, so
`σ(Y) ⊆ σ(Y, Z)`. For `p(y) > 0`,

  ```
  𝔼[ 𝔼[X|Y,Z] | Y=y ] = Σ_z p(z|y) ⋅ Σₓ x ⋅ p(x|y,z)
                      = Σ_z Σₓ x ⋅ p(x, z|y)
                      = Σₓ x ⋅ Σ_z p(x, z|y)
                      = Σₓ x ⋅ p(x|y)
                      = 𝔼[X|Y=y]
  ```

which is the first step again, carried out under `p(⋅|y)`.

Third, the general case uses the defining property. For a sub-σ-algebra
`G`, `𝔼[X|G]` is the a.s.-unique `G`-measurable `V` with `𝔼[V 1_A] = 𝔼[X 1_A]`
for all `A ∈ G`. Let `H ⊆ G`, `V = 𝔼[X|G]`, and `W = 𝔼[V|H]`. For `A ∈ H`,

  ```
  𝔼[ W 1_A ] = 𝔼[ V 1_A ]
             = 𝔼[ X 1_A ]
  ```

where the first step is the defining property of `W` (since `A ∈ H`), and the
second is the defining property of `V` (since `A ∈ H ⊆ G`). `W` is
`H`-measurable, so by uniqueness `W = 𝔼[X|H]` a.s.

Hence,

  ```
  𝔼[ 𝔼[X|Y] ]     = 𝔼[X]
  𝔼[ 𝔼[X|G] | H ] = 𝔼[X|H]     for H ⊆ G
  ```

The first identity is the second with `H = {∅, Ω}` the trivial σ-algebra,
for which `𝔼[X|H] = 𝔼[X]`. The order matters: the coarser σ-algebra wins. For
`H ⊆ G`, also `𝔼[ 𝔼[X|H] | G ] = 𝔼[X|H]`, since `𝔼[X|H]` is already
`G`-measurable.

Regularity: the proof needs `𝔼|X| < ∞` (to swap sums and for `𝔼[X|G]` to
exist), and `H ⊆ G`. In the discrete case, conditionals are defined only
where `p(y) > 0`, a set of full probability.
