# Law of Total Variance

Total variance follows from two facts:

1. the tower property, `𝔼[𝔼[X|Y]] = 𝔼[X]`, and,
2. `𝕍(V) = 𝔼[V²] − (𝔼[V])²`, applied conditionally and unconditionally.

Let,

  ```
  m(Y)   ≝ 𝔼[X|Y]
  𝕍(X|Y) ≝ 𝔼[ (X − m(Y))² | Y ] = 𝔼[X²|Y] − m(Y)²
  ```

and assume `𝔼[X²] < ∞`.

First, the mean of the conditional variance,

  ```
  𝔼[ 𝕍(X|Y) ] = 𝔼[ 𝔼[X²|Y] − m(Y)² ]
              = 𝔼[ 𝔼[X²|Y] ] − 𝔼[ m(Y)² ]
              = 𝔼[X²] − 𝔼[ m(Y)² ]
  ```

where the last step uses the tower property on `X²`.

Second, the variance of the conditional mean,

  ```
  𝕍( m(Y) ) = 𝔼[ m(Y)² ] − (𝔼[ m(Y) ])²
            = 𝔼[ m(Y)² ] − (𝔼[X])²
  ```

where the last step uses the tower property on `X`.

Third, add the two,

  ```
  𝔼[ 𝕍(X|Y) ] + 𝕍( m(Y) ) = 𝔼[X²] − 𝔼[ m(Y)² ] + 𝔼[ m(Y)² ] − (𝔼[X])²
                          = 𝔼[X²] − (𝔼[X])²
                          = 𝕍(X)
  ```

Hence,

  ```
  𝕍(X) = 𝔼[ 𝕍(X|Y) ] + 𝕍( 𝔼[X|Y] )
  ```

Both terms are nonnegative. `𝕍(𝔼[X|Y]) = 0` iff `𝔼[X|Y]` is a.s. constant,
i.e., `Y` carries no information about the mean of `X`. `𝔼[𝕍(X|Y)] = 0` iff
`X = 𝔼[X|Y]` a.s., i.e., `X` is a function of `Y`. Equivalently,
`𝕍(𝔼[X|Y]) ≤ 𝕍(X)`: conditioning never increases the variance of an
estimator (Rao–Blackwell).

Regularity: the proof needs `𝔼[X²] < ∞`, which makes `m(Y)` square-integrable
(by conditional Jensen, `m(Y)² ≤ 𝔼[X²|Y]`) and all four expectations finite.
