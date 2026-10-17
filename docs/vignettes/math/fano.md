# Fano's Inequality

Fano's inequality follows from two facts:

1. the entropy chain rule, applied to `(X, E)` with `E` the error
   indicator, and,
2. entropy is at most the log of the support size.

Let `X` take values in a finite set `𝒳` with `|𝒳| ≥ 2`, `Y` be any
observation, and,

  ```
  X̂    ≝ g(Y)                any estimator of X from Y
  E    ≝ 1[ X̂ ≠ X ]          error indicator
  P_e  ≝ P(E = 1)
  h(t) ≝ −t log t − (1 − t) log(1 − t)
  ```

First, expand `H(E, X | Y)` by the chain rule in two orders,

  ```
  H(E, X | Y) = H(X | Y) + H(E | X, Y)
              = H(X | Y) + 0
  H(E, X | Y) = H(E | Y) + H(X | E, Y)
  ```

where `H(E | X, Y) = 0` because `E` is a function of `(X, g(Y))`. So,

  ```
  H(X | Y) = H(E | Y) + H(X | E, Y)
  ```

Second, bound the first term. Conditioning does not increase entropy
(mutual-information.md), and `E` is binary,

  ```
  H(E | Y) ≤ H(E)
           = h(P_e)
  ```

Third, bound the second term by splitting on `E`,

  ```
  H(X | E, Y) = (1 − P_e) ⋅ H(X | E = 0, Y) + P_e ⋅ H(X | E = 1, Y)
              = (1 − P_e) ⋅ 0 + P_e ⋅ H(X | E = 1, Y)
              ≤ P_e ⋅ log( |𝒳| − 1 )
  ```

where the second step uses that `E = 0` forces `X = g(Y)`, and the last uses
that given `E = 1` and `Y = y`, `X` ranges over `𝒳 ∖ {g(y)}`, a set of size
`|𝒳| − 1`, and entropy on `m` points is at most `log m` (Gibbs against the
uniform distribution, gibbs-inequality.md).

Hence,

  ```
  H(X | Y) ≤ h(P_e) + P_e ⋅ log( |𝒳| − 1 )
  ```

Fourth, the lower bound on error. Use `h(P_e) ≤ log 2` and
`log( |𝒳| − 1 ) ≤ log |𝒳|`,

  ```
  H(X | Y) ≤ log 2 + P_e ⋅ log( |𝒳| − 1 )
           ≤ log 2 + P_e ⋅ log |𝒳|
  ```

and rearrange,

  ```
  P_e ≥ ( H(X | Y) − log 2 ) / log |𝒳|
      = ( H(X) − I(X;Y) − log 2 ) / log |𝒳|
  ```

where the last step is `I(X;Y) = H(X) − H(X|Y)` (mutual-information.md).
For `X` uniform, `H(X) = log |𝒳|` and this reads
`P_e ≥ 1 − ( I(X;Y) + log 2 ) / log |𝒳|`. In bits, `log 2 = 1`. The sharper
form keeps `log( |𝒳| − 1 )` in the denominator, valid for `|𝒳| ≥ 3`.

Equality in Fano holds iff `E` is independent of `Y` (second step) and,
given an error, `X` is uniform on the `|𝒳| − 1` wrong values for every `y`
(third step).

Regularity: finite `𝒳` with `|𝒳| ≥ 2`; `X̂` a deterministic function of `Y`
(a randomized `X̂ = g(Y, U)` with `U` independent of `(X, Y)` reduces to
this with `Y' = (Y, U)`, since `H(X | Y, U) = H(X | Y)`); conditional
entropies defined via regular conditional distributions of `X` given `Y`.
