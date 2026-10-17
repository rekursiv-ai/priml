# Gibbs' Inequality

Gibbs' inequality follows from one fact: `log x ≤ x − 1`, with equality iff
`x = 1`.

Let,

  ```
  p, q ≝ densities of P, Q w.r.t. a common base measure μ
  S      ≝ { x : p(x) > 0 }
  𝕂(p‖q) ≝ ∫_S p(x) ⋅ log( p(x) / q(x) ) dμ(x)
  ```

with `𝕂(p‖q) = ∞` if `q = 0` on a subset of `S` of positive `μ`-measure.

First, `log x ≤ x − 1` for all `x > 0`. Let `φ(x) ≝ x − 1 − log x`. Then
`φ'(x) = 1 − 1/x` is negative on `(0, 1)` and positive on `(1, ∞)`, so `φ`
has its unique minimum at `x = 1`,

  ```
  φ(x) ≥ φ(1)
       = 1 − 1 − log 1
       = 0
  ```

with equality iff `x = 1`.

Second, assume `q > 0` on `S` (else `𝕂 = ∞` and there is nothing to prove).
Apply the first step with `x ← q(x)/p(x)` at each point of `S`,

  ```
  −𝕂(p‖q) = ∫_S p(x) ⋅ log( q(x) / p(x) ) dμ(x)
          ≤ ∫_S p(x) ⋅ ( q(x) / p(x) − 1 ) dμ(x)
          = ∫_S q(x) dμ(x) − ∫_S p(x) dμ(x)
          = Q(S) − 1
          ≤ 0
  ```

where the last step uses `Q(S) ≤ 1`.

Hence,

  ```
  𝕂(p‖q) ≥ 0
  ```

Equality holds iff both inequalities are tight. The first is tight iff
`q(x)/p(x) = 1` for `μ`-almost every `x ∈ S`; the second iff `Q(S) = 1`.
Together these say `q = p` `μ`-almost everywhere, i.e., `P = Q`.

Equivalently, via Jensen (see jensen.md) applied to the concave `log`,
`−𝕂(p‖q) = 𝔼_p[ log(q/p) ] ≤ log 𝔼_p[ q/p ] = log Q(S) ≤ 0`. Strict
concavity gives the same equality condition: `q/p` is `p`-a.s. constant,
and `Q(S) = 1` forces the constant to be `1`.

Regularity: `P` and `Q` have densities w.r.t. a common `μ` (always true with
`μ = P + Q`), and the integral `∫_S p ⋅ log(q/p) dμ` is well defined in
`[−∞, 0]`, which holds because its integrand is bounded above by the
integrable `q − p`.
