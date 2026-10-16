# Finite-Class PAC Bound

The bound follows from two facts:

1. Hoeffding's inequality (see hoeffding.md) controls one fixed `h`, and,
2. the union bound pays `log|H|` to control every `h` at once.

Let,

  ```
  H                      finite hypothesis class
  ℓ(h, z) ∈ [0, 1]       bounded loss
  S    ≝ (z₁..zₙ) ~ Dⁿ   iid sample
  R(h) ≝ 𝔼_{z~D}[ ℓ(h, z) ]
  R̂(h) ≝ (1/n) Σᵢ ℓ(h, zᵢ)
  ```

First, for one fixed `h`, the terms `ℓ(h, zᵢ)` are iid in `[0, 1]` with mean
`R(h)`. Two-sided Hoeffding gives, for `t > 0`,

  ```
  P( |R(h) − R̂(h)| ≥ t ) ≤ 2 ⋅ exp(−2 n t² / (1 − 0)²)
                         = 2 ⋅ exp(−2 n t²)
  ```

Second, the union bound over `H`,

  ```
  P( sup_h |R(h) − R̂(h)| ≥ t ) = P( ∪_{h ∈ H} { |R(h) − R̂(h)| ≥ t } )
                               ≤ Σ_{h ∈ H} P( |R(h) − R̂(h)| ≥ t )
                               ≤ 2 |H| ⋅ exp(−2 n t²)
  ```

Third, set the right side to `δ` and solve for `t`,

  ```
  2 |H| ⋅ exp(−2 n t²) = δ
               2 n t² = log(2|H|/δ)
                    t = √( log(2|H|/δ) / (2n) )
  ```

Hence, with probability at least `1 − δ`,

  ```
  sup_{h ∈ H} |R(h) − R̂(h)| ≤ √( log(2|H|/δ) / (2n) )
  ```

and, by approx-estimation.md, ERM has `R(ĥ) − R_H ≤ 2 √(log(2|H|/δ)/(2n))`.

For the realizable case, take 0-1 loss and assume some `h ∈ H` has `R(h) = 0`.
Then ERM attains `R̂(ĥ) = 0`. Fix `ε > 0` and call `h` bad if `R(h) > ε`. A bad
`h` survives, i.e., has `R̂(h) = 0`, only if it is correct on all `n` iid
points,

  ```
  P( R̂(h) = 0 ) = (1 − R(h))ⁿ
                ≤ (1 − ε)ⁿ
                ≤ exp(−ε n)
  ```

where the last step uses `1 − ε ≤ e^{−ε}`. The union bound over at most `|H|`
bad hypotheses, set equal to `δ`,

  ```
  P( ∃ h: R(h) > ε, R̂(h) = 0 ) ≤ |H| ⋅ exp(−ε n) = δ
                             ε = log(|H|/δ) / n
  ```

Hence, in the realizable case, with probability at least `1 − δ`,

  ```
  R(ĥ) ≤ log(|H|/δ) / n
  ```

The rate is `1/n` instead of `1/√n` because a zero-error event has
probability `(1 − R)ⁿ`, not a two-sided Gaussian-like tail. The union bound is
tight when the bad events are disjoint; it is loose when hypotheses are
correlated, which is what VC and Rademacher bounds exploit.

Regularity: the proof needs `S` iid, `ℓ ∈ [0, 1]`, `H` finite, and `H` fixed
before seeing `S`. The realizable case also needs 0-1 loss and some `h ∈ H`
with `R(h) = 0`.
