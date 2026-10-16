# No Free Lunch

No free lunch follows from one fact: on a point the learner has not seen, the
labels of a uniformly random target are a fair coin, so any guess errs half
the time. With fewer than half the points seen, at least half of the
distribution is unseen, giving `1/2 ⋅ 1/2 = 1/4`.

Let,

  ```
  A                          any deterministic learner, A(S): X → {0, 1}
  n                          sample size, with |X| ≥ 2n
  C     ≝ {x₁..x_{2n}} ⊂ X   2n distinct points
  f₁..f_T                    all T ≝ 2^{2n} functions C → {0, 1}
  Dᵢ                         x uniform on C, y = fᵢ(x)
  Lᵢ(h) ≝ P_{x~Dᵢ}( h(x) ≠ fᵢ(x) ) = (1/2n) Σ_{x ∈ C} 1[h(x) ≠ fᵢ(x)]
  S₁..S_k                    all k ≝ (2n)ⁿ sequences in Cⁿ
  Sⱼⁱ   ≝ ((x, fᵢ(x)) : x ∈ Sⱼ)    sequence Sⱼ labeled by fᵢ
  ```

Each `Dᵢ` is realizable: `fᵢ` (extended arbitrarily to `X`) has `Lᵢ(fᵢ) = 0`.

First, the worst target is at least the average target. Under `Dᵢⁿ` the
instance sequence is uniform over `S₁..S_k`,

  ```
  max_i 𝔼_{S~Dᵢⁿ}[ Lᵢ(A(S)) ] = max_i (1/k) Σⱼ Lᵢ(A(Sⱼⁱ))
                              ≥ (1/T) Σᵢ (1/k) Σⱼ Lᵢ(A(Sⱼⁱ))
                              = (1/k) Σⱼ (1/T) Σᵢ Lᵢ(A(Sⱼⁱ))
                              ≥ min_j (1/T) Σᵢ Lᵢ(A(Sⱼⁱ))
  ```

where the second and fourth steps use max ≥ average ≥ min.

Second, fix `j`. Let `v₁..v_p` be the points of `C` not in `Sⱼ`. `Sⱼ` has at
most `n` distinct points, so `p ≥ 2n − n = n`. Drop the seen points from the
loss, for any `h`,

  ```
  Lᵢ(h) = (1/2n) Σ_{x ∈ C} 1[h(x) ≠ fᵢ(x)]
        ≥ (1/2n) Σ_{r=1}^{p} 1[h(v_r) ≠ fᵢ(v_r)]
        ≥ (1/2p) Σ_{r=1}^{p} 1[h(v_r) ≠ fᵢ(v_r)]
  ```

where the last step uses `p ≥ n`. Averaging over `i`,

  ```
  (1/T) Σᵢ Lᵢ(A(Sⱼⁱ)) ≥ (1/2p) Σ_r (1/T) Σᵢ 1[A(Sⱼⁱ)(v_r) ≠ fᵢ(v_r)]
                      ≥ (1/2) min_r (1/T) Σᵢ 1[A(Sⱼⁱ)(v_r) ≠ fᵢ(v_r)]
  ```

Third, fix `r`. Partition `f₁..f_T` into `T/2` pairs `(fᵢ, fᵢ′)` that differ
only at `v_r`. Since `v_r ∉ Sⱼ`, the labeled samples agree, `Sⱼⁱ = Sⱼⁱ′`, so
`A` returns the same `h` for both. Exactly one of `fᵢ(v_r)`, `fᵢ′(v_r)` equals
`h(v_r)`,

  ```
  1[h(v_r) ≠ fᵢ(v_r)] + 1[h(v_r) ≠ fᵢ′(v_r)] = 1
  ```

Summing over the `T/2` pairs,

  ```
  (1/T) Σᵢ 1[A(Sⱼⁱ)(v_r) ≠ fᵢ(v_r)] = (1/T) ⋅ (T/2)
                                    = 1/2
  ```

Hence, chaining the three steps, there is an `i` with

  ```
  𝔼_{S~Dᵢⁿ}[ Lᵢ(A(S)) ] ≥ (1/2) ⋅ (1/2)
                        = 1/4
  ```

while `Lᵢ(fᵢ) = 0`. Since `Lᵢ ∈ [0, 1]`, this also gives a high-probability
statement. With `θ ≝ Lᵢ(A(S))`, `𝔼θ ≤ 1/8 + P(θ ≥ 1/8) ⋅ (1 − 1/8)`, so

  ```
  P( θ ≥ 1/8 ) ≥ (1/4 − 1/8) / (7/8)
               = 1/7
  ```

The slack is in the second step: the seen points are dropped from the loss,
and `p ≥ n` replaces `p`. The bound says no learner works for every
distribution; a prior (a restricted class `H`) is what makes learning
possible.

Regularity: the proof needs `|X| ≥ 2n` (implied by `n < |X|/2`), binary
labels, and 0-1 loss. For a randomized learner, replace `Lᵢ(A(S))` with its
mean over the learner's internal randomness. Every step is linear in that
loss, and the pairing step still holds because both members of a pair see the
same sample, hence the same output distribution.
