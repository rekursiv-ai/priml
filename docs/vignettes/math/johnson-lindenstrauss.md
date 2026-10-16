# Johnson–Lindenstrauss

Johnson–Lindenstrauss follows from two facts:

1. a Gaussian projection of a fixed unit vector has squared norm `χ²ₖ / k`,
   which Chernoff concentrates at rate `e^{−kε²/4}`, and,
2. a union bound over the `n(n − 1)/2` pairs costs only `log n²`.

Let,

  ```
  x₁, …, xₙ ∈ ℝᵈ          the points
  G ∈ ℝ^{k×d}             iid N(0, 1) entries
  P ≝ G / √k              the projection
  u ∈ ℝᵈ, ‖u‖ = 1         a fixed direction
  Q ≝ ‖Gu‖²
  0 < ε < 1,  0 < δ < 1
  ```

First, `Q ~ χ²ₖ`. The rows `gⱼ` of `G` are iid `N(0, I_d)`, so the
coordinates `gⱼᵀu ~ N(0, ‖u‖²) = N(0, 1)` are iid, and
`Q = Σⱼ (gⱼᵀu)²`. Completing the square gives, for `λ < 1/2`,

  ```
  𝔼[ e^{λZ²} ] = ∫ (2π)^{−1/2} e^{−(1 − 2λ)z²/2} dz
               = (1 − 2λ)^{−1/2}
  𝔼[ e^{λQ} ]  = (1 − 2λ)^{−k/2}
  ```

Second, the upper tail by Chernoff (`markov-chebyshev-chernoff.md`), at the
minimizer `λ = ε / (2(1 + ε))`, where `1 − 2λ = 1/(1 + ε)`,

  ```
  P(Q ≥ (1 + ε)k) ≤ e^{−λ(1 + ε)k} ⋅ (1 − 2λ)^{−k/2}
                  = exp( −εk/2 + (k/2) log(1 + ε) )
                  = exp( −(k/2)(ε − log(1 + ε)) )
                  ≤ exp( −(k/2)(ε²/2 − ε³/3) )
  ```

where the last step uses `log(1 + ε) ≤ ε − ε²/2 + ε³/3` for `ε ≥ 0` (the
difference is `0` at `0` and has derivative `ε³/(1 + ε) ≥ 0`).

Third, the lower tail by Chernoff on `e^{−λQ}`, at `λ = ε / (2(1 − ε))`,
where `1 + 2λ = 1/(1 − ε)`,

  ```
  P(Q ≤ (1 − ε)k) ≤ e^{λ(1 − ε)k} ⋅ (1 + 2λ)^{−k/2}
                  = exp( εk/2 + (k/2) log(1 − ε) )
                  = exp( −(k/2)(−ε − log(1 − ε)) )
                  ≤ exp( −(k/2)(ε²/2) )
                  ≤ exp( −(k/2)(ε²/2 − ε³/3) )
  ```

where the fourth step uses `−ε − log(1 − ε) = Σⱼ₌₂^∞ εʲ/j ≥ ε²/2`.

Fourth, union bound over pairs. For `i < j`, `P` is linear, so
`‖P(xᵢ − xⱼ)‖² / ‖xᵢ − xⱼ‖² = Q/k` with `u = (xᵢ − xⱼ)/‖xᵢ − xⱼ‖`. Each pair
fails with probability at most `2 exp(−k(ε²/4 − ε³/6))`, so,

  ```
  P(some pair fails) ≤ (n(n − 1)/2) ⋅ 2 exp( −k(ε²/4 − ε³/6) )
                     ≤ n² exp( −k(ε²/4 − ε³/6) )
                     ≤ δ
  ```

where the last step holds once `k ≥ log(n²/δ) / (ε²/4 − ε³/6)`.

Hence, with probability at least `1 − δ`, for all `i, j`,

  ```
  (1 − ε) ‖xᵢ − xⱼ‖² ≤ ‖P xᵢ − P xⱼ‖² ≤ (1 + ε) ‖xᵢ − xⱼ‖²
  whenever  k ≥ 4 log(n²/δ) / (ε² − 2ε³/3)
  ```

Since `ε² − 2ε³/3 ≥ ε²/3` on `(0, 1)`, `k ≥ 12 log(n²/δ)/ε²` suffices, and
`k ≥ 6 log(n²/δ)/ε²` suffices for `ε ≤ 1/2`. The guarantee is for squared
distances; unsquared distances then lie within `√(1 ± ε) ⊆ [1 − ε, 1 + ε]`.

Fine print: `k = O(log n / ε²)` is optimal for any linear or nonlinear map
(Larsen–Nelson 2017, cited, not proved). The dimension `d` never enters.
The same proof works for any projection whose entries are sub-Gaussian
(e.g., random `±1`), with a different constant, via the sub-Gaussian
analogue of the first step. Dasgupta–Gupta state the bound as
`k ≥ 4 log n / (ε²/2 − ε³/3)`. That is the above with per-pair failure
`2/n²`, so the union bound gives success probability only `≥ 1/n`, boosted
by repeating the projection.

Regularity: the proof needs iid Gaussian entries (so `Q ~ χ²ₖ` exactly for
every fixed `u`), `0 < ε < 1` (so the lower-tail `λ` is positive and finite),
and distinct points (so each `u` is defined). The `n` points are fixed
before `G` is drawn.
