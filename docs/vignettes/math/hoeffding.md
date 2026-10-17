# Hoeffding

Hoeffding follows from two facts:

1. on `[a, b]`, the convex `e^{λx}` lies below its chord, and,
2. the log of that chord bound has second derivative at most `1/4`.

Let,

  ```
  X ∈ [a, b] a.s., a < b
  Y       ≝ X − 𝔼[X] ∈ [a', b'], with a' ≝ a − 𝔼[X] ≤ 0 ≤ b' ≝ b − 𝔼[X]
  b' − a' = b − a
  p       ≝ −a' / (b' − a') ∈ [0, 1],  so 1 − p = b' / (b' − a')
  h       ≝ λ ⋅ (b − a),  for real λ
  ```

First, bound `e^{λy}` by its chord on `[a', b']`. Write `y` as a convex
combination of the endpoints,

  ```
  e^{λy} ≤ (b' − y)/(b' − a') ⋅ e^{λa'} + (y − a')/(b' − a') ⋅ e^{λb'}
  ```

Take expectations, using `𝔼[Y] = 0`,

  ```
  𝔼[ e^{λY} ] ≤ b'/(b' − a') ⋅ e^{λa'} − a'/(b' − a') ⋅ e^{λb'}
              = (1 − p) ⋅ e^{λa'} + p ⋅ e^{λb'}
              = (1 − p) ⋅ e^{−ph} + p ⋅ e^{(1 − p)h}
              = e^{L(h)},   L(h) ≝ −ph + log(1 − p + p ⋅ eʰ)
  ```

where the third step uses `λa' = −ph` and `λb' = λa' + h = (1 − p)h`.

Second, Taylor-expand `L` around `0`. Let `u(h) ≝ p eʰ / (1 − p + p eʰ)`,
which lies in `[0, 1]`,

  ```
  L(0)   = 0 + log(1) = 0
  L'(h)  = −p + u(h),            L'(0) = −p + p = 0
  L''(h) = u(h) ⋅ (1 − u(h))
         ≤ 1/4
  ```

where the last step uses `u(1 − u) ≤ 1/4` for `u ∈ [0, 1]`. By Taylor's
theorem with Lagrange remainder, for some `ξ` between `0` and `h`,

  ```
  L(h) = L(0) + h ⋅ L'(0) + (h²/2) ⋅ L''(ξ)
       ≤ 0 + 0 + h²/8
       = λ² (b − a)² / 8
  ```

This is Hoeffding's lemma: `𝔼[e^{λ(X − 𝔼X)}] ≤ e^{λ²(b − a)²/8}`.

Third, sums. Let `X₁, …, Xₙ` be independent with `Xᵢ ∈ [aᵢ, bᵢ]`,
`S ≝ Σᵢ Xᵢ`, and `D ≝ Σᵢ (bᵢ − aᵢ)²`. For `λ > 0` and `t > 0`, Chernoff
(Markov on `e^{λ(S − 𝔼S)}`) gives,

  ```
  P(S − 𝔼S ≥ t) ≤ e^{−λt} ⋅ 𝔼[ e^{λ(S − 𝔼S)} ]
                = e^{−λt} ⋅ Πᵢ 𝔼[ e^{λ(Xᵢ − 𝔼Xᵢ)} ]
                ≤ e^{−λt} ⋅ Πᵢ e^{λ²(bᵢ − aᵢ)²/8}
                = exp( −λt + λ² D / 8 )
  ```

where the second step uses independence, and the third uses the lemma.

Fourth, choose `λ = 4t / D`, which minimizes the exponent,

  ```
  −λt + λ² D/8 = −4t²/D + (16t²/D²) ⋅ D/8
               = −4t²/D + 2t²/D
               = −2t²/D
  ```

Hence,

  ```
  P(S − 𝔼S ≥ t)   ≤ exp( −2t² / Σᵢ (bᵢ − aᵢ)² )
  P(|S − 𝔼S| ≥ t) ≤ 2 ⋅ exp( −2t² / Σᵢ (bᵢ − aᵢ)² )
  ```

The two-sided bound applies the one-sided bound to `−Xᵢ ∈ [−bᵢ, −aᵢ]` and
adds. For iid `Xᵢ ∈ [a, b]` with mean `μ`, taking `t = nε` gives
`P(X̄ − μ ≥ ε) ≤ exp(−2nε² / (b − a)²)`.

The constant `1/8` is sharp: for `X` uniform on `{a, b}`, `p = 1/2` and
`L''(0) = 1/4`, so `L(h) = h²/8 + O(h⁴)`. The tail bound itself is not
attained for any finite `t`.

Regularity: the proof needs each `Xᵢ` bounded a.s. in a known interval,
mutual independence (to factor the MGF), and `t > 0`. No variance
information is used, which is why the bound is loose for low-variance `Xᵢ`
(Bernstein improves it). If `a = b`, `X` is constant and the lemma is
trivial.
