# Bernstein's Inequality

Bernstein follows from two facts:

1. for `|x| < 3`, `eˣ − 1 − x ≤ (x²/2) / (1 − |x|/3)`, because `k! ≥ 2 ⋅ 3ᵏ⁻²`,
   and,
2. that bound makes the MGF depend on the variance, not just the range.

Let,

  ```
  X₁, …, Xₙ    independent, μᵢ ≝ 𝔼[Xᵢ]
  Yᵢ  ≝ Xᵢ − μᵢ,   |Yᵢ| ≤ b a.s.
  σᵢ² ≝ 𝕍(Xᵢ),     σ² ≝ Σᵢ σᵢ²
  S   ≝ Σᵢ Yᵢ
  0   < λ < 3/b,   t > 0
  ```

First, bound the exponential remainder. For `|x| < 3`,

  ```
  eˣ − 1 − x = Σₖ₌₂^∞ xᵏ / k!
             ≤ Σₖ₌₂^∞ |x|ᵏ / (2 ⋅ 3ᵏ⁻²)
             = (x²/2) ⋅ Σⱼ₌₀^∞ (|x|/3)ʲ
             = (x²/2) / (1 − |x|/3)
  ```

where the second step uses `xᵏ ≤ |x|ᵏ` and `k! ≥ 2 ⋅ 3ᵏ⁻²` for `k ≥ 2`.

Second, bound one MGF. Use `|λYᵢ| ≤ λb < 3` and `𝔼[Yᵢ] = 0`,

  ```
  𝔼[ e^{λYᵢ} ] = 1 + λ𝔼[Yᵢ] + 𝔼[ e^{λYᵢ} − 1 − λYᵢ ]
               ≤ 1 + 0 + 𝔼[ λ²Yᵢ² / (2(1 − λb/3)) ]
               = 1 + λ²σᵢ² / (2(1 − λb/3))
               ≤ exp( λ²σᵢ² / (2(1 − λb/3)) )
  ```

where the second step also uses `|Yᵢ| ≤ b` in the denominator, and the last
uses `1 + u ≤ eᵘ`.

Third, Chernoff (`markov-chebyshev-chernoff.md`) and independence,

  ```
  P(S ≥ t) ≤ e^{−λt} ⋅ Πᵢ 𝔼[ e^{λYᵢ} ]
           ≤ exp( −λt + λ²σ² / (2(1 − λb/3)) )
  ```

Fourth, choose `λ = t / (σ² + bt/3)`. It is admissible, since `λb < 3` iff
`bt < 3σ² + bt`. Then `1 − λb/3 = σ² / (σ² + bt/3)`, and,

  ```
  −λt + λ²σ² / (2(1 − λb/3)) = −λt + λ² (σ² + bt/3) / 2
                             = −t²/(σ² + bt/3) + t²/(2(σ² + bt/3))
                             = −t² / (2(σ² + bt/3))
  ```

Hence,

  ```
  P(S ≥ t) ≤ exp( −t² / (2(σ² + bt/3)) )
  ```

The bound has two regimes. Splitting on `bt/3 ≤ σ²` (denominator at most
`4σ²`) or `bt/3 > σ²` (denominator below `4bt/3`),

  ```
  P(S ≥ t) ≤ exp( −t² / (4σ²) )     for t ≤ 3σ²/b   (sub-Gaussian)
  P(S ≥ t) ≤ exp( −3t / (4b) )      for t > 3σ²/b   (sub-exponential)
  ```

For `t ≪ σ²/b` the exponent is `t²/(2σ²)(1 − O(bt/σ²))`, matching the
Gaussian tail of the CLT (`clt.md`) to leading order.

Fine print: Hoeffding (`hoeffding.md`) with range `2b` gives
`exp(−t²/(2nb²))`, which ignores variance; Bernstein wins when
`σ² ≪ nb²`, e.g., rare events. The boundedness can be weakened to the moment
condition `𝔼|Yᵢ|ᵏ ≤ (k!/2) σᵢ² bᵏ⁻²` for `k ≥ 2`, with the same proof
(denominator `1 − λb`, giving `exp(−t²/(2(σ² + bt)))`). Bennett's inequality
keeps the exact function `(1 + u) log(1 + u) − u` instead of the quadratic
bound, and Bernstein follows from it. The lower tail follows by applying the
bound to `−Yᵢ`. The upper tail needs only `Yᵢ ≤ b`: `(eˣ − 1 − x)/x²` is
increasing, so `e^{λY} − 1 − λY ≤ λ²Y² (e^{λb} − 1 − λb)/(λb)²`, and the
first step bounds the last factor by `1/(2(1 − λb/3))`.

Regularity: the proof needs independence (to factor the MGF), `|Yᵢ| ≤ b`
a.s. (or `Yᵢ ≤ b` for the upper tail, per the fine print), finite `σ²`, and
`t > 0`.
