# Dvoretzky–Kiefer–Wolfowitz–Massart

The weaker DKW bound proved here follows from two facts:

1. Hoeffding plus a union bound controls `Fₙ` on a finite grid, and,
2. the Glivenko–Cantelli bracket turns grid control into uniform control.

Massart's sharp constant, `2 e^{−2nε²}`, is cited.

Let,

  ```
  X₁, …, Xₙ iid ~ F,   F(x) ≝ P(X ≤ x)
  Fₙ(x)  ≝ (1/n) Σᵢ 1{Xᵢ ≤ x}
  Fₙ(x−) ≝ (1/n) Σᵢ 1{Xᵢ < x},   F(x−) ≝ P(X < x)
  Dₙ     ≝ sup_x |Fₙ(x) − F(x)|
  ```

Fix `ε ∈ (0, 1]` and set `k ≝ ⌈2/ε⌉`, so `1/k ≤ ε/2` and `k − 1 < 2/ε`.

First, take the grid `x₁ ≤ … ≤ xₖ₋₁` and the bracket from
glivenko-cantelli.md (first and third steps). For every `n`, with no limit
taken, they give

  ```
  Dₙ ≤ Δₙ + 1/k
  Δₙ ≝ maxⱼ max( |Fₙ(xⱼ) − F(xⱼ)|, |Fₙ(xⱼ−) − F(xⱼ−)| )
  ```

where `j` ranges over `1, …, k−1`, so `Δₙ` is a max of `2(k−1)` terms.

Second, so `Dₙ > ε` forces a large grid deviation,

  ```
  Dₙ > ε  ⇒  Δₙ > ε − 1/k
          ⇒  Δₙ > ε/2
  ```

Third, each grid term is a mean of iid indicators in `[0, 1]`. Hoeffding
(hoeffding.md) with `t = ε/2` gives, for each of the `2(k−1)` terms,

  ```
  P( |Fₙ(xⱼ) − F(xⱼ)| ≥ ε/2 ) ≤ 2 exp( −2n (ε/2)² )
                              = 2 exp( −nε²/2 )
  ```

and the same for `Fₙ(xⱼ−)`.

Fourth, take a union bound over the grid,

  ```
  P(Dₙ > ε) ≤ P(Δₙ > ε/2)
            ≤ 2(k−1) ⋅ 2 exp( −nε²/2 )
            < (2/ε) ⋅ 4 exp( −nε²/2 )
            = (8/ε) ⋅ exp( −nε²/2 )
  ```

where the third step uses `k − 1 < 2/ε`.

Hence, proved here,

  ```
  P( sup_x |Fₙ(x) − F(x)| > ε ) ≤ (8/ε) ⋅ exp( −nε²/2 )
  ```

and, cited (Massart 1990), the sharp form,

  ```
  P( sup_x |Fₙ(x) − F(x)| > ε ) ≤ 2 ⋅ exp( −2nε² )
  ```

The weaker bound loses a factor `4` in the exponent and a `1/ε` prefactor.
The factor `4` comes from splitting `ε` in half between grid and bracket. The
`1/ε` comes from the union bound over `O(1/ε)` grid points. Massart's
constant `2` cannot be lowered: as `ε → 0` with `√n ε` fixed, the
Kolmogorov limit `P(sup|B| > λ) = 2 Σ_{m≥1} (−1)^{m−1} e^{−2m²λ²}`, with `B`
a Brownian bridge, is `2e^{−2λ²}(1 − o(1))` as `λ → ∞`.

Confidence band. Set the sharp bound equal to `α`,

  ```
  εₙ ≝ √( log(2/α) / (2n) )
  P( F(x) ∈ [Fₙ(x) − εₙ, Fₙ(x) + εₙ] for all x ) ≥ 1 − α
  ```

so `[max(Fₙ − εₙ, 0), min(Fₙ + εₙ, 1)]` is a distribution-free `1 − α`
band for the whole CDF, of width `O(1/√n)`. The weaker bound gives a valid
band at least twice as wide, with an extra `log n` under the square root from
the `1/ε` prefactor.

Fine print: Massart proves the sharp form for continuous `F`. It extends to
every `F`. Write `X = F⁻¹(U)` with `U` uniform and `Gₙ` the uniform
empirical CDF. Then `Fₙ(x) − F(x) = Gₙ(t) − t` at `t = F(x)`, so
`sup_x |Fₙ(x) − F(x)| ≤ sup_t |Gₙ(t) − t|`. The weaker bound needs no such
step, because the bracket handles atoms directly. Since
`Σₙ (8/ε) e^{−nε²/2} < ∞`, Borel–Cantelli (borel-cantelli.md) recovers
Glivenko–Cantelli from the weaker bound alone. For `ε > 1` both
probabilities are `0`, since `Dₙ ≤ 1`.

Regularity: the proof needs `X₁, …, Xₙ` iid real-valued, Hoeffding for
`[0, 1]`-valued means, the union bound, and the deterministic bracket of
glivenko-cantelli.md. No condition on `F` is used. The sharp constant is
cited, not proved.
