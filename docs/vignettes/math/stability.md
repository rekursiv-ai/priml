# Uniform Stability

The stability bound follows from two facts:

1. swapping a training point with a fresh one leaves the joint law unchanged,
   so the population risk is the loss on a replaced-out training point, and,
2. a stable algorithm barely notices the replacement.

Let,

  ```
  S    ≝ (z₁..zₙ) ~ Dⁿ,   z′₁..z′ₙ ~ D        iid, independent
  S⁽ⁱ⁾ ≝ (z₁..z′ᵢ..zₙ)                    S with zᵢ replaced by z′ᵢ
  h_S  ≝ A(S)                             output of the algorithm
  R(h) ≝ 𝔼_{z~D}[ ℓ(h, z) ],    R̂_S(h) ≝ (1/n) Σᵢ ℓ(h, zᵢ)
  β-stable:  |ℓ(h_S, z) − ℓ(h_{S⁽ⁱ⁾}, z)| ≤ β     for all S, i, z′ᵢ, z
  ```

First, rewrite the population risk. `z′ᵢ` is independent of `S` with law
`D`, and `(S, z′ᵢ)` has the same joint law as `(S⁽ⁱ⁾, zᵢ)`,

  ```
  𝔼_S[ R(h_S) ] = (1/n) Σᵢ 𝔼[ ℓ(h_S, z′ᵢ) ]
                = (1/n) Σᵢ 𝔼[ ℓ(h_{S⁽ⁱ⁾}, zᵢ) ]
  ```

Second, subtract the empirical risk and apply stability at `z = zᵢ`,

  ```
  𝔼_S[ R(h_S) − R̂_S(h_S) ] = (1/n) Σᵢ 𝔼[ ℓ(h_{S⁽ⁱ⁾}, zᵢ) − ℓ(h_S, zᵢ) ]
                           ≤ (1/n) Σᵢ β
                           = β
  ```

Third, concentrate. Let `Φ(S) ≝ R(h_S) − R̂_S(h_S)` and `ℓ ∈ [0, M]`.
Replacing `zᵢ` by `z′ᵢ` moves `R(h_S)` by at most `β`, and moves the
empirical risk by

  ```
  |R̂_S(h_S) − R̂_{S⁽ⁱ⁾}(h_{S⁽ⁱ⁾})| ≤ (1/n) Σ_{j≠i} |ℓ(h_S, zⱼ) − ℓ(h_{S⁽ⁱ⁾}, zⱼ)| + (1/n) |ℓ(h_S, zᵢ) − ℓ(h_{S⁽ⁱ⁾}, z′ᵢ)|
                                  ≤ (n − 1) β / n + M / n
                                  ≤ β + M / n
  ```

So `Φ` has bounded differences `cᵢ = 2β + M/n`. McDiarmid (see
azuma-mcdiarmid.md) gives,

  ```
  P( Φ − 𝔼Φ ≥ t ) ≤ exp(−2t² / Σᵢ cᵢ²)
                  = exp(−2t² / (n (2β + M/n)²))
  ```

Setting this to `δ` gives `t = (2nβ + M) √(log(1/δ)/(2n))`.

Hence, in expectation and with probability at least `1 − δ`,

  ```
  𝔼_S[ R(h_S) − R̂_S(h_S) ] ≤ β
  R(h_S)                   ≤ R̂_S(h_S) + β + (2nβ + M) ⋅ √( log(1/δ) / (2n) )
  ```

Fourth, regularized ERM in an RKHS is stable. Let `H` have kernel `k` with
`k(x, x) ≤ κ²`, let `ℓ(f, (x, y)) = c(f(x), y)` with `c(·, y)` convex and
`ρ`-Lipschitz, and let `f_S` minimize `F_S(f) ≝ R̂_S(f) + λ ‖f‖²_H`. Write
`f ≝ f_S`, `g ≝ f_{S⁽ⁱ⁾}`. Then `F_S − λ‖·‖²` is convex, so for any `h` and
`t ∈ (0, 1)`, using `‖f + t(h − f)‖² = (1−t)‖f‖² + t‖h‖² − t(1−t)‖h − f‖²`,

  ```
  F_S(f) ≤ F_S(f + t(h − f))
         ≤ (1 − t) F_S(f) + t F_S(h) − λ t (1 − t) ‖h − f‖²
  ```

Rearranging, dividing by `t`, and letting `t → 0` gives
`F_S(h) − F_S(f) ≥ λ ‖h − f‖²`. Apply this to both problems and add,

  ```
  2λ ‖f − g‖² ≤ [F_S(g) − F_S(f)] + [F_{S⁽ⁱ⁾}(f) − F_{S⁽ⁱ⁾}(g)]
              = (1/n) [c(g(xᵢ), yᵢ) − c(f(xᵢ), yᵢ)] + (1/n) [c(f(x′ᵢ), y′ᵢ) − c(g(x′ᵢ), y′ᵢ)]
              ≤ (ρ/n) [ |g(xᵢ) − f(xᵢ)| + |f(x′ᵢ) − g(x′ᵢ)| ]
              ≤ (2ρκ/n) ‖f − g‖
  ```

where the second step uses that `F_S` and `F_{S⁽ⁱ⁾}` differ only in the
`i`-th loss term, and the last uses the reproducing property and
Cauchy–Schwarz, `|h(x)| = |⟨h, k(x, ·)⟩| ≤ κ ‖h‖`. So
`‖f − g‖ ≤ ρκ/(λn)`, and for every `z = (x, y)`,

  ```
  |ℓ(f, z) − ℓ(g, z)| ≤ ρ ⋅ |f(x) − g(x)|
                      ≤ ρκ ⋅ ‖f − g‖
                      ≤ ρ²κ² / (λn)
  ```

So regularized ERM is `β`-stable with `β = ρ²κ²/(λn)`.

The expectation bound is tight: it is an identity before the stability step,
so equality holds when every replacement changes the loss at `zᵢ` by exactly
`β`.

Fine print: the high-probability bound is nontrivial only when
`β = o(1/√n)`; the RKHS example gives `β ∝ 1/(λn)`, so the deviation term is
`O(ρ²κ²/(λ√n) + M/√n)`. A Lipschitz loss is bounded once `f_S` is:
`F_S(f_S) ≤ F_S(0)` gives `‖f_S‖² ≤ B/λ` with `B ≝ sup_y c(0, y)`, so
`M ≤ B + ρκ √(B/λ)` works. Bousquet and Elisseeff state stability with
remove-one rather than replace-one, which changes constants only. The
algorithm need not be symmetric in its inputs.

Regularity: the proof needs `S` iid and independent of the ghost points,
`β`-stability for all `S`, `i`, `z′ᵢ`, `z`, and `ℓ ∈ [0, M]` for the
high-probability bound. The RKHS example needs `k(x, x) ≤ κ²`, `c(·, y)`
convex and `ρ`-Lipschitz, `λ > 0`, and that the minimizer `f_S` exists.
