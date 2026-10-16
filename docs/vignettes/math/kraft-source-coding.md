# Kraft Inequality and Source Coding

The source coding theorem follows from two facts:

1. prefix codewords own disjoint dyadic subintervals of `[0, 1)` (Kraft),
   and,
2. KL is nonnegative (gibbs-inequality.md).

Let `X` take values in a finite alphabet `𝒳` with mass function `p`, and,

  ```
  c : 𝒳 → {0,1}*        prefix code: no codeword is a prefix of another
  ℓ(x) ≝ |c(x)|           codeword length
  L    ≝ Σₓ p(x) ⋅ ℓ(x)   expected length
  H(X) ≝ −Σₓ p(x) ⋅ log₂ p(x)
  ```

First, Kraft. Map codeword `c(x) = b₁⋯b_ℓ` to the interval
`I(x) ≝ [ 0.b₁⋯b_ℓ, 0.b₁⋯b_ℓ + 2^{−ℓ} )` in binary, of length `2^{−ℓ(x)}`.
`I(x)` is exactly the set of reals in `[0,1)` whose binary expansion starts
with `c(x)`. If `I(x)` and `I(x')` met, one expansion would start with both
`c(x)` and `c(x')`, so one codeword would be a prefix of the other. So the
intervals are disjoint subsets of `[0,1)`, and,

  ```
  Σₓ 2^{−ℓ(x)} = Σₓ |I(x)|
               ≤ |[0,1)|
               = 1
  ```

Second, the converse: if `Σₓ 2^{−ℓ(x)} ≤ 1`, a prefix code with these
lengths exists. Sort `ℓ₁ ≤ ℓ₂ ≤ ⋯`, set `sₖ ≝ Σ_{i<k} 2^{−ℓᵢ}`, and let
codeword `k` be the first `ℓₖ` bits of `sₖ`. Each `sₖ` is a multiple of
`2^{−ℓₖ}` (every earlier term is, since `ℓᵢ ≤ ℓₖ`), so `I(k) = [sₖ, sₖ₊₁)`
and these intervals are disjoint. Disjoint dyadic intervals give a prefix
code by the first step read backwards.

Third, the lower bound `L ≥ H`. Let `Z ≝ Σₓ 2^{−ℓ(x)} ≤ 1` and
`r(x) = 2^{−ℓ(x)} / Z`, a probability mass function,

  ```
  L − H(X) = Σₓ p(x) ⋅ ℓ(x) + Σₓ p(x) ⋅ log₂ p(x)
           = Σₓ p(x) ⋅ log₂( p(x) / 2^{−ℓ(x)} )
           = Σₓ p(x) ⋅ log₂( p(x) / r(x) ) − log₂ Z
           = 𝕂(p‖r) / ln 2 − log₂ Z
           ≥ 0
  ```

where the last step uses `𝕂(p‖r) ≥ 0` (Gibbs) and `log₂ Z ≤ 0` (Kraft).

Fourth, the Shannon code achieves `L < H + 1`. Take
`ℓ(x) = ⌈ −log₂ p(x) ⌉` for `p(x) > 0`. Kraft holds,

  ```
  Σₓ 2^{−ℓ(x)} ≤ Σₓ 2^{log₂ p(x)}
               = Σₓ p(x)
               = 1
  ```

so a prefix code with these lengths exists by the second step. Its expected
length is,

  ```
  L = Σₓ p(x) ⋅ ⌈ −log₂ p(x) ⌉
    < Σₓ p(x) ⋅ ( −log₂ p(x) + 1 )
    = H(X) + 1
  ```

using `⌈t⌉ < t + 1`.

Hence,

  ```
  H(X) ≤ min_{prefix c} L < H(X) + 1
  ```

`L = H` iff `Z = 1` and `p = r`, i.e., `p(x) = 2^{−ℓ(x)}` for every `x`:
the distribution is dyadic. Coding blocks of `n` iid symbols gives
`H ≤ L/n < H + 1/n` per symbol.

Regularity: finite `𝒳` (countable works with the same proof, provided
`H(X) < ∞`); symbols with `p(x) = 0` are dropped; binary code alphabet
(for `D` symbols replace `2` by `D` and `log₂` by `log_D`).
