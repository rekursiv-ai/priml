# Sauer–Shelah and the VC Bound

The VC bound follows from three facts:

1. a class cannot realize more patterns than it shatters subsets,
2. a class of VC dimension `d` shatters only subsets of size at most `d`, and,
3. Massart's lemma turns a pattern count into a Rademacher bound.

Let,

  ```
  H                          class of functions h: X → {0, 1}, nonempty
  C        ≝ {c₁..cₙ} ⊂ X    n distinct points
  H_C      ≝ {(h(c₁)..h(cₙ)) : h ∈ H}     patterns realized on C
  H shatters B ⊆ C   ⟺    |H_B| = 2^{|B|}
  sh(H, C) ≝ |{B ⊆ C : H shatters B}|
  d        ≝ VC(H) ≝ max{|B| : H shatters B}
  Π_H(n)   ≝ max_{|C| = n} |H_C|          growth function
  ```

The empty set is shattered by any nonempty `H`.

First, prove `|H_C| ≤ sh(H, C)` by induction on `n`, for all `H` at once. For
`n = 1`: if `|H_C| = 2` then `∅` and `{c₁}` are shattered, so `sh = 2`; if
`|H_C| = 1` then `∅` is shattered, so `sh ≥ 1`. For the step, let
`C′ = {c₂..cₙ}` and split the patterns on `C` by their tail,

  ```
  Y₀ = {(y₂..yₙ) : (0, y₂..yₙ) ∈ H_C  or  (1, y₂..yₙ) ∈ H_C}
  Y₁ = {(y₂..yₙ) : (0, y₂..yₙ) ∈ H_C  and (1, y₂..yₙ) ∈ H_C}
  ```

Each tail in `Y₀` is counted once, and once more if it is in `Y₁`, so
`|H_C| = |Y₀| + |Y₁|`.

For `Y₀`, note `Y₀ = H_{C′}`. The induction hypothesis gives,

  ```
  |Y₀| = |H_{C′}|
       ≤ sh(H, C′)
       = |{B ⊆ C : c₁ ∉ B, H shatters B}|
  ```

For `Y₁`, let `H′ ≝ {h ∈ H : some g ∈ H has g = h on C′ and g(c₁) ≠ h(c₁)}`.
Then `Y₁ = H′_{C′}`. If `H′` shatters `B ⊆ C′`, then `H′` shatters `B ∪ {c₁}`,
because each pattern on `B` comes with both values at `c₁`. So `H` shatters
`B ∪ {c₁}` too, since `H′ ⊆ H`. The induction hypothesis on `H′` gives,

  ```
  |Y₁| = |H′_{C′}|
       ≤ sh(H′, C′)
       ≤ |{B ⊆ C′ : H shatters B ∪ {c₁}}|
       = |{B ⊆ C : c₁ ∈ B, H shatters B}|
  ```

(If `H′` is empty, `Y₁` is empty and the bound holds trivially.) Adding,

  ```
  |H_C| = |Y₀| + |Y₁|
        ≤ |{B ⊆ C : c₁ ∉ B, H shatters B}| + |{B ⊆ C : c₁ ∈ B, H shatters B}|
        = sh(H, C)
  ```

Second, every shattered `B` has `|B| ≤ d`, so count subsets of size at most
`d`,

  ```
  Π_H(n) ≤ max_{|C| = n} sh(H, C)
         ≤ Σ_{i=0}^{d} (n choose i)
  ```

This is the Sauer–Shelah lemma. Equality holds, e.g., for
`H = {indicators of sets of size ≤ d}`.

Third, for `n ≥ d ≥ 1`, bound the binomial sum,

  ```
  (d/n)^d ⋅ Σ_{i=0}^{d} (n choose i) ≤ Σ_{i=0}^{d} (n choose i) (d/n)^i
                                     ≤ Σ_{i=0}^{n} (n choose i) (d/n)^i
                                     = (1 + d/n)^n
                                     ≤ e^d
  ```

where the first step uses `d/n ≤ 1` so `(d/n)^d ≤ (d/n)^i` for `i ≤ d`, the
third is the binomial theorem, and the last uses `1 + x ≤ eˣ`. So,

  ```
  Π_H(n) ≤ (e n / d)^d
  ```

Fourth, combine with Massart (see massart.md). Take 0-1 loss
`ℓ(h, (x, y)) = 1[h(x) ≠ y]`. On a sample `S`, the loss vector
`(ℓ(h, zᵢ))ᵢ` is determined by the pattern `(h(xᵢ))ᵢ`, so the set `A` of loss
vectors has `|A| ≤ Π_H(n)`. Each loss vector is in `{0, 1}ⁿ`, so `r ≤ √n`,

  ```
  Rad̂_S(ℓ ∘ H) ≤ √n ⋅ √(2 log|A|) / n
               ≤ √( 2 log Π_H(n) / n )
               ≤ √( 2 d log(e n / d) / n )
  ```

The bound is uniform in `S`, so it also bounds `Radₙ = 𝔼_S Rad̂_S`.
Plugging into rademacher.md gives, with probability at least `1 − δ`, for
all `h ∈ H`,

  ```
  R(h) − R̂(h) ≤ 2 ⋅ Radₙ(ℓ ∘ H) + √( log(1/δ) / (2n) )
              ≤ 2 ⋅ √( 2 d log(e n / d) / n ) + √( log(1/δ) / (2n) )
  ```

The `log n` factor is an artifact of Massart. Chaining (Dudley) removes it,
giving `O(√(d/n))`, which matches the lower bound.

Regularity: the proof needs `H` binary-valued and nonempty, `d` finite with
`n ≥ d ≥ 1`, 0-1 loss for the pattern-to-loss step, and the regularity of
rademacher.md (iid sample, measurable sup).
