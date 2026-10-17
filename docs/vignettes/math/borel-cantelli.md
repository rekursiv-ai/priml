# Borel–Cantelli

Both lemmas follow from two facts:

1. the tail of a convergent series tends to zero, and,
2. `1 − x ≤ e^{−x}`, so a divergent sum drives a product of misses to zero.

Let,

  ```
  A₁, A₂, …   events
  limsup Aₙ ≝ ∩ₘ₌₁^∞ ∪ₙ₌ₘ^∞ Aₙ = { ω : ω ∈ Aₙ for infinitely many n }
  ```

First, Borel–Cantelli I: if `Σₙ P(Aₙ) < ∞`, then `P(limsup Aₙ) = 0`. For
every `m`, `limsup Aₙ ⊆ ∪ₙ₌ₘ^∞ Aₙ`, so,

  ```
  P(limsup Aₙ) ≤ P( ∪ₙ₌ₘ^∞ Aₙ )
               ≤ Σₙ₌ₘ^∞ P(Aₙ)
               → 0   as m → ∞
  ```

where the second step is the union bound (countable subadditivity), and the
last step uses that the series converges.

Second, Borel–Cantelli II: if the `Aₙ` are independent and
`Σₙ P(Aₙ) = ∞`, then `P(limsup Aₙ) = 1`. The complement is
`(limsup Aₙ)ᶜ = ∪ₘ ∩ₙ₌ₘ^∞ Aₙᶜ`. Fix `m` and `N ≥ m`,

  ```
  P( ∩ₙ₌ₘᴺ Aₙᶜ ) = Πₙ₌ₘᴺ (1 − P(Aₙ))
                 ≤ Πₙ₌ₘᴺ e^{−P(Aₙ)}
                 = exp( −Σₙ₌ₘᴺ P(Aₙ) )
                 → 0   as N → ∞
  ```

where the first step uses independence of the complements, the second uses
`1 − x ≤ e^{−x}`, and the last uses that the tail sum `Σₙ₌ₘ^∞ P(Aₙ)` is still
infinite.

Third, assemble. The sets `∩ₙ₌ₘᴺ Aₙᶜ` decrease in `N`, so by continuity from
above,

  ```
  P( (limsup Aₙ)ᶜ ) ≤ Σₘ P( ∩ₙ₌ₘ^∞ Aₙᶜ )
                    = Σₘ lim_{N→∞} P( ∩ₙ₌ₘᴺ Aₙᶜ )
                    = Σₘ 0
                    = 0
  ```

Hence,

  ```
  Σₙ P(Aₙ) < ∞                  ⟹  P(Aₙ infinitely often) = 0
  Σₙ P(Aₙ) = ∞, Aₙ independent  ⟹  P(Aₙ infinitely often) = 1
  ```

Independence cannot be dropped from the second lemma: with `Aₙ = A` for all
`n` and `P(A) = 1/2`, the sum diverges but `P(limsup Aₙ) = P(A) = 1/2`.
Pairwise independence suffices (Erdős–Rényi, cited, not proved).

Regularity: the first lemma needs only countable subadditivity. The second
needs mutual independence of the `Aₙ` (to factor the finite intersections)
and continuity of `P` from above.
