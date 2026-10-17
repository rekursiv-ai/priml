# ε-Nets and the Gaussian Operator Norm

The operator-norm bound follows from three facts:

1. disjoint small balls around a separated set fit in a slightly larger
   ball, so the sphere has an ε-net of size `(1 + 2/ε)ⁿ`,
2. a sup over the sphere is controlled by a max over the net, and,
3. a fixed bilinear form of a Gaussian matrix is `N(0, 1)`, so a union bound
   over the nets costs only `(n + m) log 9`.

Let,

  ```
  Sⁿ⁻¹ ≝ { x ∈ ℝⁿ : ‖x‖ = 1 }
  N ⊆ Sⁿ⁻¹ an ε-net: every x ∈ Sⁿ⁻¹ has some y ∈ N with ‖x − y‖ ≤ ε
  B(r) ≝ the closed ball of radius r at 0,  vol B(r) = rⁿ vol B(1)
  A ∈ ℝ^{m×n},   ‖A‖ ≝ max_{x ∈ Sⁿ⁻¹, y ∈ Sᵐ⁻¹} ⟨Ax, y⟩
  0    < ε < 1/2
  ```

First, a net of size `(1 + 2/ε)ⁿ` exists. Let `N` be a maximal subset of
`Sⁿ⁻¹` with pairwise distances `> ε`. By maximality it is an ε-net. The
balls `y + B(ε/2)` for `y ∈ N` are disjoint and lie inside `B(1 + ε/2)`,

  ```
  |N| ⋅ (ε/2)ⁿ vol B(1) = vol ∪_{y∈N} (y + B(ε/2))
                        ≤ vol B(1 + ε/2)
                        = (1 + ε/2)ⁿ vol B(1)
  ```

so `|N| ≤ (1 + 2/ε)ⁿ`. The same bound caps every separated set, so a maximal
one exists.

Second, the net controls the norm. Let `N, M` be ε-nets of `Sⁿ⁻¹, Sᵐ⁻¹`,
let `(x*, y*)` attain `‖A‖`, and pick `x ∈ N`, `y ∈ M` within `ε`,

  ```
  ‖A‖ = ⟨Ax*, y*⟩
      = ⟨Ax, y⟩ + ⟨A(x* − x), y*⟩ + ⟨Ax, y* − y⟩
      ≤ max_{N×M} ⟨Ax, y⟩ + ε‖A‖ + ε‖A‖
  ```

where the last step uses `⟨Av, w⟩ ≤ ‖A‖ ‖v‖ ‖w‖`. Rearranging,
`‖A‖ ≤ (1 − 2ε)⁻¹ max_{N×M} ⟨Ax, y⟩`. With `ε = 1/4`,

  ```
  ‖A‖ ≤ 2 ⋅ max_{N×M} ⟨Ax, y⟩,     |N| ≤ 9ⁿ,  |M| ≤ 9ᵐ
  ```

Third, let `A` have iid `N(0, 1)` entries. For fixed unit `x, y`,
`⟨Ax, y⟩ = Σᵢⱼ Aᵢⱼ yᵢ xⱼ ~ N(0, Σᵢⱼ yᵢ²xⱼ²) = N(0, 1)`. The Gaussian Chernoff
bound (`markov-chebyshev-chernoff.md`) and a union bound give, for `u > 0`,

  ```
  P( max_{N×M} ⟨Ax, y⟩ ≥ u ) ≤ |N| |M| e^{−u²/2}
                             ≤ 9ⁿ⁺ᵐ e^{−u²/2}
                             = e^{−s}
  ```

where the last line sets `u² = 2(n + m) log 9 + 2s`, for `s > 0`.

Fourth, convert to a norm bound. On the complement event,

  ```
  ‖A‖ ≤ 2u
      ≤ 2√(2 log 9) ⋅ √(n + m) + 2√(2s)
      ≤ 2√(2 log 9) ⋅ (√m + √n) + 2√(2s)
  ```

where the second and third steps each use `√(a + b) ≤ √a + √b`.

Hence, with probability at least `1 − e^{−s}`,

  ```
  ‖A‖ ≤ C (√m + √n) + 2√(2s),     C ≝ 2√(2 log 9) ≈ 4.19
  ```

The order is sharp: `‖A‖ ≥ ‖Ae₁‖`, a column norm near `√m`, and
`‖A‖ ≥ ‖Aᵀe₁‖`, a row norm near `√n`, so `‖A‖ ≳ max(√m, √n)`.

Fine print: the one-sided net bound `‖A‖ ≤ (1 − ε)⁻¹ max_{x∈N} ‖Ax‖` follows
the same way from `‖Ax*‖ ≤ ‖Ax‖ + ε‖A‖`, but `‖Ax‖` is `χₘ`, not Gaussian, so
the bilinear form is simpler. The sharp constant `𝔼‖A‖ ≤ √m + √n` comes from
Gordon's (Slepian-type) comparison inequality, cited, not proved. Since
`A ↦ ‖A‖` is `1`-Lipschitz in Frobenius norm, `gaussian-lipschitz.md` gives
deviations of order `1` around `𝔼‖A‖`. For bounded independent entries,
replace the Gaussian tail of `⟨Ax, y⟩` by Hoeffding (`hoeffding.md`); only
the constant changes.

Regularity: the proof needs the nets to exist (volume argument in finite
dimension), `ε < 1/2` (so `1 − 2ε > 0`), and iid standard Gaussian entries
(so each fixed bilinear form is exactly `N(0, 1)`). Attainment of `‖A‖` uses
compactness of the spheres.
