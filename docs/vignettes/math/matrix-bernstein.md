# Matrix Bernstein

Matrix Bernstein follows from three facts:

1. the largest eigenvalue is bounded by a trace exponential (Laplace method),
2. Lieb's theorem makes matrix cumulant generating functions subadditive, and,
3. a bounded zero-mean matrix has a Bernstein-type mgf bound.

Let,

  ```
  X₁, …, Xₙ ∈ ℝᵈˣᵈ independent,  Xᵢᵀ = Xᵢ,  𝔼Xᵢ = 0,  ‖Xᵢ‖ ≤ L
  Y    ≝ Σᵢ Xᵢ
  V    ≝ Σᵢ 𝔼Xᵢ²,  σ² ≝ ‖V‖
  g(θ) ≝ (θ²/2) / (1 − θL/3),  0 < θ < 3/L
  ```

where `‖·‖` is the spectral norm and `⪯` is the semidefinite order.

First, the Laplace transform bound. For `θ > 0`, since `e^{θY} ≻ 0` and
`λ_max(e^{θY}) = e^{θ λ_max(Y)}`,

  ```
  P(λ_max(Y) ≥ t) = P(e^{θ λ_max(Y)} ≥ e^{θt})
                  ≤ e^{−θt} 𝔼 e^{θ λ_max(Y)}
                  = e^{−θt} 𝔼 λ_max(e^{θY})
                  ≤ e^{−θt} 𝔼 tr e^{θY}
  ```

where the second step is Markov and the last bounds the top eigenvalue of a
positive definite matrix by the sum of all of them.

Second, subadditivity of cgfs. Lieb's concavity theorem gives (Tropp 2012,
Lemma 3.4),

  ```
  𝔼 tr exp(Σᵢ θXᵢ) ≤ tr exp(Σᵢ log 𝔼 e^{θXᵢ})
  ```

Third, the scalar mgf kernel. By Taylor's theorem with integral remainder,

  ```
  f(x) = (e^{θx} − θx − 1) / x²
       = θ² ∫₀¹ (1 − s) e^{sθx} ds
  ```

which is increasing in `x`. Its value at `L` is bounded using
`k!/2 ≥ 3^{k−2}`,

  ```
  f(L) = Σ_{k≥2} θᵏ L^{k−2} / k!
       ≤ (θ²/2) Σ_{k≥2} (θL/3)^{k−2}
       = g(θ)
  ```

Fourth, the matrix mgf bound. The eigenvalues of `Xᵢ` are at most `L`, so
`f(Xᵢ) ⪯ f(L) I`, and conjugating by `Xᵢ` preserves `⪯`. Hence

  ```
  𝔼 e^{θXᵢ} = 𝔼[ I + θXᵢ + Xᵢ f(Xᵢ) Xᵢ ]
            ⪯ I + θ 𝔼Xᵢ + f(L) 𝔼Xᵢ²
            = I + f(L) 𝔼Xᵢ²
            ⪯ exp(g(θ) 𝔼Xᵢ²)
  ```

where the third step uses `𝔼Xᵢ = 0` and the last uses `1 + a ≤ eᵃ`
eigenvalue-wise with `f(L) ≤ g(θ)`. Since `log` is operator monotone,

  ```
  log 𝔼 e^{θXᵢ} ⪯ g(θ) 𝔼Xᵢ²
  ```

Fifth, combine. `A ⪯ B` implies `λⱼ(A) ≤ λⱼ(B)` for every `j` (see
courant-fischer.md), so `tr e^A ≤ tr e^B`. Chaining the first, second, and
fourth steps,

  ```
  P(λ_max(Y) ≥ t) ≤ e^{−θt} tr exp(Σᵢ log 𝔼 e^{θXᵢ})
                  ≤ e^{−θt} tr exp(g(θ) V)
                  ≤ e^{−θt} d λ_max(exp(g(θ) V))
                  = d exp(−θt + g(θ) σ²)
  ```

where the last step uses `V ⪰ 0`, so `λ_max(V) = ‖V‖ = σ²`.

Sixth, optimize `θ`. Choose `θ = t / (σ² + Lt/3)`, so that
`1 − θL/3 = σ² / (σ² + Lt/3)` and

  ```
  −θt + g(θ) σ² = −θt + (θ²/2)(σ² + Lt/3)
                = −t² / (σ² + Lt/3) + t² / (2(σ² + Lt/3))
                = −t² / (2(σ² + Lt/3))
  ```

Seventh, both tails. The `−Xᵢ` satisfy the same hypotheses with the same `L`
and `V`, and `‖Y‖ = max(λ_max(Y), λ_max(−Y))`. A union bound over the two
tails doubles the bound.

Hence,

  ```
  P(‖Σᵢ Xᵢ‖ ≥ t) ≤ 2d ⋅ exp( −(t²/2) / (σ² + Lt/3) )
  ```

For small `t` the bound is a Gaussian tail with variance `σ²`; for large `t`
it is exponential at scale `L`. The dimensional factor is needed: for diagonal
`Xᵢ` with independent Rademacher entries, `σ² = n` and `‖Y‖` is the max of `d`
independent sums, which grows like `√(2n log d)`.

Fine print: operator monotonicity of `log` follows from
`log a = ∫₀^∞ (1/(1+s) − 1/(a+s)) ds` and `A ⪯ B ⇒ (B+sI)⁻¹ ⪯ (A+sI)⁻¹`. The
latter holds because `B^{−1/2} A B^{−1/2} ⪯ I` inverts to
`B^{1/2} A⁻¹ B^{1/2} ⪰ I`. The choice of `θ` satisfies `θL < 3` iff
`σ² > 0`; if `σ² = 0` then `Y = 0` almost surely. The fourth step uses only
`λ_max(Xᵢ) ≤ L`; the seventh uses `λ_min(Xᵢ) ≥ −L`. Rectangular `Xᵢ` reduce to
this case via the Hermitian dilation `[[0, X],[Xᵀ, 0]]` with `d₁ + d₂` in
place of `d`.

Regularity: the proof needs independence (for Lieb's corollary), `𝔼Xᵢ = 0`,
`‖Xᵢ‖ ≤ L` almost surely, symmetric `Xᵢ`, and Lieb's concavity theorem, which
is cited rather than proved.
