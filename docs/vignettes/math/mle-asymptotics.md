# MLE Asymptotics

MLE asymptotics follow from three facts:

1. the true `θ₀` uniquely maximizes the expected log-likelihood (Gibbs),
2. the score at `θ₀` is a sum of iid zero-mean terms (CLT), and,
3. the score equation is linear in `θ̂ − θ₀` to first order (Taylor).

Let,

  ```
  Y₁, …, Yₙ iid ~ p(Y|θ₀),  θ₀ ∈ interior of Θ ⊆ ℝᵈ
  s(Y)  ≝ ∇ log p(Y|θ) at θ₀
  I     ≝ 𝔼_{θ₀}[ s(Y) s(Y)ᵀ ]
  ℓₙ(θ) ≝ (1/n) Σᵢ log p(Yᵢ|θ)
  M(θ)  ≝ 𝔼_{θ₀}[ log p(Y|θ) ]
  θ̂     ≝ argmax_θ ℓₙ(θ)
  ```

First, `θ₀` is the unique maximizer of `M` (Gibbs). Fix `θ`. Let `A` be the
support of `p(⋅|θ₀)`, and apply Jensen (jensen.md) to the concave `log`,

  ```
  M(θ) − M(θ₀) = 𝔼_{θ₀}[ log( p(Y|θ) / p(Y|θ₀) ) ]
               ≤ log 𝔼_{θ₀}[ p(Y|θ) / p(Y|θ₀) ]
               = log Σ_{y ∈ A} p(y|θ)
               ≤ log 1
               = 0
  ```

So `M(θ₀) − M(θ) = 𝕂(p(⋅|θ₀) ‖ p(⋅|θ)) ≥ 0`. Equality needs both
inequalities tight: strict concavity of `log` forces the ratio to be a
constant `c` on `A`, and the last step forces `Σ_{y∈A} p(y|θ) = 1`, so
`c = 1` and `p(⋅|θ) = p(⋅|θ₀)`. Identifiability (`p(⋅|θ) = p(⋅|θ₀)` only if
`θ = θ₀`) then gives `M(θ) < M(θ₀)` for every `θ ≠ θ₀`.

Second, consistency (sketch). Assume `Θ` compact and `M` continuous. Cite the
uniform law of large numbers,

  ```
  Δₙ ≝ sup_θ | ℓₙ(θ) − M(θ) | → 0   almost surely
  ```

Fix `ε > 0`. Compactness, continuity, and the first step give
`δ ≝ M(θ₀) − sup_{‖θ−θ₀‖≥ε} M(θ) > 0`. Then,

  ```
  M(θ̂) ≥ ℓₙ(θ̂) − Δₙ
       ≥ ℓₙ(θ₀) − Δₙ
       ≥ M(θ₀) − 2Δₙ
  ```

where the second step uses that `θ̂` maximizes `ℓₙ`. Once `2Δₙ < δ`,
`M(θ̂) > M(θ₀) − δ`, which forces `‖θ̂ − θ₀‖ < ε`. Hence `θ̂ → θ₀` almost
surely. Only the uniform LLN is cited.

Third, linearize the score equation. Since `θ₀` is interior and `θ̂ → θ₀`,
with probability tending to one `θ̂` is interior and `∇ℓₙ(θ̂) = 0`. Apply the
mean value theorem to each coordinate `j` of `∇ℓₙ` on the segment from `θ₀`
to `θ̂`,

  ```
  0 = ∂ⱼ ℓₙ(θ̂)
    = ∂ⱼ ℓₙ(θ₀) + ∇ ∂ⱼ ℓₙ(θ̃ⱼ)ᵀ (θ̂ − θ₀)
  ```

for some `θ̃ⱼ` on the segment. Stack the rows `∇ ∂ⱼ ℓₙ(θ̃ⱼ)ᵀ` into `Hₙ`,

  ```
  0 = ∇ℓₙ(θ₀) + Hₙ (θ̂ − θ₀)
  ```

Fourth, `Hₙ → −I` in probability. Let
`ω(r) ≝ 𝔼_{θ₀}[ sup_{‖θ−θ₀‖≤r} ‖∇² log p(Y|θ) − ∇² log p(Y|θ₀)‖ ]`. On the
event `‖θ̂ − θ₀‖ ≤ r`, every `θ̃ⱼ` lies in the ball, so

  ```
  ‖Hₙ + I‖ ≤ ‖Hₙ − ∇²ℓₙ(θ₀)‖ + ‖∇²ℓₙ(θ₀) + I‖
           ≤ d ⋅ (1/n) Σᵢ sup_{‖θ−θ₀‖≤r} ‖∇² log p(Yᵢ|θ) − ∇² log p(Yᵢ|θ₀)‖ + ‖∇²ℓₙ(θ₀) + I‖
           → d ⋅ ω(r) + ‖𝔼_{θ₀}[ ∇² log p(Y|θ₀) ] + I‖
           = d ⋅ ω(r)
  ```

where the factor `d` bounds a matrix by its `d` rows, the limit is the strong
LLN applied to both averages, and the last step is
`𝔼[∇² log p] = −I` (fisher-hessian.md). Continuity of `∇² log p` in `θ` and an
integrable envelope give `ω(r) → 0` as `r → 0` by dominated convergence.
Since `P(‖θ̂ − θ₀‖ ≤ r) → 1` for every `r > 0`, `Hₙ → −I` in probability.

Fifth, the CLT applies to the score. The terms `s(Yᵢ)` are iid with mean `0`
(cramer-rao.md, first step) and covariance `I`,

  ```
  √n ⋅ ∇ℓₙ(θ₀) = (1/√n) Σᵢ s(Yᵢ)
               → N(0, I)   in distribution
  ```

Sixth, solve and apply Slutsky. `I` is invertible, so `Hₙ` is invertible with
probability tending to one, and `Hₙ⁻¹ → −I⁻¹` in probability by continuity of
matrix inversion,

  ```
  √n (θ̂ − θ₀) = −Hₙ⁻¹ ⋅ √n ⋅ ∇ℓₙ(θ₀)
              → I⁻¹ ⋅ Z,   Z ~ N(0, I)
              ~ N(0, I⁻¹ I I⁻¹)
              = N(0, I⁻¹)
  ```

where the second step is Slutsky's lemma and the third uses
`𝕍(AZ) = A 𝕍(Z) Aᵀ` with symmetric `A = I⁻¹`.

Hence,

  ```
  θ̂ → θ₀ almost surely
  √n (θ̂ − θ₀) → N(0, I⁻¹)
  ```

The limit covariance attains the Cramér–Rao bound `I⁻¹ / n` (cramer-rao.md).
The leading term is `θ̂ − θ₀ ≈ I⁻¹ ⋅ (1/n) Σᵢ s(Yᵢ)`, affine in the score.

Regularity: the proof needs iid data, identifiability, `Θ` compact with `θ₀`
interior, `M` continuous and the uniform LLN for `ℓₙ` (cited; e.g., `log p`
continuous in `θ` with an integrable envelope `sup_θ |log p(Y|θ)|`),
`log p` twice continuously differentiable in `θ` with an integrable envelope
for `sup_{‖θ−θ₀‖≤r} ‖∇² log p(Y|θ)‖`, `∇` passing through `𝔼` twice (so the
score is zero mean and `𝔼[∇² log p] = −I`), and `I` finite and invertible.
The sketch uses `Σ` over discrete `Y`; densities replace it by `∫`.
