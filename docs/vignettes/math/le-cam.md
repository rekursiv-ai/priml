# Le Cam's Two-Point Method

Le Cam's bound follows from two facts:

1. an estimator close to one of two separated points yields a test, and,
2. no test can tell apart two laws that are close in total variation.

Let,

  ```
  θ₀, θ₁ ∈ Θ,   d a metric on Θ,   d(θ₀, θ₁) ≥ Δ > 0
  Pⱼ       ≝ law of one sample under θⱼ,   Pⱼⁿ ≝ law of Y = (Y₁, …, Yₙ)
  𝔼ⱼ       ≝ 𝔼_{Pⱼⁿ}
  TV(P, Q) ≝ sup_A |P(A) − Q(A)|
  θ̂        ≝ θ̂(Y)   any estimator
  ψ(Y)     ≝ 0 if d(θ̂, θ₀) ≤ d(θ̂, θ₁), else 1
  ```

First, an error of the test `ψ` forces a large estimation error. If
`ψ ≠ j`, then `d(θ̂, θⱼ) ≥ d(θ̂, θ₁₋ⱼ)`, so

  ```
  Δ ≤ d(θ₀, θ₁)
    ≤ d(θ̂, θ₀) + d(θ̂, θ₁)
    ≤ 2 ⋅ d(θ̂, θⱼ)
  ```

by the triangle inequality. So `d(θ̂, θⱼ) ≥ (Δ/2) ⋅ 1{ψ ≠ j}`, and,

  ```
  𝔼ⱼ[ d(θ̂, θⱼ) ] ≥ (Δ/2) ⋅ Pⱼⁿ(ψ ≠ j)
  ```

Second, the max is at least the average,

  ```
  maxⱼ 𝔼ⱼ[ d(θ̂, θⱼ) ] ≥ ½ ( 𝔼₀[ d(θ̂, θ₀) ] + 𝔼₁[ d(θ̂, θ₁) ] )
                      ≥ (Δ/4) ( P₀ⁿ(ψ = 1) + P₁ⁿ(ψ = 0) )
  ```

Third, any test's error sum is at least `1 − TV`. Let `A ≝ {ψ = 1}`,

  ```
  P₀ⁿ(A) + P₁ⁿ(Aᶜ) = 1 − ( P₁ⁿ(A) − P₀ⁿ(A) )
                   ≥ 1 − TV(P₀ⁿ, P₁ⁿ)
  ```

Hence, since `θ̂` was arbitrary,

  ```
  inf_θ̂ maxⱼ 𝔼ⱼ[ d(θ̂, θⱼ) ] ≥ (Δ/4) ⋅ ( 1 − TV(P₀ⁿ, P₁ⁿ) )
  ```

The third step is tight: the likelihood ratio test `A = {p₁ⁿ > p₀ⁿ}`
attains `1 − TV`. So the bound is as strong as the two-point reduction
allows.

Gaussian mean. Let `Pⱼ = N(θⱼ, σ²)` on `ℝ`, `d(a, b) = |a − b|`, and
`θ₁ − θ₀ = Δ`.

Fourth, the one-sample KL,

  ```
  𝕂(P₀‖P₁) = 𝔼_{P₀}[ ( (Y − θ₁)² − (Y − θ₀)² ) / (2σ²) ]
           = ( σ² + Δ² − σ² ) / (2σ²)
           = Δ² / (2σ²)
  ```

using `𝔼_{P₀}[(Y − θ₁)²] = σ² + Δ²`.

Fifth, KL tensorizes over iid samples by the chain rule (kl-chain-rule.md),
and Pinsker (pinsker.md) converts it to TV,

  ```
  TV(P₀ⁿ, P₁ⁿ) ≤ √( 𝕂(P₀ⁿ‖P₁ⁿ) / 2 )
               = √( n ⋅ 𝕂(P₀‖P₁) / 2 )
               = √( n Δ² / (4σ²) )
               = Δ √n / (2σ)
  ```

Sixth, choose `Δ = σ/√n`, so `TV ≤ ½`,

  ```
  inf_θ̂ sup_θ 𝔼_θ[ |θ̂ − θ| ] ≥ inf_θ̂ maxⱼ 𝔼ⱼ[ |θ̂ − θⱼ| ]
                             ≥ (σ / (4√n)) ⋅ (1 − ½)
                             = σ / (8√n)
  ```

So no estimator beats the `1/√n` rate. The sample mean attains it, with
`𝔼|X̄ − θ| = σ √(2/π) / √n ≈ 0.80 σ/√n`. Only the constant is loose.

Fine print: `d` need only satisfy the triangle inequality, which is all the
first step uses. For squared loss, the first step gives
`𝔼ⱼ[d(θ̂, θⱼ)²] ≥ (Δ²/4) ⋅ Pⱼⁿ(ψ ≠ j)`, so the bound becomes
`(Δ²/8)(1 − TV)`. For the Gaussian mean that is `σ²/(16 n)`. The
Bretagnolle–Huber inequality `P₀(A) + P₁(Aᶜ) ≥ ½ e^{−𝕂}` can replace
Pinsker when `𝕂` is large. Two points give rates for a single scalar
functional. Many-point versions (Assouad, Fano: fano.md) give dimension
factors.

Regularity: the proof needs `d` to obey the triangle inequality,
`d(θ₀, θ₁) ≥ Δ`, measurable `θ̂`, and, for the Gaussian example, iid samples
with known `σ²`, the KL chain rule, and Pinsker.
