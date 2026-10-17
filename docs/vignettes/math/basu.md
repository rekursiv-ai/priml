# Basu

Basu's theorem follows from two facts:

1. sufficiency makes `P(A ∈ B | S)` free of `θ`, and,
2. completeness forces that `θ`-free function to equal its constant mean.

Let,

  ```
  Y ~ p(Y|θ),  θ ∈ Θ
  S    ≝ S(Y)   complete sufficient
  A    ≝ A(Y)   ancillary: the law of A does not depend on θ
  B             any event for A;  C  any event for S
  π    ≝ P_θ(A ∈ B)           same for every θ
  η(S) ≝ P(A ∈ B | S)         free of θ by sufficiency
  ```

The claim: `S` and `A` are independent under every `θ`.

First, `η(S) − π` has zero mean under every `θ`, by the tower property,

  ```
  𝔼_θ[ η(S) − π ] = 𝔼_θ[ P(A ∈ B | S) ] − π
                  = P_θ(A ∈ B) − π
                  = 0
  ```

Completeness with `f(S) = η(S) − π` gives `η(S) = π` with `P_θ`-probability
one, for every `θ`.

Second, factor the joint probability,

  ```
  P_θ(A ∈ B, S ∈ C) = 𝔼_θ[ 1{S ∈ C} ⋅ P(A ∈ B | S) ]
                    = 𝔼_θ[ 1{S ∈ C} ⋅ η(S) ]
                    = 𝔼_θ[ 1{S ∈ C} ⋅ π ]
                    = P_θ(A ∈ B) ⋅ P_θ(S ∈ C)
  ```

where the first step conditions on `S` and pulls out the `S`-measurable
indicator.

Hence,

  ```
  S ⟂ A   under every θ
  ```

Example. Let `Y₁, …, Yₙ` be iid `N(μ, σ²)` with `σ²` known and `μ ∈ ℝ`
unknown,

  ```
  X̄  ≝ (1/n) Σᵢ Yᵢ
  S² ≝ (1/(n−1)) Σᵢ (Yᵢ − X̄)²
  ```

Third, `X̄` is sufficient. Expand the exponent around `X̄`,

  ```
  Σᵢ (yᵢ − μ)² = Σᵢ (yᵢ − ȳ + ȳ − μ)²
               = Σᵢ (yᵢ − ȳ)² + 2 (ȳ − μ) Σᵢ (yᵢ − ȳ) + n (ȳ − μ)²
               = Σᵢ (yᵢ − ȳ)² + n (ȳ − μ)²
  ```

since `Σᵢ (yᵢ − ȳ) = 0`. So `p(y|μ) = g(ȳ, μ) ⋅ h(y)` with
`g(ȳ, μ) ≝ exp(−n (ȳ − μ)² / (2σ²))`, and factorization.md applies (in its
density form).

Fourth, `X̄ ~ N(μ, σ²/n)` is complete. For any `f` with `𝔼_μ[f(X̄)] = 0` for
all `μ`,

  ```
  0 = 𝔼_μ[ f(X̄) ]
    = √(n/(2πσ²)) ∫ f(x) ⋅ exp(−n (x − μ)² / (2σ²)) dx
    = √(n/(2πσ²)) ⋅ exp(−n μ² / (2σ²)) ⋅ ∫ f(x) e^{−n x²/(2σ²)} ⋅ e^{(n μ/σ²) x} dx
  ```

so the two-sided Laplace transform of `f(x) e^{−n x²/(2σ²)}` vanishes for
every real argument `nμ/σ²`. Uniqueness of the Laplace transform (cited) gives
`f = 0` almost everywhere.

Fifth, `S²` is ancillary. Write `Yᵢ = μ + σ Zᵢ` with `Zᵢ` iid `N(0, 1)`,

  ```
  Yᵢ − X̄ = (μ + σ Zᵢ) − (μ + σ Z̄)
         = σ (Zᵢ − Z̄)
  S²     = σ² ⋅ (1/(n−1)) Σᵢ (Zᵢ − Z̄)²
  ```

whose law does not depend on `μ`.

Hence, by Basu, `X̄ ⟂ S²` under every `μ`. Since `σ²` was an arbitrary known
value, the independence holds for every `(μ, σ²)`.

Regularity: the proof needs `S` complete over the whole family, `S`
sufficient, and `A` a function of `Y` alone (so `η` is free of `θ` and has a
`θ`-free law). Ancillarity is only needed for `π` to be constant. The example
uses the density form of factorization and Laplace-transform uniqueness, both
cited.
