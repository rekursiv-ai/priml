# Approximation–Estimation Decomposition

The decomposition is adding and subtracting the best-in-class risk. The ERM
bound follows from three facts:

1. `R` and `R̂` differ by at most `sup_h |R(h) − R̂(h)|` at any `h`,
2. ERM makes `R̂(ĥ) ≤ R̂(h)` for every `h ∈ H`, and,
3. the near-best `h_η` exists for every `η > 0`.

Let,

  ```
  R(h) ≝ 𝔼_{z~D}[ ℓ(h, z) ]               population risk
  R̂(h) ≝ (1/n) Σᵢ ℓ(h, zᵢ)                 empirical risk on S ≝ (z₁..zₙ)
  R*   ≝ inf_{all measurable h} R(h)      Bayes risk
  R_H  ≝ inf_{h ∈ H} R(h)                 best-in-class risk
  ĥ ∈ argmin_{h ∈ H} R̂(h)                 ERM
  Δ    ≝ sup_{h ∈ H} |R(h) − R̂(h)|
  ```

First, add and subtract `R_H`,

  ```
  R(ĥ) − R* = (R(ĥ) − R_H) + (R_H − R*)
            = estimation    + approximation
  ```

Both terms are nonnegative: `ĥ ∈ H` gives `R(ĥ) ≥ R_H`, and `H` being a
subset of all measurable `h` gives `R_H ≥ R*`. The approximation term depends only on
`H` and `D`; the estimation term is random through `S`.

Second, fix `η > 0` and pick `h_η ∈ H` with `R(h_η) ≤ R_H + η`. Then,

  ```
  R(ĥ) − R_H ≤ R(ĥ) − R(h_η) + η
             = (R(ĥ) − R̂(ĥ)) + (R̂(ĥ) − R̂(h_η)) + (R̂(h_η) − R(h_η)) + η
             ≤ Δ             + 0                 + Δ                 + η
             = 2Δ + η
  ```

where the second step adds and subtracts `R̂(ĥ)` and `R̂(h_η)`, and the third
uses the definition of `Δ` for the outer terms and ERM for the middle term.

Third, `η > 0` was arbitrary, so let `η → 0`.

Hence,

  ```
  R(ĥ) − R*  = (R(ĥ) − R_H) + (R_H − R*)
  R(ĥ) − R_H ≤ 2 ⋅ sup_{h ∈ H} |R(h) − R̂(h)|
  ```

The bound only uses one-sided deviations: `R − R̂` at `ĥ` and `R̂ − R` at
`h_η`. Equality needs `R̂(ĥ) = R̂(h_η)` and both deviations to hit `Δ` with
opposite signs, so it is typically loose by a constant. Growing `H` shrinks the
approximation term and grows `Δ`; this is the tradeoff.

Regularity: the proof needs `R(h)` finite for every `h ∈ H`, `ĥ` an exact
minimizer of `R̂` over `H` (an `ε`-minimizer adds `ε`), and `Δ` measurable
(e.g., `H` countable or pointwise separable). No iid assumption is used here;
iid enters only when bounding `Δ`.
