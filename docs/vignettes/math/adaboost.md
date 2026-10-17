# AdaBoost Training Error

AdaBoost's training error bound follows from three facts:

1. the 0-1 loss is bounded by the exponential loss,
2. the exponential loss telescopes into the product of normalizers `Z_t`,
   and,
3. `α_t` is chosen to minimize `Z_t`, which gives `2√(ε_t(1 − ε_t))`.

Let,

  ```
  (xᵢ, yᵢ),  yᵢ ∈ {−1, +1},  i = 1..m
  D₁(i)      ≝ 1/m
  h_t: X → {−1, +1}                           weak hypothesis at round t
  ε_t        ≝ Σᵢ D_t(i) ⋅ 1[h_t(xᵢ) ≠ yᵢ],   γ_t ≝ ½ − ε_t
  α_t        ≝ ½ log((1 − ε_t) / ε_t)
  D_{t+1}(i) ≝ D_t(i) ⋅ exp(−α_t yᵢ h_t(xᵢ)) / Z_t
  Z_t        ≝ Σᵢ D_t(i) ⋅ exp(−α_t yᵢ h_t(xᵢ))
  F(x)       ≝ Σ_{t=1}^{T} α_t h_t(x),        H(x) ≝ sign F(x)
  ```

First, unroll the weights. By induction on `t`,

  ```
  D_{T+1}(i) = D₁(i) ⋅ Π_t exp(−α_t yᵢ h_t(xᵢ)) / Π_t Z_t
             = (1/m) ⋅ exp(−yᵢ F(xᵢ)) / Π_t Z_t
  ```

`D_{T+1}` sums to one, so `(1/m) Σᵢ exp(−yᵢ F(xᵢ)) = Π_t Z_t`.

Second, bound 0-1 loss by exponential loss. If `H(xᵢ) ≠ yᵢ` then
`yᵢ F(xᵢ) ≤ 0`, so `exp(−yᵢ F(xᵢ)) ≥ 1`; otherwise the exponential is
positive,

  ```
  (1/m) Σᵢ 1[H(xᵢ) ≠ yᵢ] ≤ (1/m) Σᵢ exp(−yᵢ F(xᵢ))
                         = Π_t Z_t
  ```

Third, `α_t` minimizes `Z_t`. Split by correct (`yh = +1`) and wrong
(`yh = −1`) points,

  ```
  Z_t(α)  ≝ (1 − ε_t) e^{−α} + ε_t e^{α}
  Z_t′(α) = −(1 − ε_t) e^{−α} + ε_t e^{α}
  ```

`Z_t` is convex, and `Z_t′(α) = 0` iff `e^{2α} = (1 − ε_t)/ε_t`, i.e.,
`α = α_t`. At `α_t`,

  ```
  Z_t(α_t) = (1 − ε_t) √(ε_t / (1 − ε_t)) + ε_t √((1 − ε_t) / ε_t)
           = √(ε_t (1 − ε_t)) + √(ε_t (1 − ε_t))
           = 2 √(ε_t (1 − ε_t))
  ```

Fourth, write in terms of the edge `γ_t`,

  ```
  2 √(ε_t (1 − ε_t)) = 2 √((½ − γ_t)(½ + γ_t))
                     = √(1 − 4γ_t²)
                     ≤ √(exp(−4γ_t²))
                     = exp(−2γ_t²)
  ```

where the third step uses `1 − x ≤ e^{−x}`.

Hence,

  ```
  (1/m) Σᵢ 1[H(xᵢ) ≠ yᵢ] ≤ Π_t 2 √(ε_t (1 − ε_t))
                         ≤ exp(−2 Σ_t γ_t²)
  ```

If every `γ_t ≥ γ > 0`, the training error is below `1/m`, hence zero, once
`T > log(m) / (2γ²)`.

The first and third steps are equalities. The slack is in the second step
(exponential loss over 0-1 loss) and in `1 − x ≤ e^{−x}`, which is tight as
`γ_t → 0`.

Fine print: `ε_t ∈ (0, 1)` is needed for `α_t` to be finite. `ε_t = 0` means
`h_t` is perfect and boosting can stop. `ε_t > ½` gives `α_t < 0`, which
flips `h_t`; the bound still holds because only `γ_t²` enters. Ties
`F(xᵢ) = 0` count as errors, which the second step covers. The bound is on
training error only; generalization needs a margin or VC argument. For
real-valued `h_t ∈ [−1, 1]` with edge `r_t = Σᵢ D_t(i) yᵢ h_t(xᵢ)`,
convexity of `exp` on `[−α, α]` and `α_t = ½ log((1 + r_t)/(1 − r_t))` give
`Z_t ≤ √(1 − r_t²)`, the same bound with `r_t = 2γ_t`.

Regularity: the proof needs `h_t` `{−1, +1}`-valued, `ε_t ∈ (0, 1)` for every
round, and `α_t` set as above (any other `α` gives the weaker
`error ≤ Π_t Z_t(α)`).
