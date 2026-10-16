# Neyman–Pearson

Neyman–Pearson follows from one fact: the likelihood ratio test puts its
rejections exactly where `p₁ − k p₀` is positive, so `(φ* − φ)(p₁ − k p₀) ≥ 0`
pointwise.

Let,

  ```
  H₀: Y ~ p₀,   H₁: Y ~ p₁,   α ∈ (0, 1)
  φ(y) ∈ [0, 1]       a test: reject H₀ with probability φ(y)
  𝔼ⱼ[φ] ≝ ∫ φ(y) pⱼ(y) dy
  φ*(y) ≝ 1   if p₁(y) > k p₀(y)
          γ   if p₁(y) = k p₀(y)
          0   if p₁(y) < k p₀(y)
  ```

with `k ≥ 0` and `γ ∈ [0, 1]` chosen so `𝔼₀[φ*] = α`. Let `φ` be any test of
level `α`, i.e., `𝔼₀[φ] ≤ α`.

First, such `k ≥ 0` and `γ` exist. Let `a(c) ≝ P₀(p₁(Y) > c p₀(Y))`, which is
nonincreasing and right-continuous in `c`, with left limit
`a(c⁻) = P₀(p₁(Y) ≥ c p₀(Y))`. Under `P₀`, `p₀(Y) > 0` almost surely, so
`a(c) = 1` for `c < 0`. For `c > 0`,

  ```
  a(c) = ∫_{p₁ > c p₀} p₀(y) dy
       ≤ ∫_{p₁ > c p₀} p₁(y) / c dy
       ≤ 1 / c
  ```

so `a(c) → 0` as `c → ∞`. Set `k ≝ inf{c : a(c) ≤ α}`. Then `k ≥ 0`, since
`a(c) = 1 > α` for `c < 0`, and `a(k) ≤ α ≤ a(k⁻)` by right-continuity and
the infimum. Set `γ ≝ (α − a(k)) / (a(k⁻) − a(k))` if `a(k⁻) > a(k)`, else
`γ = 0`. Then,

  ```
  𝔼₀[φ*] = a(k) + γ ⋅ P₀(p₁(Y) = k p₀(Y))
         = a(k) + γ ⋅ (a(k⁻) − a(k))
         = α
  ```

Second, the integrand is nonnegative pointwise,

  ```
  p₁(y) > k p₀(y):  φ*(y) − φ(y) = 1 − φ(y) ≥ 0
  p₁(y) < k p₀(y):  φ*(y) − φ(y) = 0 − φ(y) ≤ 0
  p₁(y) = k p₀(y):  p₁(y) − k p₀(y) = 0
  ```

so `(φ*(y) − φ(y)) ⋅ (p₁(y) − k p₀(y)) ≥ 0` for every `y`.

Third, integrate,

  ```
  0 ≤ ∫ (φ*(y) − φ(y)) ⋅ (p₁(y) − k p₀(y)) dy
    = (𝔼₁[φ*] − 𝔼₁[φ]) − k ⋅ (𝔼₀[φ*] − 𝔼₀[φ])
    = (𝔼₁[φ*] − 𝔼₁[φ]) − k ⋅ (α − 𝔼₀[φ])
  ```

Rearrange,

  ```
  𝔼₁[φ*] − 𝔼₁[φ] ≥ k ⋅ (α − 𝔼₀[φ])
                 ≥ 0
  ```

where the last step uses `k ≥ 0` and `𝔼₀[φ] ≤ α`.

Hence,

  ```
  𝔼₁[φ*] ≥ 𝔼₁[φ]   for every φ with 𝔼₀[φ] ≤ α
  ```

i.e., the likelihood ratio test is most powerful at level `α`.

Equality forces the integral in the third step to be zero, so the
nonnegative integrand vanishes almost everywhere: `φ = φ*` almost everywhere
on `{p₁ ≠ k p₀}`. If `k > 0`, equality also forces `𝔼₀[φ] = α`.

Regularity: the proof needs `p₀`, `p₁` densities (or mass functions, with
`∫` read as `Σ`) with respect to a common measure, simple hypotheses, and
randomization `γ` on the boundary `{p₁ = k p₀}` when it has positive `P₀`
mass. Without randomization, the test is most powerful among tests of level
`𝔼₀[φ*]`.
