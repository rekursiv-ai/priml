# Fisher–Neyman Factorization

Factorization follows from one fact: the event `{Y = y}` is contained in
`{S(Y) = S(y)}`, so the joint law of `(Y, S)` is the law of `Y`.

Let,

  ```
  Y discrete,  Y ~ p(y|θ),  θ ∈ Θ
  S        ≝ S(Y)
  q(y|s,θ) ≝ p(Y=y|S=s,θ)   defined when p(S=s|θ) > 0
  ```

`S` is sufficient iff, for each `s`, `q(y|s,θ)` is the same for every `θ`
with `p(S=s|θ) > 0`. Call that common value `q(y|s)`.

The theorem: `S` is sufficient iff there exist `g ≥ 0` and `h ≥ 0` with

  ```
  p(y|θ) = g(S(y), θ) ⋅ h(y)   for all y, θ
  ```

First, the containment `{Y = y} ⊆ {S = S(y)}` gives, for `s = S(y)`,

  ```
  p(Y=y, S=s|θ) = p(Y=y|θ)
  p(S=s|θ)      = Σ_{y' : S(y')=s} p(Y=y'|θ)
  ```

and `p(Y=y, S=s|θ) = 0` when `S(y) ≠ s`.

Second, factorization implies sufficiency. Fix `s` and `θ` with
`p(S=s|θ) > 0`, and `y` with `S(y) = s`,

  ```
  q(y|s,θ) = p(Y=y, S=s|θ) / p(S=s|θ)
           = p(Y=y|θ) / Σ_{y' : S(y')=s} p(Y=y'|θ)
           = g(s,θ) h(y) / Σ_{y' : S(y')=s} g(s,θ) h(y')
           = h(y) / Σ_{y' : S(y')=s} h(y')
  ```

where the second step uses the first step, the third substitutes the
factorization with `S(y') = s`, and the last cancels `g(s,θ)`. The
cancellation is valid since `g(s,θ) ⋅ Σ h(y') = p(S=s|θ) ∈ (0, 1]`, so
`g(s,θ) > 0` and `Σ h(y')` is positive and finite. For `S(y) ≠ s`,
`q(y|s,θ) = 0`. Neither case depends on `θ`.

Third, sufficiency implies factorization. Define

  ```
  g(s,θ) ≝ p(S=s|θ)
  h(y)   ≝ q(y|S(y))   if p(S=S(y)|θ') > 0 for some θ'
  h(y)   ≝ 0           otherwise
  ```

Fix `y` and `θ`, and let `s ≝ S(y)`. If `p(S=s|θ) > 0`,

  ```
  p(y|θ) = p(Y=y, S=s|θ)
         = p(S=s|θ) ⋅ q(y|s,θ)
         = p(S=s|θ) ⋅ q(y|s)
         = g(S(y),θ) ⋅ h(y)
  ```

where the third step uses sufficiency. If `p(S=s|θ) = 0`,

  ```
  p(y|θ) ≤ p(S=s|θ)
         = 0
         = g(S(y),θ) ⋅ h(y)
  ```

where the first step uses the containment.

Hence,

  ```
  S sufficient  ⇔  p(y|θ) = g(S(y), θ) ⋅ h(y)
  ```

The factorization is not unique: `g ⋅ c(s)` and `h / c(S(y))` work for any
positive `c`. In words, `θ` touches the likelihood only through `S`.

Regularity: the proof needs `Y` discrete (countable support), so that
conditioning on `{S = s}` is elementary and sums may be split. The density
case, `p(y|θ) = g(S(y),θ) h(y)` with respect to a σ-finite dominating measure,
holds by the same statement but its proof needs Radon–Nikodym arguments
(Halmos–Savage), which are cited rather than proved here.
