# Efron–Stein

Efron–Stein follows from three facts:

1. variance splits into the squared Doob martingale differences,
2. each difference is a conditional average of a one-coordinate fluctuation,
   so Jensen bounds it, and,
3. the variance of one coordinate equals half the mean squared gap between
   two independent draws.

Let,

  ```
  X    = (X₁, …, Xₙ)     independent
  X'ᵢ                    an independent copy of Xᵢ
  f    ≝ f(X),  𝔼[f²] < ∞
  f⁽ⁱ⁾ ≝ f(X₁, …, X'ᵢ, …, Xₙ)
  Fₖ   ≝ σ(X₁, …, Xₖ)
  Mₖ   ≝ 𝔼[f | Fₖ],   Dₖ ≝ Mₖ − Mₖ₋₁
  𝔼⁽ⁱ⁾f                  f averaged over Xᵢ only, others held fixed
  ```

First, the Doob differences are orthogonal, so variance splits. For `j < k`,
`Dⱼ` is `Fₖ₋₁`-measurable and `𝔼[Dₖ | Fₖ₋₁] = 0`, so `𝔼[DⱼDₖ] = 0`. Then,

  ```
  𝕍(f) = 𝔼[ (Mₙ − M₀)² ]
       = 𝔼[ (Σₖ Dₖ)² ]
       = Σₖ 𝔼[ Dₖ² ]
  ```

where the first step uses `Mₙ = f` and `M₀ = 𝔼f`.

Second, rewrite `Dₖ` with `𝔼⁽ᵏ⁾`. By independence, averaging over `Xₖ` and
then conditioning on `Fₖ` is the same as conditioning on `Fₖ₋₁`,

  ```
  Dₖ = 𝔼[ f | Fₖ ] − 𝔼[ f | Fₖ₋₁ ]
     = 𝔼[ f | Fₖ ] − 𝔼[ 𝔼⁽ᵏ⁾f | Fₖ ]
     = 𝔼[ f − 𝔼⁽ᵏ⁾f | Fₖ ]
  ```

Third, apply conditional Jensen (`jensen.md`) with `φ(u) = u²`,

  ```
  𝔼[ Dₖ² ] = 𝔼[ (𝔼[ f − 𝔼⁽ᵏ⁾f | Fₖ ])² ]
           ≤ 𝔼[ 𝔼[ (f − 𝔼⁽ᵏ⁾f)² | Fₖ ] ]
           = 𝔼[ (f − 𝔼⁽ᵏ⁾f)² ]
           = 𝔼[ 𝕍⁽ᵏ⁾(f) ]
  ```

where `𝕍⁽ᵏ⁾(f)` is the variance over `Xₖ` alone, and the third step is the
tower property (`law-of-total-expectation.md`).

Fourth, a variance is half the mean squared gap of two iid draws. With the
other coordinates fixed, `f` and `f⁽ᵏ⁾` are iid with mean `𝔼⁽ᵏ⁾f`,

  ```
  𝔼⁽ᵏ⁾[ (f − f⁽ᵏ⁾)² ] = 𝔼⁽ᵏ⁾[ (f − 𝔼⁽ᵏ⁾f)² ] + 𝔼⁽ᵏ⁾[ (f⁽ᵏ⁾ − 𝔼⁽ᵏ⁾f)² ] − 0
                      = 2 ⋅ 𝕍⁽ᵏ⁾(f)
  ```

where the cross term vanishes by independence of `Xₖ` and `X'ₖ`.

Hence,

  ```
  𝕍(f) ≤ Σᵢ 𝔼[ 𝕍⁽ⁱ⁾(f) ]
       = ½ Σᵢ 𝔼[ (f − f⁽ⁱ⁾)² ]
  ```

and, if changing coordinate `i` moves `f` by at most `cᵢ`, so
`|f − f⁽ⁱ⁾| ≤ cᵢ`,

  ```
  𝕍(f) ≤ ½ Σᵢ cᵢ²
  ```

Equality holds for additive `f = Σᵢ gᵢ(Xᵢ)`: then `f − 𝔼⁽ᵏ⁾f = gₖ − 𝔼gₖ` is
already `Fₖ`-measurable, so Jensen is tight and both sides equal
`Σᵢ 𝕍(gᵢ)`.

Fine print: equality holds only for additive `f` (Efron–Stein 1981, cited,
not proved). The bounded-difference corollary improves to `¼ Σᵢ cᵢ²`, since a
variable with range at most `cᵢ` has variance at most `cᵢ²/4`. McDiarmid
(`azuma-mcdiarmid.md`) turns the same hypothesis into a tail bound; Efron–Stein
gives only the second moment but needs no boundedness. The form
`Σᵢ 𝔼[𝕍⁽ⁱ⁾(f)]` is the tensorization of variance used in log-Sobolev and
Poincaré arguments.

Regularity: the proof needs `X₁, …, Xₙ` mutually independent (to swap the
`Xₖ`-average with conditioning on `Fₖ`) and `𝔼[f²] < ∞` (so all martingale
differences are square-integrable). No continuity of `f` is used.
