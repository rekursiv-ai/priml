# Rao–Blackwell

Rao–Blackwell follows from two facts:

1. sufficiency makes `𝔼[T|S]` free of `θ`, and,
2. conditioning never increases variance.

Let,

  ```
  Y  ~ p(Y|θ)
  𝔼_θ[⋅] ≝ 𝔼_{p(Y|θ)}[⋅]
  S      ≝ S(Y)      sufficient: p(Y|S,θ) = p(Y|S)
  T      ≝ T(Y)      𝔼_θ[T] = ψ(θ),  𝕍_θ(T) < ∞
  T*     ≝ 𝔼[ T(Y) | S ]
  ```

First, `T*` is an estimator, since sufficiency removes `θ` from the
conditional law of `Y` given `S`,

  ```
  T*(s) = Σ_y T(y) ⋅ p(Y=y|S=s,θ)
        = Σ_y T(y) ⋅ p(Y=y|S=s)
  ```

which is a function of `s` alone. The same holds for densities, replacing
`Σ_y` by `∫ dy`.

Second, `T*` has the same mean, by the tower property,

  ```
  𝔼_θ[T*] = 𝔼_θ[ 𝔼[T|S] ]
          = 𝔼_θ[T]
          = ψ(θ)
  ```

Third, apply the law of total variance (law-of-total-variance.md),

  ```
  𝕍_θ(T) = 𝔼_θ[ 𝕍(T|S) ] + 𝕍_θ( 𝔼[T|S] )
         = 𝔼_θ[ 𝕍(T|S) ] + 𝕍_θ(T*)
         ≥ 𝕍_θ(T*)
  ```

where the last step uses `𝕍(T|S) ≥ 0`.

Hence,

  ```
  𝕍_θ(T*) ≤ 𝕍_θ(T)   for every θ
  ```

Equality holds iff `𝔼_θ[𝕍(T|S)] = 0`, i.e., `T = T*` almost surely. In
words, `T` was already a function of `S`.

The same holds for any loss `L(ψ, t)` convex in `t`, with risk
`R(θ, T) ≝ 𝔼_θ[ L(ψ(θ), T) ]`. Unbiasedness is not needed. Fourth, apply
conditional Jensen (jensen.md) inside the outer expectation,

  ```
  R(θ, T*) = 𝔼_θ[ L(ψ(θ), 𝔼[T|S]) ]
           ≤ 𝔼_θ[ 𝔼[ L(ψ(θ), T) | S ] ]
           = 𝔼_θ[ L(ψ(θ), T) ]
           = R(θ, T)
  ```

where the second step uses that `ψ(θ)` is a constant given `S`, and the third
step is the tower property. Squared loss `L(ψ, t) = (t − ψ)²` recovers the
variance statement, since `T` and `T*` share the mean `ψ(θ)`. If `L` is
strictly convex in `t` and `R(θ, T) < ∞`, equality again forces `T = T*`
almost surely.

Regularity: the proof needs `S` sufficient, `𝔼_θ|T| < ∞` so `T*` exists,
`𝕍_θ(T) < ∞` for the variance form, and, for the loss form, `L(ψ, ⋅)` convex
with `𝔼_θ|L(ψ(θ), T)| < ∞`.
