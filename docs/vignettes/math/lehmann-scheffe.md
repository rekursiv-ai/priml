# Lehmann–Scheffé

Lehmann–Scheffé follows from two facts:

1. Rao–Blackwell maps any unbiased estimator to a function of `S`, and,
2. completeness allows only one unbiased function of `S`.

Let,

  ```
  Y ~ p(Y|θ),  θ ∈ Θ
  𝔼_θ[⋅] ≝ 𝔼_{p(Y|θ)}[⋅]
  S      ≝ S(Y)   sufficient and complete:
             𝔼_θ[f(S)] = 0 ∀θ  ⇒  P_θ(f(S) = 0) = 1 ∀θ
  T      ≝ g(S)   𝔼_θ[T] = ψ(θ),  𝕍_θ(T) < ∞
  ```

And let `U(Y)` be any estimator unbiased for `ψ(θ)` with `𝕍_θ(U) < ∞`. Set
`U* ≝ 𝔼[U|S] = h(S)`, which is free of `θ` by sufficiency (rao-blackwell.md).

First, `g(S)` and `h(S)` are both unbiased, so their difference has zero mean
under every `θ`,

  ```
  𝔼_θ[ g(S) − h(S) ] = 𝔼_θ[T] − 𝔼_θ[U*]
                     = ψ(θ) − ψ(θ)
                     = 0
  ```

where `𝔼_θ[U*] = 𝔼_θ[U] = ψ(θ)` by the tower property. Completeness with
`f = g − h` gives `T = U*` with `P_θ`-probability one, for every `θ`.

Second, the law of total variance (law-of-total-variance.md) compares `U` with
`U*`,

  ```
  𝕍_θ(U) = 𝔼_θ[ 𝕍(U|S) ] + 𝕍_θ(U*)
         = 𝔼_θ[ 𝕍(U|S) ] + 𝕍_θ(T)
         ≥ 𝕍_θ(T)
  ```

where the second step uses `U* = T` almost surely.

Hence,

  ```
  𝕍_θ(T) ≤ 𝕍_θ(U)   for every θ and every unbiased U
  ```

so `T` is a uniformly minimum-variance unbiased estimator (UMVUE).

Third, `T` is the unique UMVUE. Suppose `U` is also a UMVUE, so
`𝕍_θ(U) = 𝕍_θ(T)`. The second step's chain then holds with equality,

  ```
  0 = 𝕍_θ(U) − 𝕍_θ(T)
    = 𝔼_θ[ 𝕍(U|S) ]
  ```

so `U = 𝔼[U|S] = U* = T` almost surely, under every `θ`.

Regularity: the proof needs `S` sufficient (so `U*` is an estimator), `S`
complete over the whole family `{p(Y|θ) : θ ∈ Θ}`, `T` a function of `S`, and
finite variances for `T` and every competitor `U`.
