# PAC-Bayes Bound

The bound follows from three facts:

1. Donsker–Varadhan (see donsker-varadhan.md) trades any posterior `Q` for the
   prior `P` at a cost of `𝕂(Q‖P)`,
2. Markov's inequality controls the prior-side term once, for all `Q`, and,
3. the squared deviation `(R − R̂)²` has a bounded exponential moment.

Let,

  ```
  H                          hypothesis space
  P                          prior on H, chosen before seeing S
  Q                          any posterior on H, may depend on S
  ℓ(h, z) ∈ [0, 1]           bounded loss
  S    ≝ (z₁..zₙ) ~ Dⁿ       iid sample
  R(h) ≝ 𝔼_{z~D}[ ℓ(h, z) ],     R̂(h) ≝ (1/n) Σᵢ ℓ(h, zᵢ)
  Δ(h) ≝ R(h) − R̂(h)
  M(λ) ≝ sup_{h} 𝔼_S[ exp(λ ⋅ Δ(h)²) ]
  ```

The proof below works for any `λ > 0` with `M(λ) < ∞`. Two values are used:

  ```
  M(2n)       ≤ 2√n          Maurer (2004), n ≥ 8, cited
  M(2(n − 1)) ≤ 2n − 1       proved in the Fourth step, n ≥ 2
  ```

First, the prior-side moment is small on average. `P` does not depend on `S`,
so Fubini swaps the expectations,

  ```
  𝔼_S 𝔼_{h~P}[ exp(λ Δ(h)²) ] = 𝔼_{h~P} 𝔼_S[ exp(λ Δ(h)²) ]
                              ≤ 𝔼_{h~P}[ M(λ) ]
                              = M(λ)
  ```

Second, Markov's inequality on the nonnegative variable
`X(S) ≝ 𝔼_{h~P}[exp(λ Δ(h)²)]` gives,

  ```
  P_S( X(S) ≥ M(λ)/δ ) ≤ 𝔼_S[ X(S) ] ⋅ δ / M(λ)
                       ≤ δ
  ```

So the event `E ≝ {X(S) < M(λ)/δ}` has probability at least `1 − δ`. The
event `E` does not mention `Q`.

Third, on `E`, apply Donsker–Varadhan with `g(h) ≝ λ Δ(h)²`, then Jensen. For
every `Q ≪ P` simultaneously,

  ```
  λ ⋅ (𝔼_Q Δ)² ≤ λ ⋅ 𝔼_Q[ Δ² ]
               = 𝔼_Q[ λ Δ² ]
               ≤ 𝕂(Q‖P) + log 𝔼_P[ exp(λ Δ²) ]
               ≤ 𝕂(Q‖P) + log(M(λ)/δ)
  ```

where the first step is Jensen (see jensen.md) for convex `x ↦ x²`, the third
is Donsker–Varadhan, and the fourth uses the event `E`. If `Q` is not
`≪ P`, then `𝕂(Q‖P) = ∞` and the bound is vacuous. Taking the square root,
and using `𝔼_Q Δ = 𝔼_Q R − 𝔼_Q R̂`,

  ```
  𝔼_Q R ≤ 𝔼_Q R̂ + √( (𝕂(Q‖P) + log(M(λ)/δ)) / λ )
  ```

Fourth, prove `M(2(n − 1)) ≤ 2n − 1` from Hoeffding. Fix `h`. Two-sided
Hoeffding (see hoeffding.md) gives `P(Δ² > s) = P(|Δ| > √s) ≤ 2 e^{−2ns}`.
For nonnegative `X` and increasing `g` with `g(0) = 1`,
`𝔼 g(X) = 1 + ∫₀^∞ g′(s) P(X > s) ds`. With `g(s) = e^{λs}` and `λ < 2n`,

  ```
  𝔼_S[ exp(λ Δ²) ] = 1 + ∫₀^∞ λ e^{λs} ⋅ P(Δ² > s) ds
                   ≤ 1 + ∫₀^∞ λ e^{λs} ⋅ 2 e^{−2ns} ds
                   = 1 + 2λ ∫₀^∞ e^{−(2n − λ)s} ds
                   = 1 + 2λ / (2n − λ)
  ```

At `λ = 2(n − 1)`, `2n − λ = 2`, so the bound is `1 + 2(n − 1) = 2n − 1`.

Hence, with probability at least `1 − δ` over `S`, for all `Q` simultaneously,
using Maurer's `M(2n) ≤ 2√n` (cited, `n ≥ 8`),

  ```
  𝔼_Q R ≤ 𝔼_Q R̂ + √( (𝕂(Q‖P) + log(2√n/δ)) / (2n) )
  ```

and, using only the Hoeffding-based `M(2(n − 1)) ≤ 2n − 1` proved here
(`n ≥ 2`),

  ```
  𝔼_Q R ≤ 𝔼_Q R̂ + √( (𝕂(Q‖P) + log((2n − 1)/δ)) / (2(n − 1)) )
  ```

Maurer's bound is not proved here. It follows from
`𝔼_S exp(n ⋅ kl(R̂(h) ‖ R(h))) ≤ 2√n`, which reduces to the Bernoulli case by
convexity and is then a direct binomial computation, combined with Pinsker's
`kl(q ‖ p) ≥ 2(q − p)²`. The Hoeffding route loses only `log n` versus
`log √n` in the numerator and `n − 1` versus `n` in the denominator.

The `Q`-dependent steps are equalities when `Δ` is constant on the support of
`Q` (Jensen) and `dQ/dP ∝ exp(λ Δ²)` (Donsker–Varadhan). That `Q` depends on
the unknown `R`, so the slack in practice sits in Markov and in `M(λ)`.

Regularity: the proof needs `S` iid, `ℓ ∈ [0, 1]`, `P` independent of `S`,
`Δ(h)` jointly measurable in `(h, S)` for Fubini, and `n ≥ 8` (Maurer) or
`n ≥ 2` (Hoeffding route). `Q` may depend on `S` arbitrarily, since the event
`E` is chosen before `Q`.
