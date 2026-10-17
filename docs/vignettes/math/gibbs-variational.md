# Gibbs Variational Principle

The Gibbs variational principle follows from one fact: for any `P`, the gap
between `log 𝔼_Q[e^f]` and `𝔼_P[f] − 𝕂(P‖Q)` is `𝕂(P‖G)` for the Gibbs
measure `dG ∝ e^f dQ`, which is nonnegative (gibbs-inequality.md).

Let,

  ```
  Z    ≝ 𝔼_Q[ e^f ]                           0 < Z < ∞
  dG   ≝ ( e^f / Z ) dQ                       Gibbs (tilted) measure
  F(P) ≝ 𝔼_P[f] − 𝕂(P‖Q)
  ```

and let `P` range over probability measures with `𝔼_P[|f|] < ∞`.

First, if `𝕂(P‖Q) = ∞`, then `F(P) = −∞` and the bound below holds
trivially. Otherwise `P ≪ Q`, hence `P ≪ G` since `dG/dQ = e^f/Z > 0`, and
the density ratios chain,

  ```
  log( dP/dG ) = log( dP/dQ ) − log( dG/dQ )
               = log( dP/dQ ) − f + log Z
  ```

Second, take `𝔼_P` of both sides,

  ```
  𝕂(P‖G) = 𝕂(P‖Q) − 𝔼_P[f] + log Z
         = log Z − F(P)
  ```

and since `𝕂(P‖G) ≥ 0`,

  ```
  F(P) = log Z − 𝕂(P‖G)
       ≤ log Z
  ```

Third, the bound is attained at `P = G`, where `log(dG/dQ) = f − log Z`,

  ```
  F(G) = 𝔼_G[f] − 𝔼_G[ f − log Z ]
       = log Z
  ```

Hence,

  ```
  log 𝔼_Q[ e^f ] = sup_P { 𝔼_P[f] − 𝕂(P‖Q) },     P* = G,  dG ∝ e^f dQ
  ```

The gap at any `P` is exactly `𝕂(P‖G)`, so the maximizer is unique:
`P* = G`. With `Q` uniform on a finite set, `G = softmax(f)` and `log Z` is
`logsumexp(f)` minus `log |𝒴|`.

This is the convex dual of Donsker–Varadhan (donsker-varadhan.md). Let
`Λ(f) ≝ log 𝔼_Q[e^f]`, convex in `f`. The principle says `Λ` is the convex
conjugate of `P ↦ 𝕂(P‖Q)` under the pairing `⟨P, f⟩ = 𝔼_P[f]`,

  ```
  Λ(f) = sup_P { ⟨P, f⟩ − 𝕂(P‖Q) }
  ```

and DV says `𝕂(·‖Q)` is in turn the conjugate of `Λ`,

  ```
  𝕂(P‖Q) = sup_f { ⟨P, f⟩ − Λ(f) }
  ```

Both proofs use the same identity `𝕂(P‖G) = 𝕂(P‖Q) − 𝔼_P[f] + log Z`:
here one optimizes it over `P` at fixed `f`, there over `f` at fixed `P`.

Regularity: `f` measurable with `0 < 𝔼_Q[e^f] < ∞`; `P` with
`𝔼_P[|f|] < ∞` (so `F(P)` is not `∞ − ∞`); `𝔼_G[|f|] < ∞` for the attaining
step, which holds e.g. for bounded `f`. KLs are in `[0, ∞]` as in
gibbs-inequality.md.
