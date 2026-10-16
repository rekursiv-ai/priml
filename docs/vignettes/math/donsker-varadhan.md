# Donsker–Varadhan

The Donsker–Varadhan form of KL follows from one fact: for any `g`, the gap
between `𝕂(P‖Q)` and the bound is `𝕂(P‖G)` for the Gibbs measure
`dG ∝ e^g dQ`, which is nonnegative (gibbs-inequality.md).

Let,

  ```
  Z(g)  ≝ 𝔼_Q[ e^g ]
  dG    ≝ ( e^g / Z(g) ) dQ                      Gibbs (tilted) measure
  DV(g) ≝ 𝔼_P[g] − log 𝔼_Q[ e^g ]
  ```

where `g` is measurable with `Z(g) < ∞` and `𝔼_P[|g|] < ∞`.

First, if `P` is not absolutely continuous w.r.t. `Q`, then
`𝕂(P‖Q) = ∞ ≥ DV(g)` and the inequality holds trivially. So assume
`P ≪ Q`. Since `G` and `Q` are mutually absolutely continuous, `P ≪ G` and
the density ratios chain,

  ```
  log( dP/dG ) = log( dP/dQ ) + log( dQ/dG )
               = log( dP/dQ ) − g + log Z(g)
  ```

Second, take `𝔼_P` of both sides,

  ```
  𝕂(P‖G) = 𝔼_P[ log( dP/dQ ) ] − 𝔼_P[g] + log Z(g)
         = 𝕂(P‖Q) − DV(g)
  ```

and since `𝕂(P‖G) ≥ 0`,

  ```
  DV(g) = 𝕂(P‖Q) − 𝕂(P‖G)
        ≤ 𝕂(P‖Q)
  ```

Third, the bound is attained at `g* ≝ log(dP/dQ) + c` for any constant `c`,

  ```
  DV(g*) = 𝔼_P[ log(dP/dQ) ] + c − log 𝔼_Q[ (dP/dQ) ⋅ e^c ]
         = 𝕂(P‖Q) + c − log( e^c ⋅ 1 )
         = 𝕂(P‖Q)
  ```

where the second step uses `𝔼_Q[dP/dQ] = 1`.

Hence,

  ```
  𝕂(P‖Q) = sup_g 𝔼_P[g] − log 𝔼_Q[ e^g ],     g* = log(dP/dQ) + c
  ```

The gap at any `g` is exactly `𝕂(P‖G)`. It vanishes iff `G = P`, i.e., iff
`g = log(dP/dQ) + c` `P`-almost surely. DV is invariant to shifts of `g`.

DV is tighter than the f-divergence (NWJ) form,
`𝕂(P‖Q) ≥ 𝔼_P[g] − 𝔼_Q[ e^{g−1} ]` (f-divergence-variational.md), for
every `g`. Fourth, compare the two penalties using `log x ≤ x − 1` at
`x = Z/e`,

  ```
  log Z(g) = log( Z(g)/e ) + 1
           ≤ Z(g)/e − 1 + 1
           = 𝔼_Q[ e^{g−1} ]
  ```

so `DV(g) ≥ 𝔼_P[g] − 𝔼_Q[ e^{g−1} ]`, with equality iff `Z(g) = e`.
Equivalently, DV is NWJ optimized over a constant shift: maximizing
`𝔼_P[g] + c − e^{c−1} Z(g)` over `c` sets `e^{c−1} Z(g) = 1`, so
`c = 1 − log Z(g)` and the value is `𝔼_P[g] − log Z(g) = DV(g)`.

Regularity: `g` measurable with `𝔼_Q[e^g] < ∞` and `𝔼_P[|g|] < ∞`; the
attaining `g*` needs `𝕂(P‖Q) < ∞` so that `𝔼_P[|g*|] < ∞` (if
`𝕂(P‖Q) = ∞`, bounded `g` already drive `DV(g)` to `∞`, a standard fact
not proved here). Expectations of `log(dP/dQ)` under `P` are well defined in
`[0, ∞]` as in gibbs-inequality.md.
