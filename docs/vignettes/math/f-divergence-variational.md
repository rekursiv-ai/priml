# f-Divergence Variational Form

The variational form of an f-divergence follows from one fact: the
Fenchel–Young inequality `f(u) ≥ u ⋅ t − f*(t)`, applied pointwise at
`u = p(x)/q(x)` and `t = g(x)`.

Let,

  ```
  f : [0, ∞) → (−∞, ∞] convex, f(1) = 0, finite on (0, ∞)
  f*(t)    ≝ sup_{u ≥ 0} ( u ⋅ t − f(u) )
  p, q ≝ densities of P, Q w.r.t. a common base measure μ, with P ≪ Q
  r(x)     ≝ p(x) / q(x)                  on { q > 0 }
  D_f(P‖Q) ≝ ∫ q(x) ⋅ f( r(x) ) dμ(x)
  ```

and let `g` range over measurable functions with `𝔼_P[g]` and
`𝔼_Q[f*(g)]` finite.

First, Fenchel–Young. By the definition of `f*` as a supremum over `u`, for
every `u ≥ 0` and every `t`,

  ```
  f*(t) ≥ u ⋅ t − f(u)
  f(u)  ≥ u ⋅ t − f*(t)
  ```

with equality iff `u` attains the supremum, i.e., iff `t ∈ ∂f(u)`. When `f`
is differentiable at `u`, that is `t = f'(u)`.

Second, rewrite the objective as a `Q`-expectation. Since `P ≪ Q`,
`p = q ⋅ r` `μ`-a.e., so `𝔼_P[g] = 𝔼_Q[r ⋅ g]` and,

  ```
  𝔼_P[g] − 𝔼_Q[f*(g)] = ∫ q(x) ⋅ ( r(x) ⋅ g(x) − f*(g(x)) ) dμ(x)
                      ≤ ∫ q(x) ⋅ f( r(x) ) dμ(x)
                      = D_f(P‖Q)
  ```

where the inequality is the first step at `u = r(x)`, `t = g(x)`, for each
`x`, integrated against `q ≥ 0`.

Third, the bound is attained. Take `g*(x) ≝ f'( r(x) )`. Then the first step
holds with equality at every `x`, so,

  ```
  𝔼_P[g*] − 𝔼_Q[f*(g*)] = ∫ q(x) ⋅ ( r(x) ⋅ f'(r(x)) − f*( f'(r(x)) ) ) dμ(x)
                        = ∫ q(x) ⋅ f( r(x) ) dμ(x)
                        = D_f(P‖Q)
  ```

Hence,

  ```
  D_f(P‖Q) = sup_g 𝔼_P[g] − 𝔼_Q[f*(g)],     g* ≝ f'(dP/dQ)
  ```

Any `g` gives a lower bound. The gap is
`∫ q ⋅ ( f(r) − r ⋅ g + f*(g) ) dμ`, which is zero iff `g(x) ∈ ∂f(r(x))` for
`Q`-almost every `x`.

For KL, `f(u) = u log u`. Then `f*(t) = sup_u ( u t − u log u )`; setting
the derivative `t − log u − 1` to zero gives `u = e^{t−1}` and
`f*(t) = e^{t−1}`. The bound becomes the NWJ form,

  ```
  𝕂(P‖Q) = sup_g 𝔼_P[g] − 𝔼_Q[ e^{g−1} ],     g* ≝ 1 + log(dP/dQ)
  ```

Regularity: `f` convex with `f(1) = 0`; `P ≪ Q`; `g` with `𝔼_P[g]` and
`𝔼_Q[f*(g)]` finite. On `{ r = 0 }` the subgradient `∂f(0)` may be empty
(e.g., `f'(0) = −∞` for KL); there the supremum is approached by `g → −∞`
rather than attained, since `inf_t f*(t) = −f(0)`. Without `P ≪ Q`, the
singular part of `P` contributes `f'(∞) ⋅ P_⊥(𝒳)` with
`f'(∞) = lim_{u→∞} f(u)/u`, and the same pointwise argument applies there.
