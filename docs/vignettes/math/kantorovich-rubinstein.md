# Kantorovich–Rubinstein

Kantorovich–Rubinstein duality follows from two facts:

1. any 1-Lipschitz `g` integrates against any coupling to a lower bound on
   transport cost (weak duality), and,
2. LP duality, plus the observation that the optimal dual potentials can be
   taken of the form `(φ, −φ)` with `φ` 1-Lipschitz (strong duality).

Let `(𝒳, d)` be a metric space, and,

  ```
  Π(P,Q)  ≝ { couplings π of P and Q }
  W₁(P,Q) ≝ inf_{π ∈ Π(P,Q)} 𝔼_π[ d(X,Y) ]
  Lip(g)  ≝ sup_{x ≠ y} |g(x) − g(y)| / d(x,y)
  ```

with `P, Q` of finite first moment, so every 1-Lipschitz `g` is integrable.

First, weak duality. For any `π ∈ Π(P,Q)` and any `g` with `Lip(g) ≤ 1`,

  ```
  𝔼_P[g] − 𝔼_Q[g] = 𝔼_π[ g(X) ] − 𝔼_π[ g(Y) ]
                  = 𝔼_π[ g(X) − g(Y) ]
                  ≤ 𝔼_π[ d(X,Y) ]
  ```

where the first step uses that `π` has marginals `P` and `Q`, and the last
uses `g(x) − g(y) ≤ d(x,y)`. Taking `sup_g` on the left and `inf_π` on the
right,

  ```
  sup_{Lip(g) ≤ 1} 𝔼_P[g] − 𝔼_Q[g] ≤ W₁(P,Q)
  ```

For strong duality, take finite supports: `P = Σᵢ aᵢ δ_{xᵢ}`,
`Q = Σⱼ bⱼ δ_{yⱼ}`, and `dᵢⱼ = d(xᵢ, yⱼ)`. Then `W₁` is the linear program,

  ```
  W₁(P,Q) = min_π Σᵢⱼ πᵢⱼ dᵢⱼ     s.t.  Σⱼ πᵢⱼ = aᵢ,  Σᵢ πᵢⱼ = bⱼ,  πᵢⱼ ≥ 0
  ```

Second, LP duality. The primal is feasible (`πᵢⱼ = aᵢ bⱼ`) and bounded below
by `0`, so strong LP duality (cited, not proved) gives equal optimal values
for the dual,

  ```
  W₁(P,Q) = max_{u,v} Σᵢ aᵢ uᵢ + Σⱼ bⱼ vⱼ     s.t.  uᵢ + vⱼ ≤ dᵢⱼ  for all i, j
  ```

Third, any dual-feasible `(u, v)` is dominated by a 1-Lipschitz potential.
Define on all of `𝒳`,

  ```
  φ(x) ≝ minⱼ ( d(x, yⱼ) − vⱼ )
  ```

Each `x ↦ d(x, yⱼ) − vⱼ` is 1-Lipschitz by the triangle inequality, and a
minimum of 1-Lipschitz functions is 1-Lipschitz, so `Lip(φ) ≤ 1`. Moreover,

  ```
  φ(xᵢ) = minⱼ ( dᵢⱼ − vⱼ )
        ≥ uᵢ
  φ(yⱼ) ≤ d(yⱼ, yⱼ) − vⱼ
        = −vⱼ
  ```

where the first uses dual feasibility `uᵢ ≤ dᵢⱼ − vⱼ` for every `j`.
Hence `uᵢ ≤ φ(xᵢ)` and `vⱼ ≤ −φ(yⱼ)`, and since `aᵢ, bⱼ ≥ 0`,

  ```
  Σᵢ aᵢ uᵢ + Σⱼ bⱼ vⱼ ≤ Σᵢ aᵢ φ(xᵢ) − Σⱼ bⱼ φ(yⱼ)
                      = 𝔼_P[φ] − 𝔼_Q[φ]
                      ≤ sup_{Lip(g) ≤ 1} 𝔼_P[g] − 𝔼_Q[g]
  ```

Applying this at the dual optimum and combining with the second step,
`W₁(P,Q) ≤ sup_{Lip(g) ≤ 1} 𝔼_P[g] − 𝔼_Q[g]`. The first step gives `≥`.

Hence,

  ```
  W₁(P,Q) = sup_{Lip(g) ≤ 1} 𝔼_P[g] − 𝔼_Q[g]
  ```

In the finite case the supremum is attained, by the `φ` built from an
optimal dual `v`. A coupling `π` and potential `g` are jointly optimal iff
`g(x) − g(y) = d(x,y)` for `π`-almost every `(x,y)`: the first step is
tight exactly there.

For general `P, Q` with finite first moment on a Polish space, strong
duality is cited, not proved: it is the Kantorovich duality theorem for the
lower semicontinuous cost `d`, followed by the same `c`-transform argument
as the third step (for `c = d`, a `c`-concave function is 1-Lipschitz and
its `c`-transform is its negation).

Regularity: `d` a metric (the triangle inequality makes `φ` 1-Lipschitz and
gives `d(y,y) = 0`); `P, Q` with finite first moment so `𝔼|g| < ∞` for
1-Lipschitz `g`; finite supports for the LP proof; Polish `𝒳` for the cited
general case.
