# ADMM

ADMM converges because the potential
`V = (1/ρ) ⋅ ‖y − y*‖² + ρ ⋅ ‖B(z − z*)‖²` drops by the squared residuals
each iteration. This rests on three facts:

1. each primal update is optimal for a linear tilt of its objective,
2. a saddle point of the Lagrangian bounds the objective from both sides,
3. the z-update is monotone in the dual variable.

Let,

  ```
  minimize f(x) + g(z)  subject to  Ax + Bz = c
  L_ρ(x, z, y) ≝ f(x) + g(z) + yᵀ(Ax + Bz − c) + (ρ/2) ⋅ ‖Ax + Bz − c‖²
  x^{k+1}      ≝ argmin_x L_ρ(x, z^k, y^k)
  z^{k+1}      ≝ argmin_z L_ρ(x^{k+1}, z, y^k)
  y^{k+1}      ≝ y^k + ρ ⋅ r^{k+1}
  r^{k+1}      ≝ Ax^{k+1} + Bz^{k+1} − c          (primal residual)
  Δz^{k+1}     ≝ z^{k+1} − z^k
  p^k          ≝ f(x^k) + g(z^k),   p* ≝ f(x*) + g(z*)
  V^k          ≝ (1/ρ) ⋅ ‖y^k − y*‖² + ρ ⋅ ‖B(z^k − z*)‖²
  ```

where `(x*, z*, y*)` is a saddle point of the unaugmented Lagrangian
`L₀`: `Ax* + Bz* = c`, `−Aᵀy* ∈ ∂f(x*)`, and `−Bᵀy* ∈ ∂g(z*)`.

First, rewrite each update's optimality condition with `y^{k+1}`. The
x-update gives `0 ∈ ∂f(x^{k+1}) + Aᵀ(y^k + ρ(Ax^{k+1} + Bz^k − c))`, and the
z-update gives `0 ∈ ∂g(z^{k+1}) + Bᵀ(y^k + ρ r^{k+1})`. Since
`y^k + ρ(Ax^{k+1} + Bz^k − c) = y^{k+1} − ρBΔz^{k+1}`,

  ```
  x^{k+1} minimizes  f(x) + (y^{k+1} − ρBΔz^{k+1})ᵀ Ax
  z^{k+1} minimizes  g(z) + (y^{k+1})ᵀ Bz
  ```

Second, an upper bound on the objective. Compare each minimizer against
`x*` and `z*`, add, and use `Ax* + Bz* = c`,

  ```
  p^{k+1} − p* ≤ (y^{k+1})ᵀ(Ax* + Bz* − Ax^{k+1} − Bz^{k+1}) + ρ(BΔz^{k+1})ᵀA(x^{k+1} − x*)
               = −(y^{k+1})ᵀ r^{k+1} + ρ(BΔz^{k+1})ᵀA(x^{k+1} − x*)
               = −(y^{k+1})ᵀ r^{k+1} + ρ(BΔz^{k+1})ᵀ(r^{k+1} − B(z^{k+1} − z*))
  ```

where the last step uses `A(x^{k+1} − x*) = r^{k+1} − B(z^{k+1} − z*)`,
which follows from `Ax* = c − Bz*`.

Third, a lower bound. The saddle point gives
`L₀(x*, z*, y*) ≤ L₀(x^{k+1}, z^{k+1}, y*)`, i.e.,

  ```
  p* − p^{k+1} ≤ (y*)ᵀ r^{k+1}
  ```

Fourth, add the second and third steps. The objectives cancel; multiply
by 2,

  ```
  0 ≥ 2(y^{k+1} − y*)ᵀ r^{k+1} − 2ρ(BΔz^{k+1})ᵀ r^{k+1} + 2ρ(BΔz^{k+1})ᵀB(z^{k+1} − z*)
  ```

Fifth, recognize the potential. Write `a ≝ y^k − y*` and
`ρ r^{k+1} = y^{k+1} − y^k`. Expanding `‖a + ρr‖² − ‖a‖²`,

  ```
  2(y^{k+1} − y*)ᵀ r^{k+1} = 2(y^k − y*)ᵀ r^{k+1} + 2ρ ⋅ ‖r^{k+1}‖²
                           = (1/ρ)(‖y^{k+1} − y*‖² − ‖y^k − y*‖²) + ρ ⋅ ‖r^{k+1}‖²
  ```

Write `z^{k+1} − z* = Δz^{k+1} + (z^k − z*)` and expand the same way,

  ```
  2ρ(BΔz^{k+1})ᵀB(z^{k+1} − z*) = 2ρ‖BΔz^{k+1}‖² + 2ρ(BΔz^{k+1})ᵀB(z^k − z*)
                                = ρ‖BΔz^{k+1}‖² + ρ(‖B(z^{k+1} − z*)‖² − ‖B(z^k − z*)‖²)
  ```

Substituting both into the fourth step, the leftover terms form a square,

  ```
  0 ≥ V^{k+1} − V^k + ρ‖r^{k+1}‖² − 2ρ(BΔz^{k+1})ᵀ r^{k+1} + ρ‖BΔz^{k+1}‖²
    = V^{k+1} − V^k + ρ ⋅ ‖r^{k+1} − BΔz^{k+1}‖²
  ```

Sixth, the cross term has a sign. For `k ≥ 1`, `z^k` minimizes
`g(z) + (y^k)ᵀBz` by the first step. Compare `z^{k+1}` and `z^k` in both
tilted problems and add; the `g` terms cancel,

  ```
  (y^{k+1} − y^k)ᵀ(Bz^{k+1} − Bz^k) ≤ 0
  ρ ⋅ (r^{k+1})ᵀ BΔz^{k+1}          ≤ 0
  ```

so `−2ρ(BΔz^{k+1})ᵀ r^{k+1} ≥ 0`, and expanding the square in the fifth step,

  ```
  V^{k+1} ≤ V^k − ρ ⋅ ‖r^{k+1}‖² − ρ ⋅ ‖BΔz^{k+1}‖² + 2ρ(BΔz^{k+1})ᵀ r^{k+1}
          ≤ V^k − ρ ⋅ ( ‖r^{k+1}‖² + ‖BΔz^{k+1}‖² )
  ```

Seventh, sum over `k = 1, …, K` and drop `V^{K+1} ≥ 0`,

  ```
  ρ ⋅ Σ_{k=1}^K ( ‖r^{k+1}‖² + ‖BΔz^{k+1}‖² ) ≤ V¹ − V^{K+1}
                                              ≤ V¹
  ```

Hence,

  ```
  V^{k+1} ≤ V^k − ρ ⋅ ( ‖r^{k+1}‖² + ‖B(z^{k+1} − z^k)‖² )      k ≥ 1
  r^k → 0,   B(z^{k+1} − z^k) → 0,   p^k → p*
  min_{1≤k≤K} ( ‖r^{k+1}‖² + ‖BΔz^{k+1}‖² ) ≤ V¹ / (ρK)
  ```

The objective converges because `V^k ≤ V¹` keeps `y^k` and `B(z^k − z*)`
bounded. So the right sides of the second and third steps, which are
products of a bounded vector with `r^{k+1}` or `BΔz^{k+1}`, both tend to 0.
The decrease is tight when `r^{k+1} ⟂ BΔz^{k+1}` and the second and third
steps hold with equality.

Fine print: the dual residual `s^{k+1} ≝ ρAᵀBΔz^{k+1}` also tends to 0, and
`(r, s)` small is the standard stopping rule (Boyd et al., 2011, §3.3). The
sixth step needs `k ≥ 1` because `z⁰` is an initialization, not a z-update
output; the fifth step alone still makes `V` nonincreasing from `k = 0`.
The bound is on the running minimum; an ergodic `O(1/k)` rate for the
averaged iterates is He and Yuan (2012). Iterate convergence of `x^k`
needs more, e.g., `A` with full column rank.

Regularity: the proof needs `f` and `g` closed, proper, and convex, a saddle
point `(x*, z*, y*)` of `L₀` (e.g., strong duality with attained optima),
`ρ > 0`, and both argmins attained so the subgradient optimality conditions
hold. No strong convexity or rank condition is used.
