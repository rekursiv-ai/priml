# Robbins–Monro

Robbins–Monro converges, from two facts:

1. strong monotonicity shrinks the squared error by a factor `1 − Θ(α_n)` per
   step, and,
2. zero-mean noise adds only `O(α_n²)`, which is summable.

Let,

  ```
  h : ℝ^d → ℝ^d,   h(x*) = 0
  x_{n+1} ≝ x_n − α_n ⋅ ( h(x_n) + ξ_{n+1} )
  𝓕_n     ≝ σ(x₀, ξ₁, …, ξ_n)
  ```

and assume, for constants `μ > 0` and `L, σ², c ≥ 0`,

  ```
  (x − x*)ᵀ h(x)        ≥ μ ‖x − x*‖²               (strong monotonicity)
  ‖h(x)‖²               ≤ L² ‖x − x*‖²              (growth)
  𝔼[ ξ_{n+1} | 𝓕_n ]      = 0
  𝔼[ ‖ξ_{n+1}‖² | 𝓕_n ]   ≤ σ² + c ‖x_n − x*‖²
  α_n                   ≥ 0,   Σ_n α_n = ∞,   Σ_n α_n² < ∞,   𝔼‖x₀ − x*‖² < ∞
  ```

with `α_n` deterministic for the mean-square part. Write `K ≝ L² + c`.

First, a one-step conditional recursion. Write `e ≝ x_n − x*`, `h ≝ h(x_n)`,
`ξ ≝ ξ_{n+1}`,

  ```
  ‖x_{n+1} − x*‖² = ‖e − α_n (h + ξ)‖²
                  = ‖e‖² − 2 α_n eᵀ(h + ξ) + α_n² ‖h + ξ‖²
  ```

and condition on `𝓕_n`,

  ```
  𝔼[ ‖x_{n+1} − x*‖² | 𝓕_n ] = ‖e‖² − 2 α_n eᵀh + α_n² ( ‖h‖² + 𝔼[ ‖ξ‖² | 𝓕_n ] )
                             ≤ ‖e‖² − 2 μ α_n ‖e‖² + α_n² ( L² ‖e‖² + σ² + c ‖e‖² )
                             = (1 − 2 μ α_n + K α_n²) ‖e‖² + σ² α_n²
  ```

where the first step uses that `e`, `h` are `𝓕_n`-measurable and
`𝔼[ξ | 𝓕_n] = 0`, which kills `eᵀξ` and `hᵀξ`, and the second uses the
assumptions.

Second, take expectations. Let `u_n ≝ 𝔼‖x_n − x*‖²`; each `u_n` is finite by
induction from `u_0`. Since `Σ α_n² < ∞`, `α_n → 0`, so there is `N` with
`K α_n ≤ μ` and `μ α_n ≤ 1` for `n ≥ N`. Then for `n ≥ N`,

  ```
  u_{n+1} ≤ (1 − 2 μ α_n + K α_n²) u_n + σ² α_n²
          ≤ (1 − 2 μ α_n + μ α_n) u_n + σ² α_n²
          = (1 − μ α_n) u_n + σ² α_n²
  ```

with `0 ≤ 1 − μ α_n ≤ 1`.

Third, a Chung-type lemma. If `u_n ≥ 0`, `u_{n+1} ≤ (1 − a_n) u_n + b_n` for
`n ≥ N`, `a_n ∈ [0,1]`, `Σ a_n = ∞`, `b_n ≥ 0`, and `Σ b_n < ∞`, then
`u_n → 0`. Proof: for `N ≤ m ≤ n`, unroll from `m`,

  ```
  u_{n+1} ≤ Π_{k=m}^{n} (1 − a_k) ⋅ u_m + Σ_{k=m}^{n} Π_{j=k+1}^{n} (1 − a_j) ⋅ b_k
          ≤ exp(−Σ_{k=m}^{n} a_k) ⋅ u_m + Σ_{k=m}^{∞} b_k
  ```

using `1 − a ≤ e^{−a}` and that each product lies in `[0,1]`. As `n → ∞`, the
exponential goes to zero because `Σ a_k = ∞`. So
`limsup_n u_n ≤ Σ_{k≥m} b_k` for every `m`, and the tail of a convergent
series goes to zero as `m → ∞`. Apply it with `a_n ≝ μ α_n`,
`b_n ≝ σ² α_n²`.

Fourth, almost-sure convergence, citing Robbins–Siegmund (1971): if
`Y_n, β_n, ζ_n, θ_n ≥ 0` are `𝓕_n`-adapted,
`𝔼[Y_{n+1} | 𝓕_n] ≤ (1 + β_n) Y_n + ζ_n − θ_n`, and `Σ β_n`, `Σ ζ_n` are
finite a.s., then `Y_n` converges a.s. to a finite limit and `Σ θ_n < ∞` a.s.
The first step is exactly this form with `Y_n ≝ ‖x_n − x*‖²`,

  ```
  𝔼[ Y_{n+1} | 𝓕_n ] ≤ (1 + K α_n²) Y_n + σ² α_n² − 2 μ α_n Y_n
  ```

so `β_n ≝ K α_n²`, `ζ_n ≝ σ² α_n²`, `θ_n ≝ 2 μ α_n Y_n`. Hence `Y_n → Y_∞`
a.s. and `Σ α_n Y_n < ∞` a.s. On the event `Y_∞ > 0`, eventually
`α_n Y_n ≥ α_n Y_∞ / 2`, whose sum diverges; so that event has probability
zero. This part allows `α_n` to be `𝓕_n`-measurable with the step conditions
holding a.s.

Hence,

  ```
  𝔼‖x_n − x*‖² → 0,      x_n → x*   a.s.
  ```

Both step conditions are needed. If `Σ α_n < ∞` (with `α_n L < 1`) and no
noise, `‖x_{n+1} − x*‖ ≥ (1 − α_n L) ‖x_n − x*‖`, and `Π (1 − α_n L) > 0`, so
the iterate stops short. With constant `α`, `h(x) = μ (x − x*)` scalar, and
noise variance `σ²`, `u_{n+1} = (1 − αμ)² u_n + α² σ²` converges to
`α σ² / (μ (2 − αμ)) > 0`. With `α_n = a/n` and `2 μ a > 1`, `u_n = O(1/n)`
(Chung 1954), cited, not proved here. For `h` merely monotone
(`(x − x*)ᵀ h(x) > 0` for `x ≠ x*`, `h` continuous), a.s. convergence still
holds (Robbins & Monro 1951; Blum 1954), cited, not proved here.

Regularity: the proof used strong monotonicity at `x*`, the linear growth bound
on `h`, martingale-difference noise with conditional second moment at most
`σ² + c ‖x_n − x*‖²`, `Σ α_n = ∞`, `Σ α_n² < ∞`, and `𝔼‖x₀ − x*‖² < ∞`.
