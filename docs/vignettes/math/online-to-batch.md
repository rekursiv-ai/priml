# Online-to-Batch Conversion

Online-to-batch follows from two facts:

1. each online iterate is tested on a fresh iid point, so its online loss is
   an unbiased estimate of its risk, and,
2. Jensen moves the risk of the average below the average risk.

Let,

  ```
  W                                  convex parameter set
  ℓ(w, z)                            loss, convex in w for every z
  z₁..z_T ~ D                        iid
  w_t         ≝ 𝒜(z₁..z_{t−1}) ∈ W   online iterate, a function of the past only
  R(w)        ≝ 𝔼_{z~D}[ ℓ(w, z) ]
  w̄           ≝ (1/T) Σ_t w_t
  Regret_T(u) ≝ Σ_t ℓ(w_t, z_t) − Σ_t ℓ(u, z_t)
  Regret_T    ≝ sup_{u ∈ W} Regret_T(u)
  ```

First, the online loss is unbiased for the risk. `w_t` is a function of
`z₁..z_{t−1}`, and `z_t` is independent of them with law `D`,

  ```
  𝔼[ ℓ(w_t, z_t) | z₁..z_{t−1} ] = ∫ ℓ(w_t, z) dD(z)
                                 = R(w_t)
  ```

Taking expectations (tower property) and summing,

  ```
  𝔼[ Σ_t ℓ(w_t, z_t) ] = 𝔼[ Σ_t R(w_t) ]
  ```

Equivalently, `Σ_t (ℓ(w_t, z_t) − R(w_t))` is a martingale with zero mean.

Second, for a fixed comparator `u ∈ W`, `𝔼[ℓ(u, z_t)] = R(u)`. So,

  ```
  𝔼[ Regret_T(u) ] = 𝔼[ Σ_t ℓ(w_t, z_t) ] − 𝔼[ Σ_t ℓ(u, z_t) ]
                   = 𝔼[ Σ_t R(w_t) ] − T ⋅ R(u)
  ```

Third, `R` is convex as an expectation of functions convex in `w`, and
`w̄ ∈ W` since `W` is convex. Jensen (see jensen.md) on the uniform average
gives,

  ```
  R(w̄) = R( (1/T) Σ_t w_t )
       ≤ (1/T) Σ_t R(w_t)
  ```

Combining the second and third steps,

  ```
  𝔼[ R(w̄) ] − R(u) ≤ (1/T) ⋅ 𝔼[ Σ_t R(w_t) ] − R(u)
                   = 𝔼[ Regret_T(u) ] / T
                   ≤ 𝔼[ Regret_T ] / T
  ```

where the last step uses `Regret_T(u) ≤ Regret_T`.

Hence, for every `u ∈ W`, in particular `u = argmin_W R` when it exists,

  ```
  𝔼[ R(w̄) ] − R(u) ≤ 𝔼[ Regret_T ] / T
  ```

A regret bound `Regret_T ≤ G B √T` (e.g., online gradient descent with
`‖∇ℓ‖ ≤ G`, `‖u‖ ≤ B`) thus gives excess risk `G B / √T`. Equality requires
`R` affine along the iterates (Jensen tight) and the comparator `u` to attain
the sup in `Regret_T`.

Regularity: the proof needs `z_t` iid, `w_t` measurable with respect to
`z₁..z_{t−1}` (no peeking at `z_t`), `W` convex, `ℓ(·, z)` convex for every
`z`, and `𝔼|ℓ(w_t, z_t)| < ∞` and `R(u) < ∞` so that expectations split.
Convexity is used only in the Jensen step; without it, a uniformly random
iterate `w_τ` with `τ ~ Unif{1..T}` satisfies the same bound with equality in
place of Jensen.
