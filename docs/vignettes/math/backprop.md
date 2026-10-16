# Backpropagation

Reverse-mode differentiation computes the exact gradient of a scalar output
at a constant multiple of the forward cost because of two facts:

1. the chain rule lets one node's adjoint be pushed onto its parents, and,
2. processing nodes in reverse topological order touches each edge once.

Let,

  ```
  v₁, …, vₙ                                    inputs, x ≝ (v₁, …, vₙ)
  v_k                     ≝ φ_k( v_{pa(k)} )   k = n+1, …, N,   pa(k) ⊆ {1, …, k−1}
  f(x)                    ≝ v_N                scalar output
  G_k(v₁, …, v_k)                              output with v_{k+1}, …, v_N recomputed
  G_N(v₁, …, v_N)         ≝ v_N
  G_{k−1}(v₁, …, v_{k−1}) ≝ G_k(v₁, …, v_{k−1}, φ_k(v_{pa(k)}))
  ```

So `G_n = f`. Write `∂_i φ_k ≝ ∂φ_k/∂v_i`, with `∂_i φ_k = 0` for
`i ∉ pa(k)`, and `a⁽ᵏ⁾_i ≝ ∂G_k/∂v_i` for `i ≤ k`.

First, the chain rule removes one node. Differentiating the recursion for
`G_{k−1}` in `v_i`, `i < k`,

  ```
  a⁽ᵏ⁻¹⁾_i = ∂G_k/∂v_i + ∂G_k/∂v_k ⋅ ∂_i φ_k
           = a⁽ᵏ⁾_i + a⁽ᵏ⁾_k ⋅ ∂_i φ_k
  ```

Second, induct from `k = N` down to `k = n + 1`. The base is `a⁽ᴺ⁾ = e_N`.
Each step adds `a⁽ᵏ⁾_k ⋅ ∂_i φ_k` to the parents `i ∈ pa(k)` and changes no
other entry. At the end,

  ```
  a⁽ⁿ⁾_i = ∂G_n/∂v_i
         = ∂f/∂v_i            i = 1, …, n
  ```

This is reverse mode. Keep one array `v̄`, start at `v̄ = e_N`, and for
`k = N, …, n+1` do `v̄_i += v̄_k ⋅ ∂_i φ_k` for each `i ∈ pa(k)`. When node
`k` is processed, `v̄_k = a⁽ᵏ⁾_k` is final, because only nodes `j > k` write
to it. The result is exact: it is the chain rule, with no truncation.

Third, the cost. Let `c_k ≥ |pa(k)|` be the cost of evaluating `φ_k` (it
must at least read its inputs), and let all partials `∂_i φ_k` cost at most
`ω ⋅ c_k` given the stored forward values. Node `k` costs `ω c_k` for the
partials plus `2 |pa(k)|` for the multiply-adds,

  ```
  cost(reverse) ≤ Σ_k ( ω ⋅ c_k + 2 ⋅ |pa(k)| )
                ≤ Σ_k (ω + 2) ⋅ c_k
                = (ω + 2) ⋅ cost(forward)
  ```

Fourth, the contrast with forward mode. Propagating a tangent
`v̇_k = Σ_i ∂_i φ_k ⋅ v̇_i` from `v̇_{1:n} = u` gives, by the chain rule in
topological order,

  ```
  v̇_N = d/dε f(x + ε u) |_{ε=0}
      = ⟨∇f(x), u⟩
  ```

One forward pass yields one directional derivative. Recovering all `n`
partials takes `u = e₁, …, eₙ`, i.e., `n` passes, each costing
`Θ(cost(forward))`.

Hence,

  ```
  ∇f(x)    = v̄_{1:n}                           (exact)
  cost(∇f) ≤ (ω + 3) ⋅ cost(f)                 reverse mode, one pass
  cost(∇f) = Θ(n ⋅ cost(f))                    forward mode, n passes
  ```

with the `+3` counting the forward pass that stores `v_{1:N}`. For
elementary `φ_k` (`+`, `×`, `exp`, `sin`, …) `ω` is a small constant, so the
gradient is never more than a fixed factor dearer than `f`, independent of
`n`. This is the cheap gradient principle: Baur and Strassen (1983) prove
`L(f, ∇f) ≤ 3 ⋅ L(f)` in multiplicative complexity for rational `f`.

Fine print: reverse mode stores every `v_k`, so memory is `O(N)` against
`O(1)` extra for forward mode; checkpointing trades recomputation for
memory. For `f : ℝⁿ → ℝᵐ`, reverse mode needs `m` passes and forward mode
`n`, so forward wins when `n < m`. Vector-valued nodes work the same way,
with `∂_i φ_k` a Jacobian and `v̄_k` a vector-Jacobian product; then `ω`
bounds the cost of that product, not of the full Jacobian. At points where
some `φ_k` is not differentiable (e.g., ReLU at 0), the recursion returns a
chosen element, which need not be a subgradient of `f`.

Regularity: the proof needs a finite DAG with nodes in topological order,
each `φ_k` differentiable at the evaluated point (so the chain rule holds),
the forward values `v_{1:N}` stored for the reverse pass, and the cost model
`c_k ≥ |pa(k)|` with partials costing at most `ω ⋅ c_k`.
