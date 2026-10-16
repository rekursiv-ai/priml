# Moreau Decomposition

Moreau decomposition follows from three facts:

1. `p = prox_f(x)` iff `x − p ∈ ∂f(p)`,
2. `y ∈ ∂f(p)` iff `p ∈ ∂f*(y)`, by Fenchel–Young and `f** = f`, and,
3. `∂f` is monotone, which also makes `prox_f` firmly nonexpansive.

Let,

  ```
  f : ℝⁿ → ℝ ∪ {+∞}           proper, closed, convex
  prox_f(x) ≝ argmin_u ( f(u) + ½‖u − x‖² )
  P         ≝ prox_f
  ```

The objective is proper, closed, and `1`-strongly convex, so the minimizer
exists and is unique. `f*` is also proper, closed, and convex, so `prox_f*`
is well defined too.

First, characterize `prox_f`. Suppose `x − p ∈ ∂f(p)`. For any `u`,

  ```
  f(u) + ½‖u − x‖² = f(u) + ½‖u − p‖² + ⟨u − p, p − x⟩ + ½‖p − x‖²
                   ≥ f(p) + ⟨x − p, u − p⟩ + ⟨u − p, p − x⟩ + ½‖p − x‖²
                   = f(p) + ½‖p − x‖²
  ```

where the second step drops `½‖u − p‖² ≥ 0` and uses the subgradient
inequality. So `p = prox_f(x)`. Conversely, let `p` be the minimizer and
`uₜ = p + t(u − p)` with `t ∈ (0,1)`. Convexity gives
`f(uₜ) ≤ f(p) + t(f(u) − f(p))`, so

  ```
  0 ≤ f(uₜ) + ½‖uₜ − x‖² − f(p) − ½‖p − x‖²
    ≤ t (f(u) − f(p)) + t ⟨u − p, p − x⟩ + ½ t² ‖u − p‖²
  ```

Divide by `t` and let `t → 0`: `f(u) ≥ f(p) + ⟨x − p, u − p⟩`, i.e.,
`x − p ∈ ∂f(p)`.

Second, invert subgradients. By Fenchel–Young applied to `f`, and to `f*`
with `f** = f` (Fenchel–Moreau),

  ```
  y ∈ ∂f(p)  ⇔  f(p) + f*(y) = ⟨p,y⟩
             ⇔  f**(p) + f*(y) = ⟨p,y⟩
             ⇔  p ∈ ∂f*(y)
  ```

Third, decompose. Let `p ≝ prox_f(x)` and `q ≝ x − p`,

  ```
  q   = x − p ∈ ∂f(p)
  ⇔ p ∈ ∂f*(q)
  ⇔ x − q ∈ ∂f*(q)
  ⇔ q = prox_f*(x)
  ```

using the first step for `f`, the second step, `p = x − q`, and the first
step for `f*`.

Hence,

  ```
  x = prox_f(x) + prox_f*(x)
  ```

and, from the equality case of Fenchel–Young,
`⟨prox_f(x), prox_f*(x)⟩ = f(prox_f(x)) + f*(prox_f*(x))`. For a closed
convex cone `K`, `f = ι_K` has `f* = ι_K°` (polar cone), and this reads
`x = P_K x + P_K° x` with `P_K x ⊥ P_K° x`. For a subspace it is the
orthogonal decomposition.

Fourth, `∂f` is monotone. Let `a ∈ ∂f(u)` and `b ∈ ∂f(v)`. Adding
`f(v) ≥ f(u) + ⟨a, v − u⟩` and `f(u) ≥ f(v) + ⟨b, u − v⟩`,

  ```
  0 ≥ ⟨a, v − u⟩ + ⟨b, u − v⟩
    = −⟨a − b, u − v⟩
  ```

so `⟨a − b, u − v⟩ ≥ 0`.

Fifth, firm nonexpansiveness. By the first step `x − Px ∈ ∂f(Px)` and
`y − Py ∈ ∂f(Py)`. Monotonicity with `u = Px`, `v = Py` gives

  ```
  0 ≤ ⟨(x − Px) − (y − Py), Px − Py⟩
    = ⟨x − y, Px − Py⟩ − ‖Px − Py‖²
  ```

Hence,

  ```
  ‖Px − Py‖² ≤ ⟨Px − Py, x − y⟩
             ≤ ‖Px − Py‖ ‖x − y‖
  ```

by Cauchy–Schwarz, so `‖Px − Py‖ ≤ ‖x − y‖`. Equality in the first line
holds iff the monotonicity inequality is tight at `(Px, Py)`; e.g., `f = 0`
gives `P = I` and equality everywhere. Applying the same bound to
`I − P = prox_f*` shows `I − P` is also firmly nonexpansive.

Regularity: the proof needs `f` proper, closed, and convex. Closedness is
used for `f** = f` in the second step and for existence of the minimizer.
Everything holds verbatim in a Hilbert space.
