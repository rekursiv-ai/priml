# Perceptron Mistake Bound

Novikoff's bound follows from two facts:

1. each mistake grows the alignment `⟨u, w⟩` with the separator by at least
   `γ`, and,
2. each mistake grows `‖w‖²` by at most `R²`.

Linear growth in one and square-root growth in the other cannot coexist for
long.

Let,

  ```
  (x_t, y_t),  y_t ∈ {−1, +1}        any sequence of examples
  ‖x_t‖           ≤ R
  ‖u‖             = 1,  y_t ⟨u, x_t⟩ ≥ γ > 0      separator with margin γ
  w₁              ≝ 0
  if y_t ⟨w, x_t⟩ ≤ 0:  w ← w + y_t x_t     mistake update
  ```

Let `w_{k+1}` be the weight after the `k`-th mistake, made on `(x, y)`.

First, alignment grows linearly. By the margin of `u`,

  ```
  ⟨u, w_{k+1}⟩ = ⟨u, w_k⟩ + y ⟨u, x⟩
               ≥ ⟨u, w_k⟩ + γ
  ```

so by induction from `⟨u, w₁⟩ = 0`, `⟨u, w_{k+1}⟩ ≥ kγ`.

Second, the norm grows at most like a square root,

  ```
  ‖w_{k+1}‖² = ‖w_k‖² + 2 y ⟨w_k, x⟩ + y² ‖x‖²
             ≤ ‖w_k‖² + 0 + R²
  ```

where the second step uses `y ⟨w_k, x⟩ ≤ 0` (it was a mistake) and
`y² = 1`. By induction from `‖w₁‖ = 0`, `‖w_{k+1}‖² ≤ kR²`.

Third, combine with Cauchy–Schwarz and `‖u‖ = 1`,

  ```
  kγ ≤ ⟨u, w_{k+1}⟩
     ≤ ‖u‖ ⋅ ‖w_{k+1}‖
     ≤ √k ⋅ R
  ```

Hence, squaring `√k ≤ R/γ`, the number of mistakes `k` satisfies

  ```
  k ≤ (R / γ)²
  ```

The bound is tight. Take `x_t = e_t`, `y_t = +1` for `t = 1..m`, and
`u = (1..1)/√m`, so `R = 1` and `γ = 1/√m`. Each `⟨w, e_t⟩ = 0`, since `w`
only has mass on earlier coordinates. So every round is a mistake: `m`
mistakes, `= (R/γ)²`.

Fine print: the bound holds for any order and any number of passes, so it
also bounds online mistakes against an adversary; cycling until no mistakes
yields a separator after at most `(R/γ)²` updates. Only rounds with a mistake
enter the proof. A bias term is handled by appending a constant coordinate
to `x`, which changes `R` and `γ`. Ties `⟨w, x⟩ = 0` count as mistakes;
that is what makes `w₁ = 0` move.

Regularity: the proof needs a unit `u` with `y_t ⟨u, x_t⟩ ≥ γ > 0` for every
example, `‖x_t‖ ≤ R`, `w₁ = 0`, and an inner-product space (the kernel
perceptron replaces `x` with a feature map `Φ(x)` and `R²` with
`sup k(x, x)`).
