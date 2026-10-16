# Pinsker's Inequality

Pinsker's inequality follows from two facts:

1. it holds for two Bernoulli distributions, by one-variable calculus, and,
2. the data processing inequality (data-processing.md) reduces any pair
   `P, Q` to a Bernoulli pair without increasing KL.

Let,

  ```
  p, q ≝ densities of P, Q w.r.t. a common base measure μ
  TV(P,Q) ≝ ½ ∫ |p(x) − q(x)| dμ(x)
  A       ≝ { x : p(x) > q(x) }
  kl(a‖b) ≝ a ⋅ log(a/b) + (1 − a) ⋅ log( (1 − a) / (1 − b) ),   a, b ∈ [0,1]
  ```

with natural `log` and `0 ⋅ log 0 = 0`. `kl(a‖b)` is the KL between
`Bernoulli(a)` and `Bernoulli(b)`.

First, the Bernoulli case `kl(a‖b) ≥ 2 (a − b)²`. If `b ∈ {0,1}` and
`b ≠ a` then `kl = ∞`; if `b = a` both sides are `0`. Otherwise fix `a` and
let `g(b) ≝ kl(a‖b) − 2 (a − b)²` on `b ∈ (0,1)`. Differentiate,

  ```
  g'(b) = −a/b + (1 − a)/(1 − b) + 4 (a − b)
        = ( −a (1 − b) + (1 − a) b ) / ( b (1 − b) ) − 4 (b − a)
        = (b − a) / ( b (1 − b) ) − 4 (b − a)
        = (b − a) ⋅ ( 1/( b (1 − b) ) − 4 )
  ```

Since `b (1 − b) ≤ ¼`, the second factor is `≥ 0`. So `g' ≤ 0` for `b < a`
and `g' ≥ 0` for `b > a`, and `g` is minimized at `b = a`,

  ```
  g(b) ≥ g(a)
       = kl(a‖a) − 0
       = 0
  ```

Second, `TV` is attained on `A`. Since `∫ (p − q) dμ = 0`, the positive and
negative parts of `p − q` have equal mass,

  ```
  TV(P,Q) = ½ ( ∫_A (p − q) dμ + ∫_{Aᶜ} (q − p) dμ )
          = ½ ( ∫_A (p − q) dμ + ∫_A (p − q) dμ )
          = ∫_A (p − q) dμ
          = P(A) − Q(A)
  ```

where the second step uses `∫_{Aᶜ} (q − p) dμ = ∫_A (p − q) dμ`, which is
`∫ (p − q) dμ = 0` rearranged.

Third, push `P` and `Q` through the deterministic map `x ↦ 1[x ∈ A]`. The
images are `Bernoulli(P(A))` and `Bernoulli(Q(A))`, so data processing gives,

  ```
  𝕂(P‖Q) ≥ kl( P(A) ‖ Q(A) )
         ≥ 2 ( P(A) − Q(A) )²
         = 2 TV(P,Q)²
  ```

where the second step is the first step with `a = P(A)`, `b = Q(A)`, and the
last is the second step.

Hence,

  ```
  TV(P,Q) ≤ √( 𝕂(P‖Q) / 2 )
  ```

The constant is sharp: for `a = ½ + ε`, `b = ½`, `kl(a‖b) = 2ε² + O(ε⁴)` and
`TV = ε`, so no constant larger than `2` works in `𝕂 ≥ c ⋅ TV²`. Equality
holds only at `P = Q`, since `g(b) > 0` for `b ≠ a`.

Regularity: natural logarithm (with `log₂`, the bound reads
`TV ≤ √( 𝕂 ⋅ ln 2 / 2 )`); `A` measurable, which holds since `p, q` are
measurable; data processing for a deterministic kernel. The bound is
vacuous, not false, when `𝕂(P‖Q) = ∞`.
