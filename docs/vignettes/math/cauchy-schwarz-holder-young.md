# Cauchy–Schwarz, Hölder, Young

The three inequalities form a chain:

1. Young follows from concavity of `log`,
2. Hölder follows from Young applied termwise to normalized vectors, and,
3. Cauchy–Schwarz is Hölder at `p = q = 2`, with a direct proof from a
   nonnegative quadratic.

Let,

  ```
  p, q > 1,   1/p + 1/q = 1
  a, b ≥ 0
  x, y ∈ ℝⁿ,   ‖x‖_p ≝ (Σᵢ |xᵢ|ᵖ)^{1/p}
  ```

First, Young's inequality `ab ≤ aᵖ/p + b^q/q`. If `a = 0` or `b = 0` it is
trivial. Otherwise, by concavity of `log` with weights `1/p + 1/q = 1`,

  ```
  log(ab) = log a + log b
          = (1/p) log(aᵖ) + (1/q) log(b^q)
          ≤ log( (1/p) aᵖ + (1/q) b^q )
  ```

and `exp` is increasing. Strict concavity gives equality iff `aᵖ = b^q`.

Second, Hölder's inequality. If `‖x‖_p = 0` or `‖y‖_q = 0`, one vector is zero
and it is trivial. Otherwise let `aᵢ ≝ |xᵢ| / ‖x‖_p` and `bᵢ ≝ |yᵢ| / ‖y‖_q`,
so `Σᵢ aᵢᵖ = Σᵢ bᵢ^q = 1`. Applying Young to each pair,

  ```
  Σᵢ aᵢbᵢ ≤ Σᵢ ( aᵢᵖ/p + bᵢ^q/q )
          = (1/p) Σᵢ aᵢᵖ + (1/q) Σᵢ bᵢ^q
          = 1/p + 1/q
          = 1
  ```

Multiplying by `‖x‖_p ‖y‖_q` and using the triangle inequality,

  ```
  |Σᵢ xᵢyᵢ| ≤ Σᵢ |xᵢ| |yᵢ|
            ≤ ‖x‖_p ‖y‖_q
  ```

Equality in the second line holds iff `aᵢᵖ = bᵢ^q` for every `i`, i.e.,
`|xᵢ|ᵖ / ‖x‖_pᵖ = |yᵢ|^q / ‖y‖_q^q`. Equality in the first additionally needs
all nonzero `xᵢyᵢ` to share one sign.

Third, Cauchy–Schwarz. At `p = q = 2` Hölder gives `|xᵀy| ≤ ‖x‖₂ ‖y‖₂`. A
direct proof works in any real inner product space. If `y = 0` it is trivial.
Otherwise, for every real `t`,

  ```
  0 ≤ ‖x − t y‖²
    = ‖x‖² − 2t ⟨x, y⟩ + t² ‖y‖²
  ```

A quadratic in `t` with positive leading coefficient is nonnegative for all
`t` iff its discriminant is nonpositive,

  ```
  0 ≥ 4 ⟨x, y⟩² − 4 ‖x‖² ‖y‖²
  ```

Hence,

  ```
  ab        ≤ aᵖ/p + b^q/q
  |Σᵢ xᵢyᵢ| ≤ ‖x‖_p ‖y‖_q
  |⟨x, y⟩|  ≤ ‖x‖ ‖y‖
  ```

Cauchy–Schwarz is tight iff the discriminant is zero, i.e., the quadratic has
a real root `t₀` with `‖x − t₀y‖ = 0`. So equality holds iff `x` and `y` are
linearly dependent.

Regularity: Young needs `a, b ≥ 0` and conjugate exponents `p, q > 1`; Hölder
needs finite `‖x‖_p` and `‖y‖_q`, and holds verbatim for integrals with `Σ`
replaced by `∫`; Cauchy–Schwarz needs only a real inner product. Over `ℂ`,
replace `y` by `e^{iφ}y` with `φ` chosen so `⟨x, e^{iφ}y⟩ = |⟨x, y⟩|`; the
cross term is then real and the same proof applies.
