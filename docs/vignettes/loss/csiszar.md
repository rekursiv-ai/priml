# Csiszar f-Divergences

The tangent intercept of a Csiszar f-divergence follows from two facts:

1. the integrand `m f(n / m)` is homogeneous of degree 1 in `(n, m)`, and,
2. Euler's theorem splits a degree-1 function into its two marginal costs.

Let,

  ```
  p(x; θ)          ≝ model density
  p̂(x)             ≝ target density (data, or unnormalized posterior)
  u(x; θ)          ≝ p̂(x) / p(x; θ)          # written u when x, θ are clear
  f(u)             ≝ convex function on (0, ∞)
  f′(u)            ≝ derivative of f(u) in u
  T_f(u)           ≝ f(u) − u f′(u)          # the tangent intercept
  D_f[p̂, p(·; θ)]  ≝ E_{p(X; θ)}[f(u(X; θ))]
  m                ≝ model density value at x, so m = p(x; θ)
  n                ≝ target density value at x, so n = p̂(x)
  F(n, m)          ≝ m f(n / m)               # so D_f[p̂, p(·; θ)] = ∫ F(p̂(x), p(x; θ)) dx
  ```

Then, with `u = n / m`,

  ```
  ∂_m F(n, m) = f(n / m) + m f′(n / m) ⋅ (−n / m²)
              = f(u) − u f′(u)
              = T_f(u)
  ∂_n F(n, m) = m f′(n / m) ⋅ (1 / m)
              = f′(u)
  ```

where both steps are the chain rule through `u = n / m`.

So `T_f(u)` is how `D_f[p̂, p(·; θ)]` responds to more model mass at `x`; it
is the weight on the score in `∇_θ D_f[p̂, p(·; θ)]` (see
[Generalized REINFORCE](reinforce.md)).

Geometrically, the tangent to `f` at `u` splits `f(u)` into two parts,

  ```
  f(u) = (f(u) − u f′(u)) + f′(u) ⋅ u
       =    intercept     + slope ⋅ u
  ```

1. Intercept `T_f(u) = f(u) − u f′(u)`: the marginal cost of model mass at
   `x`, `∂_m F(n, m)`.
2. Slope `f′(u)`: the marginal cost of target mass at `x`, `∂_n F(n, m)`.

The split is Euler's theorem: `F(n, m)` is homogeneous of degree 1 in
`(n, m)`, so `F(n, m) = m ∂_m F(n, m) + n ∂_n F(n, m)`; dividing by `m` gives
`f(u) = T_f(u) + f′(u) ⋅ u`.

Raising `m` does two things:

1. it adds mass at `x`, contributing `+f(u)`, and,
2. it lowers the ratio `u` at `x`, contributing `−u f′(u)`.

## Choices of f

Every row below has `u = p̂(x) / p(x; θ)`, and every `f′(u)` and `T_f(u)` was
checked with sympy.

| objective            | `f(u)`                         | `f′(u)`                                   | `T_f(u)`                                |
|----------------------|--------------------------------|-------------------------------------------|-----------------------------------------|
| constant             | `1`                            | `0`                                       | `1`                                     |
| forward KL (MLE)     | `u log u`                      | `log u + 1`                               | `−u`                                    |
| reverse KL (ELBO)    | `−log u`                       | `−1 / u`                                  | `1 − log u`                             |
| Amari α, α ∉ {0, 1}  | `(u^α − 1) / (α(α − 1))`       | `u^(α − 1) / (α − 1)`                     | `−u^α / α − 1 / (α(α − 1))`             |
| Jensen–Shannon       | `u log u − (1 + u) log(1 + u)` | `log(u / (1 + u))`                        | `−log(1 + u)`                           |
| arithmetic–geometric | `(1 + u) log((1 + u) / √u)`    | `log(1 + u) − ½ log u + ½ − 1 / (2u)`     | `log(1 + u) − ½ log u − ½(u − 1)`       |
| total variation      | `½ \|u − 1\|`                  | `½ sign(u − 1)`                           | `−½ sign(u − 1)`                        |
| Pearson χ²           | `(u − 1)²`                     | `2(u − 1)`                                | `1 − u²`                                |
| Neyman-form χ²       | `u² − 1`                       | `2u`                                      | `−1 − u²`                               |
| squared Hellinger    | `(√u − 1)²`                    | `1 − 1 / √u`                              | `1 − √u`                                |
| triangular           | `(u − 1)² / (1 + u)`           | `(u − 1)(u + 3) / (1 + u)²`               | `−(u − 1)(3u + 1) / (1 + u)²`           |
| t-power              | `κ (u^t − 1)`                  | `κ t u^(t − 1)`                           | `κ ((1 − t) u^t − 1)`                   |
| exp-abs-log          | `exp(\|log u\|) − 1`           | `1` if `u > 1`, `−1 / u²` if `u < 1`      | `−1` if `u > 1`, `2 / u − 1` if `u < 1` |
| Jeffreys             | `½ (u − 1) log u`              | `½ (log u + 1 − 1 / u)`                   | `½ (1 − u − log u)`                     |
| modified GAN         | `log(1 + u) − log u`           | `1 / (1 + u) − 1 / u`                     | `log((1 + u) / u) + 1 / (1 + u)`        |
| dual of `g`          | `u g(1 / u)`                   | `g(1 / u) − g′(1 / u) / u`                | `g′(1 / u)`                             |
| symmetrized `g`      | `½ g(u) + ½ u g(1 / u)`        | `½ g′(u) + ½ g(1 / u) − g′(1 / u) / (2u)` | `½ (T_g(u) + g′(1 / u))`                |

where,

  ```
  κ       ≝ −1 if 0 < t < 1, else +1
  T_g(u)  ≝ g(u) − u g′(u)                 # the tangent intercept of g
  ```

The constant row is not a divergence (`f(1) ≠ 0`); it is listed because its
`T_f(u)` is a baseline.

Three consequences:

1. Adding `c (u − 1)` to `f(u)` adds `c (u − 1) − c u = −c` to `T_f(u)`.
   Hence the two χ² rows, which differ by `2(u − 1)`, have `T_f(u)` differing
   by a constant.
2. The dual row: `T_f(u) = g′(1 / u)` for `f(u) = u g(1 / u)`. E.g., forward KL
   `u log u` is the dual of `g(u) = −log u`, so its `T_f(u) = g′(1 / u) = −u`.
3. Forward KL: `T_f(u) = −u` and `f′(u) = log u + 1`, so adding model mass
   where `u(x; θ)` is large lowers the divergence most.

Regularity: `f(u)` differentiable at `u` (total variation and exp-abs-log fail
at `u = 1`); `p(x; θ) > 0` wherever `p̂(x) > 0`.

## References

- [f-DPG][f-DPG]: Go et al. 2023, "Aligning Language Models with Preferences
  through f-divergence Minimization", ICML. Tab. 1 lists generators of common
  f-divergences; Thm. 1 is the dual row, `T_f(u) = g′(1 / u)`.
- [k3][k3]: Schulman 2020, "Approximating KL Divergence". `f(r) − f′(1)(r − 1)`
  is the distance from `f` to its tangent at `r = 1` (a Bregman divergence).
- [f-VI][f-VI]: Wan et al. 2020, "f-Divergence Variational Inference",
  NeurIPS.

[f-DPG]: https://arxiv.org/abs/2302.08215
[k3]: http://joschu.net/blog/kl-approx.html
[f-VI]: https://arxiv.org/abs/2009.13093
