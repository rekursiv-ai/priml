# Generalized REINFORCE: One Gradient for Likelihood, RL, and Divergence Learning

**Abstract.** Most methods that learn a density, from Williams' REINFORCE
(1992) to diffusion distillation (2025), share one gradient. Any loss
`L(θ) = E_{p(X; θ)}[f(X, u)]` with `u = p̂(x) / p(x; θ)` has
`∇_θ L(θ) = E_{p(X; θ)}[T_f(X, u) s(X; θ)]`: the score `s(x; θ)` says how `θ`
moves model mass, and one scalar weight, the tangent intercept
`T_f(x, u) = f(x, u) − u ∂_u f(x, u)`, says what the loss wants. The identity
is known for divergences; new here is reading every method through that one
weight. `T_f` is the functional derivative of the loss in `p`; Euler's
theorem splits `f` into this intercept and a slope on target mass, which `θ`
cannot move. A
method is three choices: a loss shape `f`, a reward `A`, and a target `p̂`.
Maximum likelihood is the `A ≡ 0` slice, exact with no limit; REINFORCE is the
`p̂ ≡ 0` slice. Between them sit variational inference, f-GANs, RLHF and RLVR,
PPO, GRPO and DAPO, advantage-weighted and filtered regression, on-policy
distillation, and adversarial imitation, each a row of one table. The
pathwise form carries the same weight to models known only through their
scores, `h_f(u) = −u ∂_u T_f(u)`, covering diffusion distillation (VSD, DMD,
f-distill); diffusion training reaches the maximum-likelihood row through
its likelihood bound, and RL fine-tuning of diffusion is the existing table.
Unifying gradients rather than objectives is the move from M- to
Z-estimators: weights that read other samples, such as GRPO's group
normalization, still solve an estimating equation without any objective.
Outside: pairwise losses (DPO), integral probability metrics (Wasserstein,
MMD), and hinge loss, which estimates a decision, not a density.

This doc unifies methods by their gradients, not their objectives. Earlier
unifications of MLE and RL work with objectives: [RAML][RAML] writes them as
KL to an exponentiated reward in opposite directions, and
[Tan et al. 2018][Tan] put both under one variational objective, recovering
MLE as a δ-reward in the limit `α → 0`. Here every method is one weight,
the tangent intercept `T_f(x, u)` defined below, on the score in one identity,
`∇_θ L(θ) = E_{p(X; θ)}[T_f(X, u) s(X; θ)]`. For ratio-only `f(x, u)` the
identity is known: it is [tail-adaptive f-VI][tail-f-VI] Prop. 4.2, whose
`ρ_f` is `−T_f`, and [f-DPG][f-DPG] Thm. 1. LLM post-training has reached
the gradient view from the reward side: [Lv et al. 2025][UPGE] write SFT and
RL as one policy-gradient estimator, and [DFT][DFT] reads SFT as policy
gradient with reward `1 / π`, the weight `−u` of consequence 3. Neither has
`T_f`; new here is one weight covering both lines, divergences and rewards.

This is the statistician's move from M-estimators (maximize a criterion) to
Z-estimators (solve an estimating equation; [van der Vaart 1998][vdV], ch. 5).
It buys four things:

1. MLE is exact, with no limit: the `A ≡ 0` slice with `f(x, u) = u log u`.
2. Distinct objectives can share one gradient, e.g. the two χ² rows
   (consequence 2).
3. Hybrids follow mechanically, since `f ↦ T_f` is linear.
4. A weight needs no objective. GRPO's `std_G r`, DAPO's token count, and
   PPO's per-token critic have no `f(x, u)`, but each still weights the
   score (see Regularity).

The objective view keeps what the gradient view hides: which distribution
supplies the samples. MLE's identity holds only after `u` cancels the
sampling from `p(·; θ)` (consequence 3); Tan et al.'s exploration picture
shows this directly. Diffusion enters through the
[pathwise form](#pathwise-form), which carries the same weight to models
known only through their scores.

REINFORCE for an objective that reads the model's own density follows from
three facts:

1. the product rule, since `f(x, u)` depends on `θ` through `u(x; θ)`,
2. the ratio's derivative, `∇_θ u(x; θ) = −u(x; θ) s(x; θ)`, and,
3. the log-derivative, `∇_θ p(x; θ) = p(x; θ) s(x; θ)`.

Let,

  ```
  s(x; θ)      ≝ ∇_θ log p(x; θ)          # the score
  p̂(x)         ≝ target density (data, or unnormalized posterior)
  u(x; θ)      ≝ p̂(x) / p(x; θ)           # written u when x, θ are clear
  f(x, u)      ≝ any function, differentiable in u
  ∂_u f(x, u)  ≝ derivative of f(x, u) in u
  T_f(x, u)    ≝ f(x, u) − u ∂_u f(x, u)    # the tangent intercept
  ```

and assume `f(x, u)` depends on `θ` only through `u(x; θ)`. Then,

  ```
  ∇_θ E_{p(X; θ)}[f(X, u)]
    = ∫ ∇_θ [p(x; θ) f(x, u)] dx
    = ∫ [f(x, u) ∇_θ p(x; θ) + p(x; θ) ∂_u f(x, u) ∇_θ u] dx
    = ∫ [f(x, u) − u ∂_u f(x, u)] ∇_θ p(x; θ) dx
    = E_{p(X; θ)}[(f(X, u) − u ∂_u f(X, u)) s(X; θ)]
    = E_{p(X; θ)}[T_f(X, u) s(X; θ)]
  ```

where the second step is the product rule plus the chain rule, the third uses
`∇_θ u(x; θ) = −u(x; θ) ∇_θ p(x; θ) / p(x; θ)`, and the fourth uses
`∇_θ p(x; θ) = p(x; θ) s(x; θ)`.

So weighting the score by `f(x, u)` is biased unless `∂_u f(x, u) = 0`; the
unbiased weight is the tangent intercept `T_f(x, u)`: where the tangent to
`u ↦ f(x, u)` crosses `u = 0`.

The tangent intercept is the only such weight, up to a baseline, for a
weight that must work for every model:

**Claim.** Fix `θ`, write `p = p(·; θ)`, and let `w(x)` be a weight with
`E_p[|w(X)|] < ∞` that may depend on `x` and on `p`, but not on which model
passes through `p`. Then `E_p[w(X) s(X)] = ∇ L` for every smooth model
through `p` if and only if `w(x) = T_f(x, u) + c` for `p`-almost every `x`,
for some constant `c`. Off the support of `p`, `w` is unconstrained.

**Proof.** Subtracting the identity above, `E_p[(w(X) − T_f(X, u)) s(X)] = 0`
for every model through `p`. Take the model
`p_ε(x) = p(x) (1 + ε φ(x))` with `φ` bounded and `E_p[φ(X)] = 0`; it is a
density for small `ε`, and its score at `ε = 0` is `φ(x)`. So
`E_p[g(X) φ(X)] = 0` for every bounded mean-zero `φ`, where
`g(x) ≝ w(x) − T_f(x, u)`. Take `φ = g_M − E_p[g_M(X)]` with
`g_M ≝ max(−M, min(g, M))`; then `E_p[g(X) φ(X)] = 0` for every `M`, and
letting `M → ∞` (monotone convergence on `g g_M ≥ 0`, dominated convergence
on `E_p[g_M(X)]`) gives `E_p[g(X)²] = E_p[g(X)]² < ∞`, i.e.
`Var_p(g(X)) = 0`, so `g(x) = c` almost everywhere. Conversely,
`T_f(x, u) + c` is unbiased for every model, by the identity above and
`E_p[s(X)] = 0` (consequence 1). ∎

Both directions need the regularity of the identity above (see Regularity
at the end of [Methods](#methods)). Necessity needs `w` fixed across models: a
weight allowed to change with the model is the fixed-model case below.

The quantifier matters. For one fixed `k`-parameter model, unbiasedness
imposes only `k` conditions: any `w(x)` with `w − T_f` orthogonal to the `k`
components of `s(x; θ)` also works. The intercept is the one weight that
works for every model.

This is the semiparametric canonical gradient
([van der Vaart 1998][vdV], ch. 25). Along any model through `p` with score
`φ`, the pathwise derivative of `L` is `E_p[T_f(X, u) φ(X)]`, so

  ```
  ψ_p(x) ≝ T_f(x, u) − E_p[T_f(X, u)]
  ```

is the gradient of `L` at `p` in the nonparametric tangent space
`{φ : E_p[φ(X)] = 0, E_p[φ(X)²] < ∞}`: its influence function. That space
holds every score, so its gradient is unique (the claim). REINFORCE's weight
is the influence function plus a baseline; centring removes the baseline.

Geometrically, the tangent to `u ↦ f(x, u)` at `u` splits `f(x, u)` into two
parts,

  ```
  f(x, u) = (f(x, u) − u ∂_u f(x, u)) + ∂_u f(x, u) ⋅ u
          =         intercept         +    slope    ⋅ u
  ```

1. Intercept `T_f(x, u) = f(x, u) − u ∂_u f(x, u)`: the marginal cost of model
   mass at `x`.
2. Slope `∂_u f(x, u)`: the marginal cost of target mass at `x`.

The split is Euler's theorem. With `m = p(x; θ)` and `n = p̂(x)`, the
integrand `F(n, m) ≝ m f(x, n / m)` is homogeneous of degree 1 in `(n, m)`,
so `F(n, m) = m ∂_m F(n, m) + n ∂_n F(n, m)`. The chain rule through
`u = n / m` gives `∂_m F(n, m) = T_f(x, u)` and `∂_n F(n, m) = ∂_u f(x, u)`;
dividing by `m` gives `f(x, u) = T_f(x, u) + ∂_u f(x, u) ⋅ u`.

REINFORCE weights the score by the intercept because `θ` moves only model
mass; target mass is fixed. See [Csiszar f-divergences](csiszar.md) for the
derivation.

The two factors are the chain rule in function space. `θ` changes `L(θ)` only
through `p(·; θ)`, so

  ```
  ∇_θ L(θ) = ∫ (δL/δp)(x) ∇_θ p(x; θ) dx = E_{p(X; θ)}[(δL/δp)(X) s(X; θ)]
  ```

where `(δL/δp)(x)` is the functional derivative: the change in `L(θ)` per
unit of model mass added at `x`. For `L(θ) = ∫ p f(x, p̂ / p) dx` it is
`∂_m [m f(x, n / m)] = T_f(x, u)` (see [Csiszar f-divergences](csiszar.md)).
So:

1. `T_f(x, u)` carries the loss: `f(x, u)`, `A(x)`, and `p̂(x)`. Methods
   differ only here.
2. `s(x; θ)` carries the model: how `θ` moves mass. No method changes it.
3. `δL/δp` is defined only up to a constant, since `∫ ∇_θ p(x; θ) dx = 0`:
   mass added at `x` is removed elsewhere. That constant is the baseline
   (consequence 1).
4. The [pathwise form](#pathwise-form) uses the same `δL/δp`, moving samples
   instead of mass; its weight comes from `∇_x T_f(x, u)`.

## Choices of f

Every `f(x, u)` is a loss: training minimizes `L(θ) ≝ E_{p(X; θ)}[f(X, u)]`
by stepping against

  ```
  ∇_θ L(θ) = E_{p(X; θ)}[T_f(X, u) s(X; θ)],   u = p̂(X) / p(X; θ)
  ```

A method is three choices: a loss shape `f(x, u)`, a reward `A(x)` (which
enters as `−A(x)`, since `f(x, u)` is minimized), and a target `p̂(x)`. The
table below lists the distinct `f(x, u)`; the [Methods](#methods) table then
names each method's choices.

| name                 | reads  | `f(x, u)`                                                | `T_f(x, u)`                             |
|----------------------|--------|----------------------------------------------------------|-----------------------------------------|
| constant             | none   | `1`                                                      | `1`                                     |
| linear               | `A`    | `−A(x) + b(x)`                                           | `−A(x) + b(x)`                          |
| forward KL           | `u`    | `u log u`                                                | `−u`                                    |
| reverse KL           | `u`    | `−log u`                                                 | `1 − log u`                             |
| Amari α, α ∉ {0, 1}  | `u`    | `(u^α − 1) / (α(α − 1))`                                 | `−u^α / α − 1 / (α(α − 1))`             |
| Jensen–Shannon       | `u`    | `u log u − (1 + u) log(1 + u)`                           | `−log(1 + u)`                           |
| arithmetic–geometric | `u`    | `(1 + u) log((1 + u) / √u)`                              | `log(1 + u) − ½ log u − ½(u − 1)`       |
| total variation      | `u`    | `½ \|u − 1\|`                                            | `−½ sign(u − 1)`                        |
| Pearson χ²           | `u`    | `(u − 1)²`                                               | `1 − u²`                                |
| Neyman-form χ²       | `u`    | `u² − 1`                                                 | `−1 − u²`                               |
| squared Hellinger    | `u`    | `(√u − 1)²`                                              | `1 − √u`                                |
| triangular           | `u`    | `(u − 1)² / (1 + u)`                                     | `−(u − 1)(3u + 1) / (1 + u)²`           |
| t-power              | `u`    | `κ (u^t − 1)`                                            | `κ ((1 − t) u^t − 1)`                   |
| exp-abs-log          | `u`    | `exp(\|log u\|) − 1`                                     | `−1` if `u > 1`, `2 / u − 1` if `u < 1` |
| Jeffreys             | `u`    | `½ (u − 1) log u`                                        | `½ (1 − u − log u)`                     |
| modified GAN         | `u`    | `log(1 + u) − log u`                                     | `log((1 + u) / u) + 1 / (1 + u)`        |
| dual of `g`          | `u`    | `u g(1 / u)`                                             | `g′(1 / u)`                             |
| symmetrized `g`      | `u`    | `½ g(u) + ½ u g(1 / u)`                                  | `½ (T_g(u) + g′(1 / u))`                |
| KL-regularized       | `A, u` | `−A(x) + β ⋅` reverse KL                                 | `−A(x) − β log u + β`                   |
| weighted MLE         | `A, u` | `w(x) ⋅` forward KL                                      | `−w(x) u`                               |
| clipped              | `A, u` | `−min(A(x), m(u) A(x))`                                  | `−A(x) ⋅ 1[unclipped]`                  |
| KL-penalized         | `A, u` | `−A(x) + β ⋅` forward KL                                 | `−A(x) − β u`                           |
| clipped + k3         | `A, u` | clipped `+ β ⋅` (reverse KL `+ (v − 1)`) at `v = k(x) u` | `−A(x) ⋅ 1[unclipped] − β log(k(x) u)`  |
| g-regularized        | `A, u` | `−A(x) + β g(u)`                                         | `−A(x) + β T_g(u)`                      |
| weighted g           | `A, u` | `w(x) g(u)`                                              | `w(x) T_g(u)`                           |
| clipped + g          | `A, u` | clipped `+ β g(k(x) u)`                                  | `−A(x) ⋅ 1[unclipped] + β T_g(k(x) u)`  |

`reads` names what `f(x, u)` depends on. Rows that do not read `u` are the
`p̂ ≡ 0` slice: `u ≡ 0` and `T_f(x, 0) = f(x, 0)`. Rows that read only `u` are
divergences `D_f[p̂, p(·; θ)]` (see [Csiszar f-divergences](csiszar.md)); they
are the `A ≡ 0` slice.

Hybrid rows are built from the rows above them, and their `T_f(x, u)`
follows by three rules. For `f₁(x, u)`, `f₂(x, u)`, `h(u)`, and `c(x)`, `k(x)`
free of `u`,

  ```
  T_{f₁ + f₂}(x, u)   = T_{f₁}(x, u) + T_{f₂}(x, u)
  T_{c f₁}(x, u)      = c(x) T_{f₁}(x, u)
  T_{h(k u)}(x, u)    = T_h(k(x) u)
  ```

where the first two hold because `f ↦ f − u ∂_u f` is linear, and the third
because `u ∂_u h(k(x) u) = (k(x) u) h′(k(x) u)`. E.g., KL-penalized is
`−A(x) + β u log u`, so its `T_f(x, u)` is `−A(x) + β (−u)`. In the clipped
row, `f(x, u)` is `−A(x)` (linear) or `−m(u) A(x)` with `m(u)` a multiple of
`u` (whose `T_f(x, u)` is `0`). The last three rows generalize the hybrids:
KL-regularized, weighted MLE and clipped + k3 are `g-regularized`,
`weighted g` and `clipped + g` with `g` reverse KL, forward KL, and
reverse KL `+ (v − 1)`. Each hybrid's `T_f(x, u)` was checked with
sympy.

## Methods

| method                                                                                            | `f(x, u)`                                                        | `A(x)`                                    | `p̂(x) = u ⋅ p(x; θ)`                  | unpublished                                            |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------|---------------------------------------|--------------------------------------------------------|
| [REINFORCE][REINFORCE]                                                                            | linear, `b(x) = b` constant                                      | any                                       | `0`                                   |                                                        |
| [RLOO][RLOO]                                                                                      | linear, `b(x)` leave-one-out                                     | `r(x)`                                    | `0`                                   |                                                        |
| MLE                                                                                               | forward KL                                                       | `0`                                       | data                                  |                                                        |
| VI (ELBO)                                                                                         | reverse KL                                                       | `0`                                       | unnormalized posterior                |                                                        |
| f-VI ([f-VI][f-VI], [tail-adaptive][tail-f-VI])                                                   | any ratio-only `g`                                               | `0`                                       | unnormalized posterior                |                                                        |
| [f-DPG][f-DPG]                                                                                    | dual of `g`                                                      | `0`                                       | target, e.g. `π_ref(x) exp(r(x) / β)` |                                                        |
| [f-GAN][f-GAN]                                                                                    | any ratio-only `g`                                               | `0`                                       | data, via `u ≈ D(x) / (1 − D(x))`     |                                                        |
| [RLHF][RLHF]                                                                                      | KL-regularized                                                   | `r(x)`, reward model                      | `π_ref(x)`                            |                                                        |
| [RLVR][RLVR]                                                                                      | KL-regularized                                                   | `1[verifier accepts x]`                   | `π_ref(x)`                            |                                                        |
| entropy-regularized                                                                               | KL-regularized                                                   | any                                       | `1`                                   |                                                        |
| `g`-regularized RLHF ([Wu et al. 2026][fRLHF], [χPO][chiPO])                                      | g-regularized                                                    | `r(x)`, reward model                      | `π_ref(x)`                            |                                                        |
| [AWR][AWR]                                                                                        | weighted MLE, `w(x) = exp(A(x) / β)`                             | advantage from a critic                   | data                                  |                                                        |
| [MPO][MPO] M-step                                                                                 | weighted MLE, `w(x) = exp(A(x) / η) / Z`                         | Q-value from a critic                     | `p_old(x)`                            |                                                        |
| [ReST][ReST], [RAFT][RAFT]                                                                        | weighted MLE, `w(x) = 1[A(x) ≥ τ]`                               | `r(x)`                                    | `p_old(x)`                            |                                                        |
| [PPO][PPO]                                                                                        | clipped, `ε_low = ε_high = ε`                                    | GAE, from a critic                        | `p_old(x)`                            |                                                        |
| [PPO][PPO] with KL penalty                                                                        | KL-penalized                                                     | GAE, from a critic                        | `p_old(x)`                            |                                                        |
| [DAPO][DAPO]                                                                                      | clipped, `ε_low < ε_high`                                        | `(r(x) − mean_G r) / std_G r`             | `p_old(x)`                            |                                                        |
| [GRPO][GRPO]                                                                                      | clipped + k3, `ε_low = ε_high = ε`                               | `(r(x) − mean_G r) / std_G r`             | `p_old(x)`                            |                                                        |
| [Dr. GRPO][Dr-GRPO]                                                                               | clipped, `ε_low = ε_high = ε`                                    | `r(x) − mean_G r`                         | `p_old(x)`                            |                                                        |
| [FRPO][FRPO]                                                                                      | clipped + g, `ε_low = ε_high = ε`                                | `(r(x) − mean_G r) / std_G r`             | `p_old(x)`                            |                                                        |
| PPO-GAN ([GAIL][GAIL], [f-MAX][f-MAX], [f-GAIL][f-GAIL])                                          | clipped, `ε_low = ε_high = ε`                                    | `−T_g(D(x) / (1 − D(x)))`                 | `p_old(x)`                            |                                                        |
| [GKD][GKD]                                                                                        | Jensen–Shannon (skewed), on student samples                      | `0`                                       | `p_teacher(x)`                        |                                                        |
| [RAL][RAL]                                                                                        | linear                                                           | co-trained LLM judge vs reference answers | `0`                                   |                                                        |
| on-policy f-distillation ([f-distill][f-distill], [GKD][GKD], [PowerOPD][PowerOPD], [ABKD][ABKD], [CADENCE][CADENCE]) | clipped, `ε_low = ε_high = ε`                                    | `−T_g(p_teacher(x) / p_old(x))`           | `p_old(x)`                            |                                                        |
| divergence-regularized RLVR ([DPH-RL][DPH-RL], [APO][APO])                                        | g-regularized, `g` forward KL, JS or Amari α                     | `1[verifier accepts x]`                   | `π_ref(x)`                            |                                                        |
| α-weighted MLE                                                                                    | weighted g, `g` Amari α, `w(x) = 1[A(x) ≥ τ]` or `exp(A(x) / β)` | `r(x)` or advantage                       | `p_old(x)`                            | ✓ (close: [IRIS][IRIS], [Tsallis offline RL][Tsallis]) |

`unpublished` marks rows not found by a targeted literature search
(Semantic Scholar + OpenAlex, 2026-11-11), run in each row's home field's
terms: RLHF and RLVR regularization, imitation learning, knowledge
distillation, reward-weighted regression. A ✓ is absence of evidence, not
proof of novelty.

where,

  ```
  A(x)            ≝ reward or advantage of x, per the Methods table
  r(x)            ≝ scalar reward of x (reward model or verifier)
  b(x)            ≝ baseline, free of x's own sample: a constant b, or for RLOO the
                      mean reward of the other samples for the same prompt
  w(x)            ≝ nonnegative sample weight, per the Methods table
  π_ref, p_old    ≝ reference and pre-update policies, frozen (no gradient)
  Z               ≝ E_{p_old}[exp(A(X) / η)], the per-prompt normalizer
  p_teacher(x)    ≝ a frozen teacher model's density
  D(x)            ≝ discriminator, P(x drawn from p̂ | x), frozen during each policy step;
                      trained against p(·; θ) (f-GAN) or p_old (PPO-GAN)
  x₁, …, x_G      ≝ G responses sampled from p_old for the same prompt as x
  mean_G r        ≝ (1 / G) Σᵢ r(xᵢ)
  std_G r         ≝ ((1 / G) Σᵢ (r(xᵢ) − mean_G r)²)^½
  ρ               ≝ 1 / u = p(x; θ) / p_old(x)          # PPO's ratio
  m(u)            ≝ clip(1, (1 − ε_low) u, (1 + ε_high) u)
  1[unclipped]    ≝ 1 if min(A(x), m(u) A(x)) = A(x), else 0
  k(x)            ≝ π_ref(x) / p_old(x),  so  k(x) u = π_ref(x) / p(x; θ)
  g(u)            ≝ any convex function on (0, ∞), as in the ratio table
  T_g(u)          ≝ g(u) − u g′(u)                       # as in csiszar.md
  1[·]            ≝ 1 if the condition holds, else 0
  H(p(·; θ))      ≝ −E_{p(X; θ)}[log p(X; θ)], the entropy
  ```

`β, η > 0`, `ε, ε_low, ε_high > 0`, `τ` a threshold, and `κ = −1` if
`0 < t < 1`, else `+1`. For a policy, `x` is a whole response (or trajectory)
and every density is conditioned on the prompt (or initial state), omitted
above. Every `T_f(x, u)` was checked with sympy.

Nine consequences:

1. Constant: the score has zero mean, since densities integrate to one,

     ```
     E_{p(X; θ)}[s(X; θ)] = ∫ p(x; θ) ∇_θ log p(x; θ) dx
                          = ∫ ∇_θ p(x; θ) dx
                          = ∇_θ ∫ p(x; θ) dx
                          = ∇_θ 1
                          = 0
     ```

   so any constant added to `T_f(x, u)` is a baseline and leaves the gradient
   unchanged. Hence `b` in the linear row changes no gradient. RLOO's `b(x)`
   is free of `x`'s own sample, so it is a baseline too.
2. Adding `c (u − 1)` to `f(x, u)` adds `−c` to `T_f(x, u)`, a baseline. Hence
   the two χ² rows give the same gradient.
3. MLE: with `p̂(x) = (1 / n) Σᵢ δ(x − xᵢ)`, forward KL gives
   `E_{p(X; θ)}[−u s(X; θ)] = −(1 / n) Σᵢ s(xᵢ; θ)`.
   The gradient is exact, but a Monte Carlo estimate from `X ~ p(·; θ)` has
   variance growing with `u`; reverse KL's `T_f(x, u)` grows only as `log u`.
   [RAML][RAML] reads MLE and entropy-regularized RL as KL in opposite
   directions; [Tan et al. 2018][Tan] recover MLE as RL with a δ-reward.
4. KL-regularized: `L(θ) = −E_{p(X; θ)}[A(X)] + β KL(p(·; θ) ‖ π_ref)`, the
   RLHF objective. Dropping the baseline `β`, its per-sample weight is
   `−(A(x) − β log(p(x; θ) / π_ref(x)))`: the reward minus the KL penalty.
   With `p̂ = 1` the penalty is `−β H(p(·; θ))`, entropy regularization. RLHF
   and RLVR differ only in `A(x)`; with `β = 0` both are REINFORCE.
5. Weighted MLE: `∇_θ L(θ) = E_{p(X; θ)}[−w(X) u s(X; θ)] = −E_{p̂}[w(X) s(X; θ)]`,
   i.e., MLE on samples from `p̂` with weight `w(x) ≥ 0`. AWR, MPO, ReST and
   RAFT differ only in `w(x)` and `p̂`; the weight stays nonnegative so the
   loss stays a weighted forward KL.
6. Clipped: clipping lives in `f(x, u)`. With `ρ = 1 / u`, PPO's surrogate
   `E_{p_old}[−min(ρ A(X), clip(ρ, 1 − ε_low, 1 + ε_high) A(X))]` equals
   `E_{p(X; θ)}[−min(A(X), m(u) A(X))]`, since `u ⋅ clip(1 / u, a, b) =
   clip(1, a u, b u)` for `u > 0` (checked numerically on 10⁵ random
   `(A(x), u)`): PPO with `ρ = 1` and its clip bounds scaled by `u`. So the
   clipped row is the linear row `−A(x)` until `ρ` leaves
   `[1 − ε_low, 1 + ε_high]` in the direction `A(x)` favors; there it becomes
   `−(1 + ε_high) u A(x)` or `−(1 − ε_low) u A(x)`, a multiple of `u`, whose
   `T_f(x, u)` is `0`. Clipping zeroes the weight. PPO, DAPO, Dr. GRPO and
   PPO-GAN share this `f(x, u)` and differ in `A(x)` and the clip bounds.
7. Clipped + k3: GRPO's KL term is the k3 estimator
   `k(x) u − log(k(x) u) − 1`, which is reverse KL to `π_ref` plus
   `k(x) u − 1`, so its tangent intercept is
   `−β log(k(x) u) = β log(p(x; θ) / π_ref(x))`, the RLHF penalty without the
   baseline `β`.
8. f-GAN: when `p̂` is samples without a density, a discriminator estimates
   the ratio. At its optimum `D(x) = p̂(x) / (p̂(x) + p(x; θ))`, so
   `u = D(x) / (1 − D(x))`, and every ratio-only row becomes a GAN generator
   step. Jensen–Shannon gives `T_g(u) = −log(1 + u) = log(1 − D(x))`.
9. PPO-GAN: train `D(x)` against `p_old`, so `D(x) / (1 − D(x)) = p̂(x) / p_old(x)`,
   and run the clipped row with `A(x) = −T_g(p̂(x) / p_old(x))`. At
   `θ = θ_old`, `u = 1` and nothing is clipped, so
   `T_f(x, u) = −A(x) = T_g(p̂(x) / p(x; θ))`: the step starts on the exact
   `∇_θ D_g[p̂, p(·; θ)]`. With `g` Jensen–Shannon, `A(x) = −log(1 − D(x))`,
   the [GAIL][GAIL] cost with [TRPO][TRPO] replaced by clipping. Other `g`
   are [f-MAX][f-MAX] and [Ke et al. 2019][Ke]; [f-GAIL][f-GAIL] learns `g`. Forward KL
   gives `A(x) = D(x) / (1 − D(x))`, importance-weighted MLE in the spirit of
   [MaliGAN][MaliGAN].

Regularity: `p(x; θ)` differentiable in `θ` with support free of `θ`; `∇_θ`
passes through `∫`; `p(x; θ) > 0` wherever `p̂(x) > 0`; `f(x, u)`
differentiable in `u` (total variation and exp-abs-log fail at `u = 1`, a null
set when `u` has a density). The assumption fails when `u` averages
`p̂(x) / p(x; θ)` over several samples (importance-weighted bounds): then
`f(x, u)` reads `p(·; θ)` at points other than `X`. Three pieces sit outside
`f(x, u)`: GRPO's division by `std_G r`, which rescales each prompt by its
other samples; PPO's critic, which uses time structure inside `x`; and DAPO's
token-level loss, which divides by the token count of the whole batch. None
reads `θ`, so each is still a weight on the score: with `x = (x₁, …, x_T)`
and per-token score `s_t ≝ ∇_θ log p(x_t | x_{<t}; θ)`, the update is
`E[Σ_t w_t s_t]` with `w_t` built from rewards, the critic, and batch counts.
That is an estimating equation without an objective. The GAN rows' ratio is
different: it is exact only at the discriminator's optimum, so away from it
the weight is the right form with the wrong value. Pairwise offline methods
([DPO][DPO] and its variants, e.g. [f-PO][f-PO]) fall outside entirely: they differentiate `log p(x; θ)` at fixed data pairs, not
`E_{p(X; θ)}[f(X, u)]`.

## Pathwise form

Diffusion models give scores `∇_x log p(x)`, not densities, so the identity
above, which needs `u`, does not apply. When the sample is a differentiable
function of noise, the same gradient has a second form that uses scores.

Let,

  ```
  Z ∈ ℝᵈ            ~ p_Z, noise free of θ
  G(z; θ) ∈ ℝᵈ      ≝ generator; X = G(Z; θ) has density p(x; θ)
  J_G(z; θ) ∈ ℝᵈˣᵏ  ≝ ∂_θ G(z; θ), the Jacobian in θ ∈ ℝᵏ
  θ₀                ≝ θ held fixed (no gradient)
  ∇_x log u(x; θ)   ≝ ∇_x log p̂(x) − ∇_x log p(x; θ)    # score difference
  h_f(u)            ≝ −u ∂_u T_f(u) = u² ∂²_u f(u)
  ```

and take a ratio-only `f(u)` (the `A ≡ 0` slice). Then,

  ```
  ∇_θ L(θ) = ∫ T_f(u(x; θ₀)) ∇_θ p(x; θ) dx
           = ∇_θ E_{p_Z}[T_f(u(G(Z; θ); θ₀))]
           = E_{p_Z}[J_G(Z; θ)ᵀ ∂_u T_f(u) ∇_x u(X; θ₀)]
           = E_{p_Z}[−h_f(u) J_G(Z; θ)ᵀ ∇_x log u(X; θ)]
  ```

where the first step is the identity above written as an integral, the second
holds because `∫ g(x) ∇_θ p(x; θ) dx = ∇_θ E_{p_Z}[g(G(Z; θ))]` for any `g`
free of `θ`, the third is the chain rule through `X = G(Z; θ)`, and the
fourth uses `∇_x u = u ∇_x log u`. The score-function form moves the density
at a fixed sample; the pathwise form moves the sample under a fixed density.
Where both apply they are the same gradient; the weight changes from the
intercept `T_f(u)` to `h_f(u)`, minus `u` times its slope.

Six consequences:

1. Per row, `h_f(u)` is `1` (reverse KL), `u` (forward KL), `u / (1 + u)`
   (Jensen–Shannon), `½ √u` (squared Hellinger), `2 u²` (Pearson χ²), and
   `½ (1 + u)` (Jeffreys), checked with sympy. These match
   [Xu et al. 2025][f-distill-diff] Table 1 up to the scale of `f`.
2. Reverse KL alone has constant `h_f`, so its gradient needs only the score
   difference, no value of `u`. That is [VSD][VSD] and [DMD][DMD]: a teacher
   diffusion model supplies `∇_x log p̂`, a fake-score network trained on the
   generator's samples supplies `∇_x log p(·; θ)`. Every other `f` needs `u`
   too, which Xu et al. estimate with a discriminator, `u ≈ D(x) / (1 − D(x))`
   as in consequence 8 above.
3. Any continuous `h(u) ≥ 0` is some row's weight: `∂²_u f(u) = h(u) / u² ≥ 0`
   integrates to a convex `f(u)` (Xu et al. Prop. 1). This is the pathwise
   twin of choosing the weight first.
4. For diffusion, `p̂` and `p(·; θ)` are their noised versions at noise level
   `σ(t)`, `X = G(Z; θ) + σ(t) ε` with `ε ~ N(0, I)`, and the loss averages
   over `t`.
5. Training a diffusion model on data is the forward-KL row. With likelihood
   weighting, the denoising score-matching loss upper-bounds
   `−E_{p̂}[log p(X; θ)]` ([Song et al. 2021][Song-MLE]), so it is MLE on
   samples from `p̂`, as in consequence 3 above.
6. RL fine-tuning of a diffusion model uses the score-function form: each
   denoising step is a Gaussian policy with tractable `log p`, so the Methods
   rows apply to the chain of steps. [DDPO][DDPO] is the linear row (its
   score-function variant) or the clipped row (its importance-sampled
   variant); [DPOK][DPOK] is KL-regularized.

Regularity: `p_Z` free of `θ`; `G(z; θ)` differentiable in `θ`; `x`
continuous; `p̂(x)` and `p(x; θ)` differentiable in `x`; `∇_θ` passes through
`E_{p_Z}`. Discrete `x` (tokens, actions) has no pathwise form, and rows with a
reward need `∇_x A(x)`, which verifiers and environments do not give. Learned
scores, like the GAN rows' ratio, give the right form with the wrong value
away from their optimum.

## References

- [REINFORCE][REINFORCE]: Williams 1992, "Simple statistical gradient-following
  algorithms for connectionist reinforcement learning", Machine Learning.
- [f-DPG][f-DPG]: Go et al. 2023, "Aligning Language Models with Preferences
  through f-divergence Minimization", ICML. Thm. 1 is the dual row.
- [RLHF][RLHF]: Ouyang et al. 2022, "Training language models to follow
  instructions with human feedback", NeurIPS, eq. 2; the per-sample
  reward minus `β log(π_θ / π_ref)` is from [Ziegler et al. 2019][Ziegler].
- [RLVR][RLVR]: Lambert et al. 2024, "Tülu 3: Pushing Frontiers in Open
  Language Model Post-Training".
- [PPO][PPO]: Schulman et al. 2017, "Proximal Policy Optimization
  Algorithms"; its advantage is [GAE][GAE] (Schulman et al. 2015).
- [GRPO][GRPO]: Shao et al. 2024, "DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning in Open Language Models"; its KL term is
  [k3][k3] (Schulman 2020, "Approximating KL Divergence").
- [AWR][AWR]: Peng et al. 2019, "Advantage-Weighted Regression".
- [MPO][MPO]: Abdolmaleki et al. 2018, "Maximum a Posteriori Policy
  Optimisation", ICLR.
- [ReST][ReST]: Gulcehre et al. 2023, "Reinforced Self-Training (ReST) for
  Language Modeling".
- [RAFT][RAFT]: Dong et al. 2023, "RAFT: Reward rAnked FineTuning for
  Generative Foundation Model Alignment", TMLR.
- [RAML][RAML]: Norouzi et al. 2016, "Reward Augmented Maximum Likelihood for
  Neural Structured Prediction", NeurIPS. Eqs. 5-8: MLE and entropy-regularized
  RL minimize KL to an exponentiated reward in opposite directions.
- [Tan et al. 2018][Tan]: "Connecting the Dots Between MLE and RL for Sequence
  Prediction". §3.2: MLE is entropy-regularized policy optimization with a
  δ-reward, in the limit `α → 0`.
- [Lv et al. 2025][UPGE]: "Towards a Unified View of Large Language Model
  Post-Training". A Unified Policy Gradient Estimator: mask, reference
  denominator, advantage, likelihood gradient.
- [DFT][DFT]: Wu et al. 2025, "On the Generalization of SFT: A Reinforcement
  Learning Perspective with Reward Rectification", ICLR. SFT's gradient as
  policy gradient with an inverse-probability reward.
- [van der Vaart 1998][vdV]: "Asymptotic Statistics", Cambridge University
  Press, ch. 5, "M- and Z-Estimators"; ch. 25, "Semiparametric Models"
  (influence functions, canonical gradients).
- [RLOO][RLOO]: Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE
  Style Optimization for Learning from Human Feedback in LLMs".
- [Dr. GRPO][Dr-GRPO]: Liu et al. 2025, "Understanding R1-Zero-Like
  Training: A Critical Perspective".
- [DAPO][DAPO]: Yu et al. 2025, "DAPO: An Open-Source LLM Reinforcement
  Learning System at Scale", NeurIPS.
- [DPO][DPO]: Rafailov et al. 2023, "Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model", NeurIPS.
- [FRPO][FRPO]: Yao et al. 2026, "Future-KL Regularized GRPO: Process-Level
  Credit Assignment from f-Divergence Regularization". GRPO with token-wise
  f-divergence regularization.
- [CADENCE][CADENCE]: Kumar & Jha 2026, "CADENCE: Closing the Reasoning Gap
  via Coverage-Adaptive On-Policy Distillation". On-policy distillation with
  mixtures of forward and reverse KL.
- [Wu et al. 2026][fRLHF]: "f-Divergence Regularized RLHF: Two Tales of
  Sampling and Unified Analyses". Online RLHF with any f-divergence regularizer.
- [χPO][chiPO]: Huang et al. 2024, "Correcting the Mythos of KL-Regularization:
  Direct Alignment without Overoptimization via Chi-Squared Preference
  Optimization". χ² in place of KL.
- [GKD][GKD]: Agarwal et al. 2023, "On-Policy Distillation of Language Models:
  Learning from Self-Generated Mistakes", ICLR. Generalized JSD on student samples.
- [RAL][RAL]: Bou et al. 2026, "Learning to Reason from Reference Samples
  Alone: Reinforced Adversarial Learning for Modern LLM Post-Training". A
  co-trained judge as the sole reward.
- [IRIS][IRIS]: Liao et al. 2026, "IRIS: Interpolative Rényi Iterative
  Self-play for Large Language Model Fine-Tuning".
- [f-MAX][f-MAX]: Ghasemipour et al. 2019, "A Divergence Minimization
  Perspective on Imitation Learning Methods", CoRL. Adversarial imitation with
  any f-divergence.
- [Ke et al. 2019][Ke]: "Imitation Learning as f-Divergence Minimization",
  WAFR. Reverse-KL adversarial imitation.
- [f-GAIL][f-GAIL]: Zhang et al. 2020, "f-GAIL: Learning f-Divergence for
  Generative Adversarial Imitation Learning", NeurIPS.
- [f-distill][f-distill]: Wen et al. 2023, "f-Divergence Minimization for Sequence-Level Knowledge
  Distillation", ACL.
- [PowerOPD][PowerOPD]: Zhao et al. 2026, "PowerOPD: Stabilizing On-Policy Distillation with
  Bounded Power Transformation", EMNLP. Box-Cox (α) rewards for on-policy distillation.
- [ABKD][ABKD]: Wang et al. 2025, "ABKD: Pursuing a Proper Allocation of the Probability
  Mass in Knowledge Distillation via α-β-Divergence", ICML.
- [DPH-RL][DPH-RL]: Li et al. 2025, "The Choice of Divergence: A Neglected Key to Mitigating
  Diversity Collapse in Reinforcement Learning with Verifiable Reward", ICLR.
  Forward KL and JS to the initial policy in RLVR.
- [APO][APO]: Wang 2025, "APO: Alpha-Divergence Preference Optimization". Amari α
  interpolating forward and reverse KL, on math RL.
- [Tsallis offline RL][Tsallis]: Zhu et al. 2024, "Offline Reinforcement
  Learning via Tsallis Regularization", TMLR. Tsallis (α-family) regularized
  policy extraction.
- [Xu et al. 2025][f-distill-diff]: Xu, Nie & Vahdat, "One-step Diffusion
  Models with f-Divergence Distribution Matching". Thm. 1: the pathwise
  gradient with weight `f″(u) u²`.
- [VSD][VSD]: Wang et al. 2023, "ProlificDreamer: High-Fidelity and Diverse
  Text-to-3D Generation with Variational Score Distillation", NeurIPS.
- [DMD][DMD]: Yin et al. 2024, "One-Step Diffusion with Distribution Matching
  Distillation", CVPR.
- [Song et al. 2021][Song-MLE]: "Maximum Likelihood Training of Score-Based
  Diffusion Models", NeurIPS.
- [DDPO][DDPO]: Black et al. 2024, "Training Diffusion Models with
  Reinforcement Learning", ICLR.
- [DPOK][DPOK]: Fan et al. 2023, "DPOK: Reinforcement Learning for
  Fine-tuning Text-to-Image Diffusion Models", NeurIPS.
- [f-GAN][f-GAN]: Nowozin et al. 2016, "f-GAN: Training Generative Neural
  Samplers using Variational Divergence Minimization", NeurIPS.
- [GAIL][GAIL]: Ho & Ermon 2016, "Generative Adversarial Imitation Learning",
  NeurIPS; its policy step is [TRPO][TRPO] (Schulman et al. 2015).
- [MaliGAN][MaliGAN]: Che et al. 2017, "Maximum-Likelihood Augmented Discrete
  Generative Adversarial Networks".
- [Tail-adaptive f-VI][tail-f-VI]: Wang et al. 2018, "Variational Inference
  with Tail-adaptive f-Divergence", NeurIPS. Prop. 4.2:
  `∇_θ D_f = −E_q[ρ_f(w) ∇_θ log q]` with `ρ_f(t) = f′(t) t − f(t)`.
- [f-VI][f-VI]: Wan et al. 2020, "f-Divergence Variational Inference",
  NeurIPS.
- [f-PO][f-PO]: Han et al. 2024, "f-PO: Generalizing Preference Optimization
  with f-divergence Minimization", AISTATS.

[REINFORCE]: https://doi.org/10.1007/BF00992696
[f-DPG]: https://arxiv.org/abs/2302.08215
[RLHF]: https://arxiv.org/abs/2203.02155
[Ziegler]: https://arxiv.org/abs/1909.08593
[RLVR]: https://arxiv.org/abs/2411.15124
[PPO]: https://arxiv.org/abs/1707.06347
[GAE]: https://arxiv.org/abs/1506.02438
[GRPO]: https://arxiv.org/abs/2402.03300
[k3]: http://joschu.net/blog/kl-approx.html
[AWR]: https://arxiv.org/abs/1910.00177
[MPO]: https://arxiv.org/abs/1806.06920
[ReST]: https://arxiv.org/abs/2308.08998
[RAFT]: https://arxiv.org/abs/2304.06767
[RAML]: https://arxiv.org/abs/1609.00150
[Tan]: https://arxiv.org/abs/1811.09740
[UPGE]: https://arxiv.org/abs/2509.04419
[DFT]: https://arxiv.org/abs/2508.05629
[vdV]: https://doi.org/10.1017/CBO9780511802256
[RLOO]: https://arxiv.org/abs/2402.14740
[Dr-GRPO]: https://arxiv.org/abs/2503.20783
[DAPO]: https://arxiv.org/abs/2503.14476
[DPO]: https://arxiv.org/abs/2305.18290
[f-GAN]: https://arxiv.org/abs/1606.00709
[f-distill-diff]: https://arxiv.org/abs/2502.15681
[VSD]: https://arxiv.org/abs/2305.16213
[DMD]: https://arxiv.org/abs/2311.18828
[Song-MLE]: https://arxiv.org/abs/2101.09258
[DDPO]: https://arxiv.org/abs/2305.13301
[DPOK]: https://arxiv.org/abs/2305.16381
[Tsallis]: https://openreview.net/forum?id=HNqEKZDDRc
[f-distill]: https://arxiv.org/abs/2307.15190
[PowerOPD]: https://arxiv.org/abs/2606.17199
[ABKD]: https://arxiv.org/abs/2505.04560
[DPH-RL]: https://arxiv.org/abs/2509.07430
[APO]: https://arxiv.org/abs/2512.22953
[f-MAX]: https://arxiv.org/abs/1911.02256
[Ke]: https://arxiv.org/abs/1905.12888
[f-GAIL]: https://arxiv.org/abs/2010.01207
[fRLHF]: https://arxiv.org/abs/2605.06977
[chiPO]: https://arxiv.org/abs/2407.13399
[GKD]: https://arxiv.org/abs/2306.13649
[RAL]: https://arxiv.org/abs/2610.38851
[IRIS]: https://arxiv.org/abs/2604.20933
[FRPO]: https://arxiv.org/abs/2601.10201
[CADENCE]: https://arxiv.org/abs/2607.16955
[GAIL]: https://arxiv.org/abs/1606.03476
[TRPO]: https://arxiv.org/abs/1502.05477
[MaliGAN]: https://arxiv.org/abs/1702.07983
[tail-f-VI]: https://arxiv.org/abs/1810.11943
[f-VI]: https://arxiv.org/abs/2009.13093
[f-PO]: https://arxiv.org/abs/2410.21662
