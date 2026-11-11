# REINFORCE

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

Geometrically, the tangent to `u ↦ f(x, u)` at `u` splits `f(x, u)` into two
parts,

  ```
  f(x, u) = (f(x, u) − u ∂_u f(x, u)) + ∂_u f(x, u) ⋅ u
          =         intercept         +    slope    ⋅ u
  ```

1. Intercept `T_f(x, u) = f(x, u) − u ∂_u f(x, u)`: the marginal cost of model
   mass at `x`.
2. Slope `∂_u f(x, u)`: the marginal cost of target mass at `x`.

REINFORCE weights the score by the intercept because `θ` moves only model
mass; target mass is fixed. See [Csiszar f-divergences](csiszar.md) for the
derivation.

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
| g-regularized        | `A, u` | `−A(x) + β ⋅ g`                                          | `−A(x) + β T_g(u)`                      |
| weighted g           | `A, u` | `w(x) ⋅ g`                                               | `w(x) T_g(u)`                           |
| clipped + g          | `A, u` | clipped `+ β ⋅ g` at `v = k(x) u`                        | `−A(x) ⋅ 1[unclipped] + β T_g(k(x) u)`  |

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

| method                                    | `f(x, u)`                                                        | `A(x)`                          | `p̂(x) = u ⋅ p(x; θ)`                  | unpublished                      |
|-------------------------------------------|------------------------------------------------------------------|---------------------------------|---------------------------------------|----------------------------------|
| [REINFORCE][REINFORCE]                    | linear, `b(x) = b` constant                                      | any                             | `0`                                   |                                  |
| [RLOO][RLOO]                              | linear, `b(x)` leave-one-out                                     | `r(x)`                          | `0`                                   |                                  |
| MLE                                       | forward KL                                                       | `0`                             | data                                  |                                  |
| VI (ELBO)                                 | reverse KL                                                       | `0`                             | unnormalized posterior                |                                  |
| [f-DPG][f-DPG]                            | dual of `g`                                                      | `0`                             | target, e.g. `π_ref(x) exp(r(x) / β)` |                                  |
| [f-GAN][f-GAN]                            | any ratio-only `g`                                               | `0`                             | data, via `u ≈ D(x) / (1 − D(x))`     |                                  |
| [RLHF][RLHF]                              | KL-regularized                                                   | `r(x)`, reward model            | `π_ref(x)`                            |                                  |
| [RLVR][RLVR]                              | KL-regularized                                                   | `1[verifier accepts x]`         | `π_ref(x)`                            |                                  |
| entropy-regularized                       | KL-regularized                                                   | any                             | `1`                                   |                                  |
| [AWR][AWR]                                | weighted MLE, `w(x) = exp(A(x) / β)`                             | advantage from a critic         | data                                  |                                  |
| [MPO][MPO] M-step                         | weighted MLE, `w(x) = exp(A(x) / η) / Z`                         | Q-value from a critic           | `p_old(x)`                            |                                  |
| [ReST][ReST], [RAFT][RAFT]                | weighted MLE, `w(x) = 1[A(x) ≥ τ]`                               | `r(x)`                          | `p_old(x)`                            |                                  |
| [PPO][PPO]                                | clipped, `ε_low = ε_high = ε`                                    | GAE, from a critic              | `p_old(x)`                            |                                  |
| [PPO][PPO] with KL penalty                | KL-penalized                                                     | GAE, from a critic              | `p_old(x)`                            |                                  |
| [DAPO][DAPO]                              | clipped, `ε_low < ε_high`                                        | `(r(x) − mean_G r) / std_G r`   | `p_old(x)`                            |                                  |
| [GRPO][GRPO]                              | clipped + k3, `ε_low = ε_high = ε`                               | `(r(x) − mean_G r) / std_G r`   | `p_old(x)`                            |                                  |
| [Dr. GRPO][Dr-GRPO]                       | clipped, `ε_low = ε_high = ε`                                    | `r(x) − mean_G r`               | `p_old(x)`                            |                                  |
| PPO-GAN ([GAIL][GAIL] for Jensen–Shannon) | clipped, `ε_low = ε_high = ε`                                    | `−T_g(D(x) / (1 − D(x)))`       | `p_old(x)`                            | ✓ for `g` ≠ Jensen–Shannon       |
| PPO-distillation                          | clipped, `ε_low = ε_high = ε`                                    | `−T_g(p_teacher(x) / p_old(x))` | `p_old(x)`                            | ✓ for `g` other than KL mixtures |
| α-RLVR                                    | Amari α                                                          | `0`                             | `π_ref(x) 1[verifier accepts x]`      | ✓                                |
| [FRPO][FRPO]                              | clipped + g, `ε_low = ε_high = ε`                                | `(r(x) − mean_G r) / std_G r`   | `p_old(x)`                            |                                  |
| α-weighted MLE                            | weighted g, `g` Amari α, `w(x) = 1[A(x) ≥ τ]` or `exp(A(x) / β)` | `r(x)` or advantage             | `p_old(x)`                            | ✓                                |

`unpublished` means not found in the references below; no exhaustive
literature search was done.

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
   the [GAIL][GAIL] cost with [TRPO][TRPO] replaced by clipping. Forward KL
   gives `A(x) = D(x) / (1 − D(x))`, importance-weighted MLE in the spirit of
   [MaliGAN][MaliGAN].

Regularity: `p(x; θ)` differentiable in `θ` with support free of `θ`; `∇_θ`
passes through `∫`; `p(x; θ) > 0` wherever `p̂(x) > 0`; `f(x, u)`
differentiable in `u` (total variation and exp-abs-log fail at `u = 1`, a null
set when `u` has a density). The assumption fails when `u` averages
`p̂(x) / p(x; θ)` over several samples (importance-weighted bounds): then
`f(x, u)` reads `p(·; θ)` at points other than `X`. Three pieces sit outside
`f(x, u)`: GRPO's division by `std_G r`, which rescales each prompt by its
other samples; PPO's critic, which uses time structure inside `x`; and the GAN
rows' ratio, which is exact only at the discriminator's optimum. DAPO's
token-level loss also sits outside, since it divides by the token count of
the whole batch. Pairwise offline methods ([DPO][DPO] and its variants) fall
outside entirely: they differentiate `log p(x; θ)` at fixed data pairs, not
`E_{p(X; θ)}[f(X, u)]`.

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
- [f-GAN][f-GAN]: Nowozin et al. 2016, "f-GAN: Training Generative Neural
  Samplers using Variational Divergence Minimization", NeurIPS.
- [GAIL][GAIL]: Ho & Ermon 2016, "Generative Adversarial Imitation Learning",
  NeurIPS; its policy step is [TRPO][TRPO] (Schulman et al. 2015).
- [MaliGAN][MaliGAN]: Che et al. 2017, "Maximum-Likelihood Augmented Discrete
  Generative Adversarial Networks".
- [Tail-adaptive f-VI][tail-f-VI]: Wang et al. 2018, "Variational Inference
  with Tail-adaptive f-Divergence", NeurIPS.
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
[RLOO]: https://arxiv.org/abs/2402.14740
[Dr-GRPO]: https://arxiv.org/abs/2503.20783
[DAPO]: https://arxiv.org/abs/2503.14476
[DPO]: https://arxiv.org/abs/2305.18290
[f-GAN]: https://arxiv.org/abs/1606.00709
[FRPO]: https://arxiv.org/abs/2601.10201
[CADENCE]: https://arxiv.org/abs/2607.16955
[GAIL]: https://arxiv.org/abs/1606.03476
[TRPO]: https://arxiv.org/abs/1502.05477
[MaliGAN]: https://arxiv.org/abs/1702.07983
[tail-f-VI]: https://arxiv.org/abs/1810.11943
[f-VI]: https://arxiv.org/abs/2009.13093
[f-PO]: https://arxiv.org/abs/2410.21662
