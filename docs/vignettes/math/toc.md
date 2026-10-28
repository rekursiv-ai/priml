# Vital Proofs

Each vignette proves one result in the style of [Cramér–Rao](cramer-rao.md):
the idea, then the core argument as aligned equation chains, then the fine
print (edge cases, cited steps, regularity).

## Probability

- Foundations
  - [Fundamental theorem of calculus](fundamental-theorem-calculus.md)
  - [Fubini–Tonelli](fubini-tonelli.md)
- Expectation and variance
  - [Jensen's inequality](jensen.md)
  - [Law of total expectation (tower property)](law-of-total-expectation.md)
  - [Law of total variance](law-of-total-variance.md)
- Concentration
  - [Markov, Chebyshev, Chernoff](markov-chebyshev-chernoff.md)
  - [Hoeffding's lemma and inequality](hoeffding.md)
  - [Azuma and McDiarmid (bounded differences)](azuma-mcdiarmid.md)
  - [Efron–Stein](efron-stein.md)
- High-dimensional probability
  - [Bernstein and sub-exponential tails](bernstein.md)
  - [Johnson–Lindenstrauss](johnson-lindenstrauss.md)
  - [ε-net covering and random matrix norms](epsilon-net.md)
  - [Gaussian concentration of Lipschitz functions](gaussian-lipschitz.md)
  - [Matrix Bernstein](matrix-bernstein.md)
- Limit theorems
  - [Weak law of large numbers](weak-law.md)
  - [Central limit theorem](clt.md)
  - [Borel–Cantelli](borel-cantelli.md)
  - [Slutsky and the delta method](delta-method-slutsky.md)
- Change of measure and Gaussians
  - [Change of variables](change-of-variables.md)
  - [Gaussian conditioning](gaussian-conditioning.md)
- Stein identities (integration by parts against a density)
  - [Stein's lemma](stein-lemma.md)
  - [Stein identity and score matching](stein-identity.md)
  - [Second-order Stein and Hermite polynomials](stein-second-order.md)
  - [Bonnet and Price](bonnet-price.md)
  - [Gaussian Poincaré inequality](gaussian-poincare.md)
- Gradient estimators
  - [Reparameterization vs score function](reparameterization-vs-score.md)
  - [Antithetic and orthogonal perturbations](antithetic-orthogonal.md)
  - [Control variates](control-variates.md)
- Martingales
  - [Optional stopping](optional-stopping.md)
  - [Doob's maximal inequality](doob-maximal.md)
  - [Doob decomposition](doob-decomposition.md)
  - [Martingale convergence](martingale-convergence.md)
- Markov chains and sampling
  - [Metropolis–Hastings detailed balance](metropolis-hastings.md)
  - [Perron–Frobenius and ergodic mixing](perron-frobenius-mixing.md)

## Information

- Divergences and entropy
  - [Gibbs' inequality](gibbs-inequality.md)
  - [Chain rule for KL](kl-chain-rule.md)
  - [Data processing inequality](data-processing.md)
  - [Pinsker's inequality](pinsker.md)
  - [Mutual information as KL](mutual-information.md)

- Variational representations
  - [f-divergence variational form](f-divergence-variational.md)
  - [Donsker–Varadhan](donsker-varadhan.md)
  - [Gibbs variational principle](gibbs-variational.md)
  - [Evidence lower bound](elbo.md)
  - [Kantorovich–Rubinstein duality](kantorovich-rubinstein.md)
- Coding and estimation limits
  - [Kraft inequality and source coding](kraft-source-coding.md)
  - [Fano's inequality](fano.md)

## Choosing a Distribution

Each picks the distribution a criterion makes extreme over a constraint set:
entropy or KL to a reference (uncertainty about `Y`), Fisher information
(how learnable a location is), or mutual information (how far data move a
prior). Some choose a data density, some a prior. Max entropy is the
I-projection onto moments from a uniform reference, and agrees with min
Fisher information on the Gaussian under a variance constraint. The least
favorable forms are saddle points: nature picks the noise or the parameter,
the statistician its best response, so a robust loss or a minimax rule is the
statistician's side of the game. The reference prior is the least favorable
prior for prediction under log loss.

- Data densities
  - [Maximum entropy and the exponential family](max-entropy.md)
  - [I-projection and exponential tilting](i-projection.md)
  - [Minimum Fisher information](min-fisher-information.md)
- Saddle points
  - [Least favorable distributions: probability floors and the Huber loss](least-favorable-density.md)
  - [Least favorable prior and minimax rules](least-favorable-prior.md)
- Priors
  - [Reference priors and Jeffreys](reference-priors.md)

## Optimization

Almost every first-order method is one majorization–minimization step:
minimize a model of `f` plus a penalty on moving
(majorization-minimization.md). A method is a point on four axes.

1. Geometry: the norm the step is bounded in. With a linear model, the step
   is that norm's linear minimization oracle,
   `argmin_{‖d‖≤1} ⟨g, d⟩`.

    ```
    norm                  zero-memory step       method
    ℓ₂                    −g / ‖g‖₂              normalized GD
    ℓ∞                    −sign(g)               sign descent
    spectral (per matrix) −U Vᵀ                  Muon
    ‖·‖_H                 −H⁻¹ g (∝)             Newton
    Bregman D_ψ           mirror map             mirror descent
    ```

2. Memory: how the norm is estimated from past gradients. Accumulating
   squared gradients learns the norm from data; forgetting them fixes it.

    ```
    memory                  vector (diagonal)   matrix (Kronecker)
    none                    sign descent        Muon
    exponential average     Adam, RMSProp       SOAP, EMA-Shampoo
    full sum, regret-opt.   AdaGrad             Shampoo
    ```

   So `Muon : Shampoo :: sign descent : AdaGrad`. The square root in both
   regret-optimal preconditioners equalizes per-direction regret. Geometry
   and memory are one axis seen twice: zero memory is a fixed norm, full
   memory a learned one.

3. Momentum: a linear filter on the steps (heavy ball, Nesterov, the `m_t`
   of Adam and Muon). It smooths the direction over time; it does not
   change the norm.

4. Influence: a nonlinearity `ψ` applied before averaging. The same operation
   at different granularities:

    ```
    one unit treated as a draw   ψ there                robust to outlying
    a residual coordinate        Huber / z cap          coordinates
    a sample's gradient          per-sample clip        samples
    a batch's gradient           global norm clip       batches
    a stretch of steps           Adam's √v (no clip)    nothing: rescales only
    ```

   Only a `ψ` at the outliers' granularity bounds their influence
   (polyak-tsypkin.md). Everything after the average controls step size,
   not who picks the direction.

A successor is a move along an axis its parent zeroed out or approximated:
Muon's descendants restore per-coordinate scale (memory), clip singular
values or logits (influence), or approximate the orthogonalization at
scale (geometry).

- Convex duality
  - [Fenchel–Young inequality](fenchel-young.md)
  - [Fenchel–Moreau theorem](fenchel-moreau.md)
  - [Lagrange duality via separating hyperplane](lagrange-duality.md)
  - [KKT conditions and Slater](kkt.md)
  - [Moreau decomposition and firm nonexpansiveness](moreau-decomposition.md)
- Convergence rates
  - [Descent lemma](descent-lemma.md)
  - [Gradient descent convergence](gd-convergence.md)
  - [SGD convergence](sgd-convergence.md)
  - [Nesterov acceleration](nesterov.md)
- Generalized descent
  - [A taxonomy of first-order methods](optimizer-taxonomy.md)
  - [Continuous-time optimization](continuous-time-optimization.md)
  - [Majorization–minimization](majorization-minimization.md)
  - [Gradient descent as proximal gradient](gd-as-proximal.md)
  - [Mirror descent](mirror-descent.md)
  - [Natural gradient](natural-gradient.md)
  - [Newton as proximal step](newton-as-proximal.md)
  - [Adam](adam.md)
  - [Muon](muon.md)
- Stochastic approximation
  - [Robbins–Monro](robbins-monro.md)
  - [Polyak–Tsypkin: the optimal nonlinearity](polyak-tsypkin.md)
  - [Two-timescale tracking](two-timescale-tracking.md)
- Beyond smooth gradients
  - [Subgradient method](subgradient-method.md)
  - [Frank–Wolfe](frank-wolfe.md)
  - [ADMM](admm.md)
  - [EM monotonicity](em-monotonicity.md)
  - [Backpropagation](backprop.md)
- Fixed points
  - [Banach fixed-point theorem](banach.md)
  - [Krasnosel'skii–Mann iteration](krasnoselskii-mann.md)
  - [Brouwer fixed-point theorem](brouwer.md)

## Statistics

- Unbiased estimation
  - [Cramér–Rao](cramer-rao.md)
  - [A-optimal design](a-optimal-design.md)
  - [Rao–Blackwell](rao-blackwell.md)
  - [Lehmann–Scheffé](lehmann-scheffe.md)
  - [Fisher–Neyman factorization](factorization.md)
  - [Basu's theorem](basu.md)
- Maximum likelihood
  - [Fisher information as expected Hessian](fisher-hessian.md)
  - [MLE consistency and asymptotic normality](mle-asymptotics.md)
- Decisions and tests
  - [Neyman–Pearson lemma](neyman-pearson.md)
  - [Bayes decision rules](bayes-decision.md)
  - [James–Stein](james-stein.md)
  - [Bootstrap consistency](bootstrap.md)
  - [Glivenko–Cantelli](glivenko-cantelli.md)
  - [Dvoretzky–Kiefer–Wolfowitz–Massart](dkwm.md)
  - [Le Cam two-point lower bound](le-cam.md)
- Losses
  - [Bregman losses and the conditional mean](bregman-conditional-mean.md)
  - [Proper scoring rules](proper-scoring.md)
- Exchangeability
  - [Conformal coverage](conformal-coverage.md)

## Learning Theory

- Risk decomposition
  - [Bias–variance decomposition](bias-variance.md)
  - [Approximation–estimation decomposition](approx-estimation.md)
- Uniform convergence
  - [Finite-class PAC bound](finite-class-pac.md)
  - [Rademacher bound via symmetrization](rademacher.md)
  - [Massart's lemma](massart.md)
  - [Sauer–Shelah and the VC bound](vc-sauer-shelah.md)
  - [Talagrand contraction lemma](contraction-lemma.md)
  - [Dudley chaining](dudley-chaining.md)
  - [Algorithmic stability](stability.md)
  - [PAC-Bayes](pac-bayes.md)
  - [No free lunch](no-free-lunch.md)
- Online learning
  - [Hedge regret](hedge-regret.md)
  - [Online-to-batch conversion](online-to-batch.md)
  - [Perceptron mistake bound](perceptron.md)
  - [AdaBoost training error](adaboost.md)
- Kernels and networks
  - [Representer theorem](representer.md)
  - [Mercer's theorem](mercer.md)
  - [Kernel ridge regression as GP posterior mean](kernel-ridge-gp.md)
  - [Universal approximation](universal-approximation.md)

## Linear Algebra

- Decompositions
  - [Spectral theorem](spectral-theorem.md)
  - [Singular value decomposition](svd.md)
  - [Eckart–Young](eckart-young.md)
- Variational characterizations
  - [Courant–Fischer](courant-fischer.md)
  - [PCA](pca.md)
  - [Power iteration](power-iteration.md)
- Identities and inequalities
  - [Woodbury identity](woodbury.md)
  - [Matrix determinant lemma](determinant-lemma.md)
  - [Schur complement](schur-complement.md)
  - [Cauchy–Schwarz, Hölder, Young](cauchy-schwarz-holder-young.md)

## Reinforcement Learning and Control

- Dynamic programming
  - [Bellman contraction](bellman-contraction.md)
  - [Policy improvement theorem](policy-improvement.md)
  - [Policy iteration termination](policy-iteration.md)
- Policy optimization
  - [Performance difference lemma](performance-difference.md)
  - [Policy gradient theorem](policy-gradient.md)
  - [Compatible function approximation](compatible-function-approx.md)
- Stochastic approximation
  - [TD(0) convergence with linear features](td-convergence.md)
  - [Q-learning convergence](q-learning.md)
- Entropy regularization
  - [Soft Bellman and the softmax policy](soft-bellman.md)
- Bandits
  - [UCB regret](ucb-regret.md)
  - [EXP3 regret](exp3-regret.md)
  - [Thompson sampling regret](thompson-sampling.md)
- Control
  - [LQR and the Riccati equation](lqr-riccati.md)
  - [Kalman filter](kalman.md)
