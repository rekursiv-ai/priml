# Vital Proofs

Each vignette proves one result in the style of [Cramér–Rao](cramer-rao.md):
the idea, then the core argument as aligned equation chains, then the fine
print (edge cases, cited steps, regularity).

## Probability

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
  - [Stein's lemma](stein-lemma.md)
  - [Gaussian conditioning](gaussian-conditioning.md)
  - [Reparameterization vs score function](reparameterization-vs-score.md)
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
  - [Maximum entropy and the exponential family](max-entropy.md)
- Variational representations
  - [f-divergence variational form](f-divergence-variational.md)
  - [Donsker–Varadhan](donsker-varadhan.md)
  - [Gibbs variational principle](gibbs-variational.md)
  - [Evidence lower bound](elbo.md)
  - [Kantorovich–Rubinstein duality](kantorovich-rubinstein.md)
- Coding and estimation limits
  - [Kraft inequality and source coding](kraft-source-coding.md)
  - [Fano's inequality](fano.md)

## Optimization

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
  - [Gradient descent as proximal gradient](gd-as-proximal.md)
  - [Mirror descent](mirror-descent.md)
  - [Natural gradient](natural-gradient.md)
  - [Newton as proximal step](newton-as-proximal.md)
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
  - [Robbins–Monro](robbins-monro.md)
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
