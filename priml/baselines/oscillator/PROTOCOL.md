# Oscillator transfer benchmark, version 0.1

## Question

How much does a fixed amount of accurate reference data help a forecaster correct
a simulator with missing physics? Does the correction transfer to new initial
conditions, including a larger displacement range?

This is a controlled synthetic benchmark. It does not establish real-world
forecasting performance, a new identification algorithm, or universal difficulty.
The earlier delayed-pulse experiments remain diagnostics and do not enter this
benchmark's aggregate score.

## Task and information

Each task is one autonomous damped oscillator with an additional restoring force
or drag. Position and velocity are fully observed. The cheap simulator supplies
the same linear stiffness and damping, but omits the extra forces. All methods
receive that simulator and independent reference training and validation windows.
The evaluator keeps the reference equation and test targets outside their API.

The context has 64 observations and the horizon has 24. Observations are 0.1 time
units apart, with eight semi-implicit Euler steps per interval. Histories and
targets come from one continuous rollout. No future forcing or observation noise
is hidden from a method. The reference-data budgets are 64 training plus 32
validation trajectories, and 256 training plus 128 validation trajectories.
Larger budgets retain the smaller budget's prefixes within each split. A further
256 cheap trajectories are supplied to every method, whether it uses them or not.

Each familiar and shifted test set has 128 independent trajectories. Training and
familiar initial displacement magnitudes are 0.2--1.5; shifted magnitudes are
1.5--2.5. Initial velocities range from -0.5 to 0.5. A shifted initial condition
does not guarantee that every later observation lies outside the training range.
Splits and systems have distinct seeds; no sliding windows cross splits.

## Development and audit

Development uses matched linear motion, a missing cubic restoring force, and
missing quadratic drag. These cases may guide implementation and recipe choices.
The audit uses four previously excluded families: a combination of the two
development forces, saturating restoration, sinusoidal restoration, and drag
whose strength depends on position. There are four independently sampled systems
per audit family. Coefficients are drawn once per system and shared by every
method and data budget. They are fixed during a trajectory.

Each method fits anew on a task's reference data. This tests transfer of a fixed
learning recipe to an unseen mechanism; it does not test a single set of learned
weights across all mechanisms. The combined-force family is a compositional
holdout, while the other audit forces are absent from the development library.

Choose the complete experiment configuration before evaluating the audit. A test
result may not be used to change a recipe and retain the same audit claim. A
later revision must disclose prior audit exposure and use a fresh audit cohort.
The native experiment saves its declared template configuration and each
attempt's actual finalized configuration with the results; this
is a record of the run, not an independently timestamped preregistration.

## Comparisons

The adapted controls include DLinear, affine simulator correction, a nonlinear
kernel correction, and a fixed force-library search. Neural controls include a
two-block history network, patch attention, and a recurrent sequence model.
Their optimizer budgets and validation cadence are explicit in each recipe.
Each neural fit allows 2,400 updates. A development-only check against 600 updates
found lower validation error for all four neural recipes on all three development
systems at the larger data budget. This is a finite compute allowance, not a claim
that every optimizer has converged.
Pretrained Chronos models are zero-shot anchors: they ignore the reference
training and validation targets and are reported separately from adapted models.
Their pretraining compute is neither free nor matched to local training.

Neural model seeds are repeated within each system. Deterministic methods run
once. Data budgets are matched within an adapted comparison; wall time and
optimizer updates are reported rather than presented as equal compute.
The force-library control knows a small set of candidate force shapes. No
competitor receives the audit reference equation. An equation-aware rollout is
used only to check data generation and integration error, not ranked as a learner.

## Scores and uncertainty

The primary score is mean squared forecast error divided by each channel's
reference-training variance, averaged over channels, timesteps, and trajectories.
For reporting, that variance always uses the common 64-trajectory training prefix,
so increasing the adaptation budget does not change the scoring denominator.
Familiar and shifted results are separate. Original-unit MSE and MAE are retained.
Cheap-simulator and last-value errors accompany every result. Skill means
`100 * (1 - model MSE / cheap MSE)`, not percent-correct accuracy; it is undefined
when the cheap control's error is zero.

First average model seeds within a system. Then average systems within each
family and give the four families equal weight. Confidence intervals resample
independent systems within each fixed family, keeping paired method comparisons
together. Model seeds and individual timesteps are not independent task samples.
Intervals describe these sampled systems, not all possible dynamical laws.

Each summary row names its comparator in `baseline` (DLinear by default).
`paired_difference` is the method's scaled MSE minus that comparator's scaled
MSE; `paired_difference_interval` reports the paired bootstrap bounds.

All attempted runs must remain in the report. Missing runs invalidate a complete
audit. A nonfinite forecast is a recorded failure, not a row to discard; a method
with a failed system receives no finite aggregate score. Individual families and
the observed worst system are reported alongside means.

## Release checks

The evaluator checks split isolation, budget accounting, deterministic task
generation, and consistency between a reference rollout and its targets. It
compares the reference forecast with a four-times-finer integration grid, recording
the discrepancy rather than claiming continuous-time ground truth. Tests cover
test-target isolation, incomplete results, and resuming with changed configuration
or default tensor dtype.

Public pretrained weights are pinned to exact revisions and prepared before
evaluation; their local files are checked against manifests. The simulator and
offline tests require no model download. Runs retain every attempt and record
the full configuration and default tensor dtype. Generated data and full run
logs stay outside Git. Reported results must state the hardware, software
versions, and all limitations.

## Scope of a first release

This release studies noiseless, two-variable motion with known linear physics.
It is not a secure hidden test set, an active-query policy, or an LLM benchmark.
Related work already studies physics-assisted forecasting and out-of-domain
dynamical systems. The contribution here is a reproducible PRIML evaluation with
explicit adaptation budgets and mechanism holdouts, not a claim that this research
area is new. Independent review remains necessary before broader conclusions.

Related work: [APHYNITY](https://proceedings.mlr.press/v144/wang21a.html),
[out-of-domain dynamics](https://proceedings.mlr.press/v235/goring24a.html),
[GIFT-Eval](https://arxiv.org/abs/2410.10393), and
[Chronos](https://github.com/amazon-science/chronos-forecasting).
