# Forecasting with an imperfect simulator

This is a preliminary comparison for the proposed forecasting benchmark. It
establishes ordinary forecasting controls before adding an agent that chooses
experiments. It is not a finished benchmark or a claim of a new forecasting method.

The existing DLinear-on-ETTh1 contribution provides the model and training step.
Those are imported directly. This directory supplies a new task and evaluation;
it does not change the ETTh1 experiments or their saved reference results.

## Task

Each sample is an independent motion trajectory, recording position and velocity.
The reference system follows a damped oscillator with a cubic restoring force:

```text
acceleration = -stiffness * position - damping * velocity - cubic * position^3
```

The cheap simulator omits the cubic term. Both use the same semi-implicit Euler
integrator, observation interval, and substeps. This deliberately isolates model
discrepancy from differences in numerical precision. A matched control removes
the cubic term from the reference too, making both simulators identical.

Models see 24 observations and predict the next 12. The fixed reference budget
contains 32 training trajectories and 16 validation trajectories, each 36 steps
long. Validation counts toward the budget. All methods are given the same data;
ridge and force calibration use the training part only. There are also 256 cheap
training trajectories. These are supplied in a fixed batch; adaptive querying is
not implemented yet.

The evaluator creates two independent test sets after fitting. One uses the
training range of initial displacements, from 0.2 to 1.5 in magnitude. The other
uses larger displacements, from 1.5 to 2.5. Each has 64 trajectories, with its own
seed. Trajectories do not share sliding windows across splits. The evaluator's
test queries are separate from the learner's reference budget.

## Known baselines

- `exp000` trains the existing DLinear model on reference data only. It uses
  Adam at `1e-3` for 300 updates. This is a new recipe on simulated data, not a
  reproduction of the ETTh1 score or its ten-epoch training recipe.
- `exp001` starts from `exp000` and spends its first 150 updates on cheap data.
  Its remaining 150 updates use the same reference sample. Pretraining counts
  toward the total budget. Both DLinear runs use batches of eight trajectories.
- `exp002` fits ridge regression to the difference between the reference targets
  and the cheap simulator's forecasts. Its inputs are flattened observed histories.
  The penalty is fixed at `0.01`; it does not use validation to tune that value.
- `exp003` starts from `exp002` and replaces the output correction with a fitted
  cubic force. It keeps the cheap simulator's other settings and fits its cubic
  coefficient using consecutive observations in the reference training data.
  This method knows the missing force is cubic; it does not discover that form.

Every run also reports the cheap simulator alone and repeating the last observed
value. The cheap simulator starts from the last observed position and velocity,
so it receives a fully observed physical state. This is a useful privileged
control, not an equivalent neural architecture.

DLinear normalization uses reference training data only. Validation chooses its
weights, including a check after the final update. The models receive no test
targets during fitting. MSE and MAE are reported in original units. Scaled MSE
divides errors by the reference training standard deviation of each channel;
lower is better. Familiar and shifted scores are kept separate.

## Run locally

Run from the PRIML checkout with the usual project dependencies installed. The
pilot needs no downloaded dataset, API key, or GPU. Choose an unused output path:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.compare \
  --directory /opt/scratch/artifacts/oscillator/pilot
```

The default run compares the three recipes on matched and mismatched simulators,
using model seeds 2021, 2022, and 2023. Data seeds stay fixed, so this measures
training-seed variation, not uncertainty across independent task populations.
Ridge is deterministic; its repeated rows are not independent evidence.

Pass `--experiments exp_smoke --seeds 2021` for two tiny integration runs.
Existing run directories are never overwritten. The output includes each finalized config and per-run scores,
plus `comparison.json` with environment details, source hashes, and Git status.
These short pilot runs start fresh and do not implement checkpoint resumption.

```sh
uv --quiet run --frozen pytest priml/baselines/oscillator priml/baselines/etth1 -o addopts= -q
```

Training tests carry the repository's `compute_training` marker. The default
fast tier skips them and checks the fitting logic with scripted weight updates.
The command above also runs real optimizer updates. Run the usual checks with:

```sh
uv --quiet run --frozen --group pre-commit pre-commit run \
  --files $(git ls-files priml/baselines/oscillator)
```

The numerical goldens in `testdata/` pin DLinear initialization and two-update
oscillator runs, with and without cheap-data pretraining. They keep initial
weights, diagnostic losses, and final state; generated inputs and intermediate
training tensors are not saved. Replay uses exact equality, with tests that
deliberately change the learning rate or one output bit to check that it fails.
These are regression records from this implementation, not an independent
reference implementation of the oscillator benchmark.

Both recording and replay run through pytest with PRIML's numerical environment
and `host_agnostic_numerics()`. Only regenerate after an intentional recipe
change, inspect the new records, then replay without regeneration:

```sh
uv --quiet run --frozen pytest priml/baselines/oscillator/regression_test.py -o addopts= -q --regenerate-b4b
uv --quiet run --frozen pytest priml/baselines/oscillator/regression_test.py -o addopts= -q
```

## Local pilot

The first CPU comparison on October 15, 2026 used the settings above. Mean
scaled MSE across the three model seeds was:

| Simulator | Method | Familiar scaled MSE | Shifted scaled MSE |
| --- | --- | ---: | ---: |
| Matched | DLinear (`exp000`) | 0.001361 | 0.001357 |
| Matched | Pretrained DLinear (`exp001`) | 0.001309 | 0.001301 |
| Matched | Ridge correction (`exp002`) | 0 | 0 |
| Matched | Cheap simulator | 0 | 0 |
| Mismatched | DLinear (`exp000`) | 0.012306 | 1.586426 |
| Mismatched | Pretrained DLinear (`exp001`) | 0.012438 | 1.596659 |
| Mismatched | Ridge correction (`exp002`) | 0.000086 | 1.060733 |
| Mismatched | Cheap simulator | 0.044674 | 0.395752 |

Zero error is expected for the matched cheap simulator and ridge correction:
the reference uses the same equations and observed state.

The simple correction fits familiar conditions well but extrapolates poorly.
On the shifted test, leaving the cheap simulator alone beats all three fitted
recipes. This is a reason to keep that control in future comparisons, not
evidence that an agent will solve the problem. No settings were tuned after
inspecting these test scores. Rerunning the command above writes the per-run
measurements and provenance to its `--directory`.

## Fitting the missing force

`exp003` asks whether the earlier failure comes from correcting forecast outputs
instead of fitting the dynamics. Its loss uses every adjacent observation pair
within each training trajectory, including the training targets. Pairs never
cross trajectory boundaries. Validation and test data are not used for fitting.

The search is Brent's bounded minimization, mixing golden-section and parabolic
steps, over coefficients from 0 to 1, with `xatol=1e-6` and at most 64 search
evaluations. It never evaluates the interval's endpoints. The unchanged simulator
is also scored and retained if its training loss is better. Each evaluation
advances all training pairs by one observation interval using the same four
substeps as the supplied simulator. Loss calculations use float64; forecasts use
the input dtype. The search is an injectable config, including its bounds and
stopping rules.

Run the existing controls and the new method together:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.compare \
  --experiments exp000 exp001 exp002 exp003 --seeds 2021 2022 2023 \
  --directory /opt/scratch/artifacts/oscillator/calibration
```

The October 17, 2026 CPU run used the original data splits and all four recipes,
for 24 runs across the matched and mismatched cases. On mismatched dynamics:

| Method | Familiar scaled MSE | Shifted scaled MSE |
| --- | ---: | ---: |
| DLinear (`exp000`) | 0.012306 | 1.586426 |
| Pretrained DLinear (`exp001`) | 0.012438 | 1.596659 |
| Ridge correction (`exp002`) | 0.000086 | 1.060733 |
| Calibrated simulator (`exp003`) | 3.04e-15 | 2.01e-14 |
| Cheap simulator | 0.044674 | 0.395752 |

Calibration recovered a coefficient of 0.40000005 from training data, against
the reference's 0.4, using nine simulator evaluations including the unchanged
candidate. In the matched case it retained zero, with zero forecast error.
Calibration and ridge are deterministic here; their repeated model-seed rows
are not independent trials. The reports count simulator evaluations separately
from gradient updates, which are zero for both methods. Compute is not matched
to DLinear's 300 optimizer updates.

This is a parameter-fitting control with privileged knowledge of the equation
family, not a new forecasting method. Both simulators share their integrator,
the observed states have no noise, and only one force coefficient is unknown.
Near-zero error shows that this version of the task is easy under those
assumptions. It does not establish robustness to unknown force laws or real data.
The recipe was fixed before this comparison; no settings were selected using
these test scores. The earlier baselines and their numerical goldens are unchanged.

## Choosing missing force terms

The next comparison adds quadratic drag to the reference system:

```text
acceleration = -position - 0.15 * velocity - 0.4 * position^3
               - 0.35 * velocity * abs(velocity)
```

The cheap simulator still has only its linear restoring force and linear drag.
The reference equation is available to the evaluator, not passed to the method.
This changes the task; scores should only be compared between methods on this
same system. Extra drag also damps the motion faster, so adding another force
does not necessarily make every forecasting method perform worse.

- `exp004` forks `exp003`, changing only the reference system. It tests whether
  fitting a cubic coefficient can compensate for an unmodelled drag force.
- `exp005` forks `exp004`, replacing calibration with force selection. It tries
  all one- and two-term combinations from six supplied force shapes, plus the
  unchanged simulator. Training transitions determine the coefficients.
- `exp006` forks `exp005`, removing quadratic drag from the candidate library.
  This checks what happens when the correct equation cannot be selected.
- `exp007` and `exp008` fork `exp004`, using the original DLinear and ridge
  methods on exactly the same data as the force-fitting methods.

The library contains position powers 1, 3, and 5, and signed velocity powers
1, 2, and 3. A signed power is `-s * abs(s)^(power - 1)`. Coefficients stay
between 0 and 2. Each combination is fitted with bounded nonlinear least squares
on adjacent training observations, using the actual integration substeps rather
than a finite-difference estimate of acceleration. The fit is Levenberg-Marquardt
with forward-difference Jacobians; a coefficient pressing against a bound is held
there while the others move. Separate validation trajectories choose the
combination by full-horizon scaled MSE plus `1e-6` per added term. Test targets
never enter fitting or selection.

The full library needs 21 candidate fits and one unchanged control; the ablation
needs 15 fits and the control. Fits start at 0.1, allow 40 solver evaluations,
and use `ftol=xtol=gtol=1e-9`. Reported training evaluations count every residual
call, including calls used for numerical Jacobians. Validation calls are counted
separately. An unconverged or overflowing candidate cannot replace the control.
The reference-data budget stays at 32 training and 16 validation trajectories;
the methods do not have equal compute budgets.

This follows the established idea of sparse equation discovery, with a small
exhaustive search and nonlinear transition fits. It is not an implementation of
the original SINDy solver or a claim of a new identification algorithm. Supplying
the right force shapes remains a substantial advantage, which `exp006` tests.

Run the comparison with fresh data samples shared by every method:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.compare \
  --experiments exp004 exp005 exp006 exp007 exp008 --seeds 2021 \
  --data-seed-offsets 0 1000 2000 3000 4000 \
  --directory /opt/scratch/artifacts/oscillator/force-discovery
```

Each offset changes every data-split seed while keeping the splits separate.
The default offset is zero, preserving the earlier comparison's data and file
names. Reports include the offset and complete finalized configs. These five
offsets use disjoint seed sets. They vary trajectory samples, not the underlying
equation or coefficient values; DLinear uses one model seed in this comparison.

The October 17, 2026 CPU comparison completed 50 runs: five methods, five data
samples, and the matched/mismatched controls. Mean scaled MSE on the mismatched
system was:

| Method | Familiar scaled MSE | Shifted scaled MSE |
| --- | ---: | ---: |
| Cubic calibration (`exp004`) | 0.000820 | 0.002398 |
| Force selection (`exp005`) | 2.23e-16 | 8.30e-16 |
| Force selection without quadratic drag (`exp006`) | 0.000177 | 0.000217 |
| DLinear (`exp007`) | 0.010923 | 0.144221 |
| Ridge output correction (`exp008`) | 0.000023 | 0.007222 |
| Cheap simulator | 0.006589 | 0.050387 |

`exp005` selected the cubic restoring force and quadratic drag on all five
samples, recovering coefficients near 0.4 and 0.35. It used 241--247 training
residual evaluations, including one starting check per candidate, and 22
validation batches per sample. Those counts come from an October 20, 2026 rerun
with the current least-squares fit, which reproduced the selections and scores of
`exp005` and `exp006`. In the matched control it selected no extra terms, keeping
exact zero test error.

Without the correct drag term, `exp006` selected cubic drag instead. Its shifted
scaled MSE ranged from 0.000182 to 0.000270. That is a useful approximation over
the tested range, not recovery of the reference equation. Better forecasts alone
do not establish that the learned physical explanation is correct.

The settings were fixed before this comparison. The observations are still
noiseless and fully observed, both simulators use the same integrator, and only
one new reference equation is tested. The full library still solves this setup
almost exactly. This adds stronger controls and a measurable library ablation;
it does not establish a difficult general forecasting benchmark. No active query
selection or real-data evaluation is included.

## Delayed interactions

`exp009` starts a separate response task. Earlier observations contain smooth
pulses, followed by 32 observations with zero position and velocity. The future
depends on a product of those older pulses:

```text
delayed_force(t) = gain * tanh(product(position(t - lag) for lag in lags))
acceleration(t) = -stiffness * position(t) - damping * velocity(t) + delayed_force(t)
```

The supplied history is a boundary condition, not a rollout of the autonomous
oscillator before the forecast begins. Its velocity is the derivative of the
prescribed displacement. The delayed force is held fixed within each observation
interval, with four integration substeps. The cheap simulator omits the delayed
force and therefore predicts zero from the final resting state. The full observed
history contains every input needed to forecast the response; no unseen future
forcing or random target noise is added.

The two-pulse task observes 96 timesteps and predicts 48, using lags 80 and 48.
`exp019` extends this to four pulses, 160 observed timesteps, and lags 144, 112,
80, and 48. Both tasks keep a 48-step horizon within the observed delays. The
rollout also supports longer horizons using its own predictions, but those are
not part of this comparison.

For each independent magnitude tuple, the dataset includes every sign combination:
four histories for two pulses, or sixteen for four pulses. Flipping one pulse
reverses the future response while leaving the final resting state unchanged.
Training, validation, and test sets use separate magnitude draws and seeds; a
sign group never crosses splits. The main reference budgets are:

| Task | Training histories | Validation histories | Independent training / validation tuples |
| --- | ---: | ---: | ---: |
| Two pulses | 32 | 16 | 8 / 4 |
| Four pulses | 128 | 64 | 8 / 4 |

Each familiar or shifted test set contains sixteen independent magnitude tuples.
The shifted set increases the first pulse's displacement range from 0.2--1.5 to
1.5--2.5. Other pulse magnitudes retain the training range. Validation still
counts toward the reference budget. The four-pulse task has more histories so
that it does not have fewer independent tuples than the two-pulse task.

This construction makes a specific limitation testable. The balanced response is
orthogonal to every affine function of the history, so DLinear cannot improve on
zero-response MSE, even with perfect optimization. With four pulses it is also
orthogonal to every polynomial history feature of degree below four. Tests check
these identities directly. This is a controlled diagnostic of interaction order,
not a claim that delayed nonlinear forecasting is a new research problem.

### Controls

The added methods use the same reference splits within each task:

- `exp009` uses DLinear. `exp010` uses affine ridge correction, `exp011` fits
  the old cubic force, and `exp012` selects from the old memoryless force library.
- `exp013` fits a quadratic polynomial kernel correction, using training-only
  normalization and a fixed ridge penalty of `1e-3`. The dual solve avoids
  explicitly storing all interaction features.
- `exp014` uses a single 64-hidden-unit SwiGLU block over the flattened history.
  It trains through PRIML's shared `TrainStep`, with Adam at `1e-3`, 300 updates,
  batches of eight, and validation every 25 updates including the final update.
  The complete best training state is restored. Compilation is disabled for this
  CPU comparison. Accumulation, clipping, EMA, nonfinite-step skipping and device
  placement remain available; autocast and time or epoch budgets are rejected.
- `exp015` is a privileged learnability control. It receives the correct delay
  equation and lags, but fits the gain from training responses using a bounded
  search over 0--3. It does not discover that equation.
- `exp016` gives the network 3,000 updates. `exp017` also increases training and
  validation counts eightfold, to a reference budget of 384. `exp018` masks the
  older history from the equation-aware control during fitting and prediction.

On four pulses, `exp019`--`exp024` repeat DLinear, affine ridge, the quadratic
kernel, sparse memoryless forces, the single gated block, and the equation-aware
control. `exp025` gives the gated block 3,000 updates. `exp026` changes the kernel
degree to four. `exp027` gives the 3,000-update network eight times the reference
data, a budget of 1,536. `exp028` enables biases in the single gated block;
`exp029` replaces it with two composed gated blocks, keeping 3,000 updates and the
main 192-history reference budget. The first block's output projection starts
nonzero so that both blocks receive gradients.

The bias and depth controls matter. For one default bias-free SwiGLU block,
`SiLU(z) - SiLU(-z) = z` makes the even part of its output quadratic in the input.
It therefore cannot represent the balanced four-way response. This statement
applies to that standalone block with fixed training normalization. It does not
apply to arbitrary transformers. Biases or a second nonlinear block remove that
restriction, so their measured failures cannot be attributed to the same proof.

### Local delayed-response results

An initial 90-run two-pulse pilot used data offsets 0, 1000, and 2000. Its strong
quadratic-kernel result motivated the four-pulse extension. After fixing the
expanded recipes, the October 17 CPU audit ran `exp009`--`exp029` on fresh data
offsets 3000, 4000, and 5000, with model seeds 2021--2023: 189 runs in total.
Settings were not changed after inspecting these audit scores.

The following values are mean forecast skill, `100 * (1 - method MSE / cheap MSE)`.
100% means an exact forecast, 0% matches the cheap zero-response forecast, and a
negative value is worse. This is error reduction, not percent-correct accuracy.
Matched cases have zero control error and report undefined skill as `null`.
Original-unit MSE and MAE remain in every run record.

| Two-pulse method | Reference budget | Updates | Familiar skill | Shifted skill |
| --- | ---: | ---: | ---: | ---: |
| DLinear (`exp009`) | 48 | 300 | -0.25% | -0.32% |
| Affine ridge (`exp010`) | 48 | 0 | 0.00% | 0.00% |
| Cubic calibration (`exp011`) | 48 | 0 | 0.00% | 0.00% |
| Memoryless force selection (`exp012`) | 48 | 0 | 0.00% | 0.00% |
| Quadratic kernel (`exp013`) | 48 | 0 | 94.35% | 54.26% |
| Single gated block (`exp014`) | 48 | 300 | 95.08% | 55.79% |
| Known delay equation (`exp015`) | 48 | 0 | ~100% | ~100% |
| Longer network training (`exp016`) | 48 | 3,000 | 95.83% | 61.80% |
| More reference data (`exp017`) | 384 | 3,000 | 96.22% | 73.78% |
| Known equation, older history masked (`exp018`) | 48 | 0 | 0.00% | 0.00% |

| Four-pulse method | Reference budget | Updates | Familiar skill | Shifted skill |
| --- | ---: | ---: | ---: | ---: |
| DLinear (`exp019`) | 192 | 300 | -1.10% | -0.72% |
| Affine ridge (`exp020`) | 192 | 0 | 0.00% | 0.00% |
| Quadratic kernel (`exp021`) | 192 | 0 | ~0% | ~0% |
| Memoryless force selection (`exp022`) | 192 | 0 | 0.00% | 0.00% |
| Single gated block (`exp023`) | 192 | 300 | -0.06% | -0.09% |
| Known delay equation (`exp024`) | 192 | 0 | ~100% | ~100% |
| Longer network training (`exp025`) | 192 | 3,000 | -0.01% | -0.01% |
| Fourth-order kernel (`exp026`) | 192 | 0 | 33.81% | 8.83% |
| More reference data (`exp027`) | 1,536 | 3,000 | -0.01% | -0.02% |
| Single gated block with biases (`exp028`) | 192 | 3,000 | 1.40% | -2.92% |
| Two gated blocks (`exp029`) | 192 | 3,000 | 54.61% | -21.58% |

The deeper model learned a substantial familiar response, but its shifted skill
ranged from -211.52% to 66.32% across the nine runs. The fourth-order kernel ranged
from -113.93% to 80.53%. These averages are unstable extrapolation results, not
uniform failures on every seed. The known-equation control recovered gain 2 to
numerical precision. Memoryless methods cannot use the older cues and retain the
zero-response forecast. The masked control confirms that the cues carry the
information needed for the task.

`exp028` and `exp029` are sensitive to floating-point differences between hosts.
On an aarch64 Linux host with torch 2.11.0, the same command gave 0.75% / -0.27%
and 54.50% / -17.17%, with `exp029` shifted skill from -158.88% to 67.15%.
Every other row in these tables reproduced there to the printed precision.

There are three independent data samples, each reused across three model seeds.
Deterministic methods repeat across model seeds; the nine rows per method are
not nine independent task populations. The comparisons also do not match compute:
kernel methods solve a linear system, the privileged control fits one known
coefficient, and neural models take the stated optimizer updates.

Reproduce the audit in a new output directory:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.compare \
  --experiments exp009 exp010 exp011 exp012 exp013 exp014 exp015 exp016 exp017 \
    exp018 exp019 exp020 exp021 exp022 exp023 exp024 exp025 exp026 exp027 exp028 exp029 \
  --cases mismatched --seeds 2021 2022 2023 --data-seed-offsets 3000 4000 5000 \
  --directory /opt/scratch/artifacts/oscillator/delayed-audit
```

These are fixed pulse shapes and a deliberately balanced synthetic task. The
history is prescribed rather than generated by the forecast equation, and the
known-equation control has considerable prior information. No foundation model,
LLM, or experiment-selecting agent was evaluated. A broader benchmark would need
unseen dynamics and stronger general-purpose forecasting controls before making
claims about current models or real applications.

## What this establishes

This checks whether the evaluation pipeline works and gives the proposed agent
concrete baselines to compare against. A useful next question is whether choosing
reference queries beats random sampling and this simple residual correction at
the same query budget. DLinear is one forecasting control, not a complete baseline
for the agent's experiment-selection policy.

The original experiments are deterministic and fully observed, and their equations are
public. It is not a secure hidden benchmark. There is only one family of dynamics,
and no real-world evaluation. Equal DLinear update counts are not equal wall time;
ridge uses a direct solve, so its compute is not matched to neural training.
Poor scores here would not establish that the benchmark is broadly difficult.
The ETTh1 test MSE cannot be compared numerically with these synthetic scores.

## Version 0.1 transfer benchmark

`exp030`--`exp039` define the controls for a separate, versioned evaluation.
`exp040` combines them under the [declared protocol](PROTOCOL.md). It evaluates
adaptation with 96 or 384 accurate trajectories, counting validation data, on
independent oscillator systems. The audit excludes the development equations and
includes a compositional holdout. Familiar and larger-displacement test scores are
reported separately. The earlier pulse-parity tasks do not enter its score.

The nine methods include a fixed force library and sequence models alongside
DLinear. Chronos-Bolt Tiny and Chronos-2 are zero-shot anchors with pinned public
weights. They use no adaptation targets, and their pretraining compute is not
matched to local training. Their dependencies are isolated from the project's
dependency files. Ordinary tests do not download or load pretrained weights.

### Prerequisites

The benchmark runs on CPU; the reported audit used one Torch thread and 16 GiB
of memory. Preparing the pretrained weights requires network access and the
`hub` extra. Install the separate, hash-checked pretrained dependencies:

```sh
uv sync --frozen --group pre-commit --extra hub
uv pip install --python .venv/bin/python --no-deps --require-hashes -r priml/baselines/oscillator/pretrained-requirements.txt
```

### Model files

Prepare the pinned Chronos-Bolt Tiny and Chronos-2 files once:

```sh
uv --quiet run --frozen --no-sync python -m priml.baselines.oscillator.scripts.prepare_assets
```

The preparation step uses the Hub cache and installs verified local copies under
`/opt/scratch/caches/chronos/`. Existing copies are checked against their manifests.
It prepares weights only; simulated trajectories are generated by the experiment.

### Experiment

Launch the declared audit through PRIML's standard experiment entry point:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE
uv --quiet run --frozen --no-sync python -m priml priml.baselines.oscillator.experiments.exp040
```

### Outputs

`exp040` defaults to the audit phase and writes beneath its configured
`working_dir`, `/opt/scratch/artifacts/oscillator/exp040`. Each run needs an unused
directory. The experiment config owns the phase and output location.

`results.json` contains every attempt, the integration diagnostics, and paired
system-level summaries. `protocol.txt` contains the complete declared template
config; each attempt also has its actual finalized config and atomic result file.
A failed method receives no finite aggregate score. The native job saves the
report before raising if a method failed.

Do not tune settings after inspecting audit results and reuse the same audit
claim. This remains a small synthetic benchmark with public equations, not a
secure hidden test or an LLM evaluation. See [the release record](RELEASE.md) for
local results and remaining limitations.

## References

- Zeng et al., [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
- Toner and Darlow, [An Analysis of Linear Time Series Forecasting Models](https://proceedings.mlr.press/v235/toner24a.html)
- Hoerl and Kennard, [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634)
- Brent, [Algorithms for Minimization without Derivatives](https://maths-people.anu.edu.au/~brent/pub/pub011.html)
- Madsen, Nielsen and Tingleff, [Methods for Non-Linear Least Squares Problems](https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf)
- Brunton et al., [Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://doi.org/10.1073/pnas.1517384113)
