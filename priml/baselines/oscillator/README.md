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
ridge uses the training part only. There are also 256 cheap training trajectories.
These are supplied in a fixed batch; adaptive querying is not implemented yet.

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
  --directory /opt/scratch/runs/oscillator/pilot
```

The default run compares the three recipes on matched and mismatched simulators,
using model seeds 2021, 2022, and 2023. Data seeds stay fixed, so this measures
training-seed variation, not uncertainty across independent task populations.
Ridge is deterministic; its repeated rows are not independent evidence.

Add `--smoke` for two tiny integration runs. On a laptop, use a writable
`--directory` such as a folder under Documents. Existing run directories are
never overwritten. The output includes each finalized config and per-run scores,
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
  --files $(rg --files priml/baselines/oscillator)
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
BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/baselines/oscillator/regression_test.py -o addopts= -q
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
inspecting these test scores. Per-run measurements and provenance are in
`results/local_cpu.json`.

## What this establishes

This checks whether the evaluation pipeline works and gives the proposed agent
concrete baselines to compare against. A useful next question is whether choosing
reference queries beats random sampling and this simple residual correction at
the same query budget. DLinear is one forecasting control, not a complete baseline
for the agent's experiment-selection policy.

The current system is deterministic and fully observed, and its equations are
public. It is not a secure hidden benchmark. There is only one family of dynamics,
and no real-world evaluation. Equal DLinear update counts are not equal wall time;
ridge uses a direct solve, so its compute is not matched to neural training.
Poor scores here would not establish that the benchmark is broadly difficult.
The ETTh1 test MSE cannot be compared numerically with these synthetic scores.

## References

- Zeng et al., [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
- Hoerl and Kennard, [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634)
