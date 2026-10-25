# Part two: forecasting after an observed change

Part one fitted a separate model to each fixed oscillator. Part two fits one
recipe across independently sampled dynamics, then asks it to forecast a new
episode. The main question is whether the method can use recent evidence when
older observations describe a different system.

This is a controlled synthetic study. It evaluates forecasting methods, not an
autonomous scientist, and does not introduce a new physical equation.

## What changed

Every episode follows the same damped cubic oscillator family:

```text
acceleration = -stiffness * position - damping * velocity - cubic * position^3
```

Each episode receives its own coefficients and initial state. Methods observe
96 states and forecast the following 64 at an observation interval of 0.05.
The reference advances each interval with eight semi-implicit Euler substeps
in float64. These discrete transitions define the targets; they are not exact
solutions of the continuous differential equation.

The four conditions separate access to state from changes in dynamics:

- `exp041`: position and velocity, stationary dynamics.
- `exp042`: position alone, stationary dynamics.
- `exp043`: position and velocity, one change inside the observed history.
- `exp044`: position alone, one change inside the observed history.

In the changed conditions, new coefficients take effect with 24 to 48 observed
states remaining before the forecast. Position and velocity remain continuous;
the acceleration law changes. No further change occurs in the forecast window.
The model is therefore given evidence of the new regime before it must predict.
Neither the change time nor the coefficients are model inputs.

Paired conditions share their initial-state draws and future coefficients. The
stationary control uses those future coefficients throughout its history.
Changing the observation transform preserves the entire underlying trajectory.
Different schedules produce different histories and forecast-start states, so
the paired schedule comparison is not a comparison from identical final states.

Position-only methods receive no velocity labels during training. Their tensors
also have separate storage from hidden states and future targets. Full-state
methods predict both observed coordinates, but the primary metric always scores
position. Removing velocity thus changes the available supervision as well as
the inputs. DLinear maps channels independently; the nonlinear history model
can mix them. The observation comparison should be interpreted accordingly.

## Methods and information

The numerical forecasting methods are DLinear, pooled affine ridge, and a
nonlinear history model using PRIML's replaceable SwiGLU block. Ridge fits a
history-to-future map directly; it receives no simulator forecast or force law.
The neural methods use the shared `TrainStep`, including its accumulation,
clipping, compilation, EMA, and nonfinite-gradient controls. Training uses
float32 after normalization fitted on training windows alone. Unsupported
autocast and time-based or epoch-based budgets fail explicitly.

Two physics controls know the cubic equation form but not its coefficients.
One estimates the coefficients from the newest 24 observed states; the other
uses the entire history. Both use finite differences computed from visible
values. In the position-only condition they infer the latest velocity too.
The recent-window control benefits from the protocol's guarantee that its
entire window follows the change. It does not detect the change point.

These controls test whether the available observations support prediction under
a strong declared physical prior. They are not evidence that an unknown force
law was discovered. A fixed approximate simulator and last-value persistence
provide additional checks. A separate evaluator-only oracle uses the actual
state and future coefficients to verify exact target continuation; it is not
ranked against learners.

## Data, fitting, and scoring

Training coefficients are sampled independently from stiffness `[0.5, 2.0]`,
damping `[0.05, 0.3]`, and cubic restoration `[0.0, 0.6]`. Initial displacement
magnitude lies in `[0.3, 1.5]`, with random sign; initial velocity lies in
`[-0.5, 0.5]`. The small budget contains 256 training and 64 validation episodes.
Both count toward the reference-data allocation. Neural fitting uses 1,200 Adam
updates, batches of 32, and checkpoint comparisons every 50 updates, including
the final update. Validation selects the checkpoint, not the evaluation pool.

`exp045` runs all four conditions on the separate development stream. The audit
in `exp046` uses four reserved domains, each with 32 independent episodes:

- Stiffening: stiffness `[2.2, 3.2]`.
- Softening: stiffness `[0.2, 0.45]`.
- Stronger drag: damping `[0.35, 0.6]`.
- Stronger cubic restoration: cubic coefficient `[0.8, 1.2]`.

Other coefficient ranges stay unchanged. Before a change, coefficients still
come from the training domain. Audit data is generated after each method has
finished fitting. These are fresh parameter domains within the same equation
family, not held-out physical mechanisms. Published streams are reproducible,
not a permanently secret test set.

`exp047` repeats the audit with four times the training and validation data.
The smaller pools remain exact prefixes. `exp048` instead retains the small
data budget and gives neural methods four times the optimizer updates. Neither
control changes the audit streams or selects settings from their scores.

Every condition includes seven methods and eleven fits: three initialization
seeds for each neural method and one for each deterministic method. A campaign
therefore contains 44 fits. All declared attempts are retained, including
failures. A failed method receives no finite aggregate score.

The primary score is position MSE divided by the variance of initial positions
in a fixed 256-episode training prefix. This denominator is shared across
observation conditions and data budgets. Lower is better. It is not a percentage
correct, and scores from part one's different task and normalization cannot be
compared numerically with it.

Scores retain individual episodes and forecast horizons. Model seeds are
averaged before bootstrapping episodes within each family; families receive
equal aggregate weight. Reports include paired method-minus-reference
differences with 95% percentile intervals. These intervals are conditional on
the chosen parameter families and training pools. They do not measure
uncertainty over all physical systems or independent retrainings of the data.

## Reproduce

Run from the repository root, overriding only the output path for a local
machine. Each command expects a new destination.

```bash
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp045 --override working_dir=/absolute/path/development
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp046 --override working_dir=/absolute/path/audit
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp047 --override working_dir=/absolute/path/large-data
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp048 --override working_dir=/absolute/path/long-training
```

The defaults keep generated records under `/opt/scratch`. Each condition writes
the finalized protocol, source fingerprints, environment metadata, individual
attempts, and an aggregate report. The Python `run(directory, resume=True)` API
can reuse an interrupted campaign only when its source, configuration, and
environment manifest match. It checks retained attempt identities and score
shapes before aggregation. Generated records are not committed to the package.

## Local run: 25 October 2026

The four declared campaigns completed all 176 fits on a local Apple CPU using
one Torch thread, Python 3.14.9, and Torch 2.11.0. Neural fits completed all
201,600 planned optimizer updates without skipped steps. These are local
measurements, separate from the reference-hardware results marked TBD in the
experiment docstrings.

On the development stream, DLinear's full-state error rose from `0.007118` with
stationary dynamics to `0.017697` after an observed change, about 2.49 times the
error. Position-only scores were `0.009295` and `0.018027`, respectively. Thus the
change made this in-domain comparison harder, while hiding velocity alone had
a smaller effect.

Generalization to the reserved coefficient domains was substantially harder.
For position-only stationary episodes, DLinear's audit score was `0.216153`,
compared with `0.009295` on the development distribution. The position-only
changed condition scored `0.088068`, with a 95% within-family episode-bootstrap
interval of `[0.064083, 0.117614]`. The corresponding pooled ridge and nonlinear
history scores were `0.089888` and `0.087842`. The paired nonlinear-minus-DLinear
interval crossed zero, so this run does not establish that the nonlinear model
was better in that condition.

The physics controls isolate an interpretable failure. In the position-only
changed audit, fitting the entire history scored `0.060295`; fitting only the
newest 24 observations scored `0.000002124`. Their paired recent-minus-whole
difference was `-0.060293`, with interval `[-0.107742, -0.027866]`. The recent
control receives no true coefficient or velocity, but it does know the correct
force basis and a window guaranteed to follow the change. This result supports
learnability under that prior and shows how mixing old and new regimes can
damage identification.

The larger-data control lowered DLinear's position-only changed audit score
to `0.083163`, a 5.57% point-estimate reduction. Four times the optimizer updates
gave `0.089684`, a 1.84% increase. For stationary position-only episodes, both
controls helped more: larger data scored `0.165504`, and longer fitting scored
`0.129909`. More optimization is therefore useful in some conditions, but it
does not remove the observed generalization gap. These comparisons do not
prove an inherent limitation of the architecture.

The audit also contradicts a simple difficulty ordering: DLinear's changed
episodes were easier than the stationary ones in these coefficient domains.
The changed histories and forecast-start states differ, and their earlier
coefficients come from the familiar training range. Partial observation and
regime changes should therefore be reported as measured conditions, not assumed
to increase difficulty universally. The present task remains easy for the
informed recent-window physics control.

## Limits and next steps

Observations are noiseless and regularly sampled. The recent physics control
knows the correct equation basis and a safe recent-window length. There is no
unseen forcing, ambiguous future intervention, or real-data evaluation here.
Success on this suite does not establish broad forecasting ability, and failure
does not establish that a model family cannot solve the problem with a different
recipe. The learning budgets and architectures are part of the result.

The next useful step is to test whether the observed failure persists under
sensor noise while retaining an appropriate learnability control. A later
agent-facing task could ask a system to diagnose a failed forecast, propose a
change, and validate it on new systems. That research loop has not been measured
by the current forecasting comparison.
