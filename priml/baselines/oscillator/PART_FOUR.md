# Part four: a hidden drive, repeated changes, and a glitchy sensor

Part three's DLinear explained almost all of the future: its development score,
`0.0088`, beat the physics reference. Part four removes what made that easy,
while keeping the future exactly determined by the hidden state.

## Task

The observed mass may sit in a double well, is pushed by a periodic force
that is never observed, and is coupled to a hidden partner:

```text
a1 = -k1 x1 - (c1 + e x1^2) v1 - q x1^2 - a x1^3 - mu tanh(v1 / 0.05)
     + kc (x2 - x1) + F cos(phase)
a2 = -k2 x2 - c2 v2 - kc (x2 - x1)
phase' = omega
```

- Methods see only `x1` through a sensor with per-episode Gaussian noise (std
  0.02 to 0.06), glitches on 2% of readings (magnitude 0.5 to 1.5, random sign)
  and 0.01 quantization. Training labels pass through the same sensor; scores
  use the clean `x1`.
- 256 observed states, 128 forecast states, interval 0.05, eight substeps.
- Zero to three changes inside the history, with probabilities 0.15, 0.35, 0.3
  and 0.2. The last leaves 64 to 120 states before the forecast; earlier
  changes are 24 to 64 states apart. Each change redraws the observed mass's six
  coefficients. The partner, coupling and drive persist. Methods are given no
  change times or count.
- Training ranges: `k1` in `[-1.5, 1.0]`, `c1` in `[0.02, 0.12]`, `a` in
  `[0.3, 1.2]`, `mu` in `[0, 0.05]`, `kc` in `[0.05, 0.4]`, `k2` in `[0.5, 2]`,
  `c2` in `[0.01, 0.1]`, `F` in `[0.3, 0.8]`, `omega` in `[0.6, 1.6]`. The
  asymmetric `q` and self-excitation `e` are zero in training.

`exp051` evaluates the training ranges. `exp052` audits four reserved families
generated after fitting: a chaotic forced double well, asymmetric restoring
(`q` 0.3 to 0.8), a self-excited limit cycle (`c1` -0.2 to -0.05, `e` 0.3 to
0.8), and a faster drive (`omega` 2.5 to 4).

The physics reference, `library_shooting`, fits the full library, the
partner's state and the drive phase to the newest 64 readings with a Cauchy
loss that discounts glitches. It is given no change time, coefficients, hidden
state or noise level. Scoring is unchanged from part three.

## Local run: 5 November 2026

Both declared campaigns completed on a local Apple CPU with one Torch thread,
32 episodes per family. Skill is `1 - MSE / MSE(zero forecast)`: one is
perfect, zero matches forecasting the resting position, and negative is worse.

| Method | Training ranges | Chaotic well | Asymmetric | Self-excited | Fast drive |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLinear | 0.23 | 0.16 | 0.13 | 0.11 | -0.08 |
| History net | 0.02 | 0.04 | -1.74 | -0.18 | -0.73 |
| Ridge | -6.11 | -8.53 | -7.28 | -9.13 | -11.43 |
| Recent cubic physics | -2.61 | -1.36 | -5.81 | -2.02 | -6.98 |
| Last value | -1.13 | -0.82 | -0.73 | -0.34 | 0.24 |
| Library shooting | -1.25 | -0.74 | -15.82 | -9.66 | -2.53 |

DLinear's development scaled MSE rose from `0.0088` in part three to `1.124`.
No method explains more than a quarter of the future anywhere. On the fast
drive, only persistence beats the zero forecast.

The exact continuation from the hidden state reproduces every target, and the
future is insensitive to small identification errors: perturbing the final
state by 0.01 costs `0.0004` scaled MSE, and 5% coefficient errors cost
`0.016`. The data therefore pin down the future; the failure is identification.

Learnability calibration, run outside the declared experiments:

- Fitting only the readings after the last change does not work, even from
  the true coefficients. With 24 readings it had no skill; with 48 its median
  development episode scored `0.054`. Beyond that, random restarts on the
  newest 64 readings overfit: going from 16 to 64 restarts raised the median
  from `0.91` to `1.09`, because spurious solutions fit the window as well as
  the truth. Hence the final regime is at least 64 readings long, and no
  single-window fit suffices.
- Fitting the whole history jointly, sharing the partner and drive across one
  known change, does identify the system. On eight development episodes with
  eight restarts, the robust loss at the true parameters was never higher than
  the best fit found. The three episodes whose fit reached the truth's loss
  forecast at `0.10` to `0.12`, against `0.70` for the zero forecast; the five
  that stalled above it forecast at `0.36` to `2.98`.

The data therefore support forecasts with most of the variance explained, but
only for a method that locates the changes, pools the persistent partner and
drive across regimes, and solves a rugged global search. Every tested baseline
falls short of that.

## Limits

The future is deterministic given the hidden state, and the evaluator checks
that the exact continuation reproduces every target. The joint-fit
calibration knew the change time, allowed exactly one change and used eight
episodes, so it shows identifiability, not a method that solves the benchmark.
No AI agent or pretrained forecaster (Chronos) has been evaluated yet.

## Reproduce

```bash
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp051 --override working_dir=/absolute/path/development
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp052 --override working_dir=/absolute/path/audit
```
