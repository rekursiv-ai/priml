# Part six: a longer outage, judged inside the declared model

Part five left every method at or below zero on its training ranges, but two
audit families still scored above zero, and about one calibrated episode in
eight admitted a fit that matched the history as well as the truth yet
forecast badly. Part six finds the cause of that ambiguity, removes it from
the solvability check, and spends the recovered margin on a longer outage.

## Task

Part five's generator, equations, sensor, changes and ranges are unchanged
except for the outage and the final regime:

- The newest 24 to 32 readings are an outage (part five: 16 to 24), frozen at
  the last good value. Every forecast begins 24 to 32 steps after the last
  live reading.
- The last change leaves 112 to 176 states, including the outage (part five:
  112 to 168), so 80 to 152 live readings follow it.
- Development uses a fresh stream (seed 3306) and the audit families move to
  seeds `20_261_401` to `20_261_405`, so part five's published futures reveal
  nothing about part six's.

`exp055` evaluates the training ranges. `exp056` audits part five's five
reserved families on their new streams.

## Why part five was ambiguous

Small errors were never the problem. Perturbing the true state at the last
live reading by 0.01, or every coefficient by 1%, cost at most 0.11 skill in
all sixteen calibrated part five episodes.

The bad fits were distant ones. Every random-start fit that matched the
history but forecast badly used negative linear damping together with
self-excitation or a quadratic restoring term. Neither appears in the
training ranges; the shooting bounds admit them only because they are shared
with the audit families. Searching only the declared training ranges removed
every counterexample in the same sixteen episodes:

| Part five, seeds 777 and 778 | Wide bounds | Declared ranges |
| --- | --- | --- |
| Fits below or within 0.01 of the truth's loss | 6 | 6 |
| Of those, forecasting below zero | 2 (-4.17 and -3.71) | 0 (0.06 to 0.91) |

An eight-episode third seed (779) also passed. So part five is well-posed
inside its declared model, and `joint_calibration --declared` is the
solvability check from here on.

## Local run: 19 November 2026

Both campaigns completed on a local Apple CPU with one Torch thread, 32
episodes per family. Skill is `1 - MSE / MSE(zero forecast)`. Part five was
rerun on the same code and reproduces its published table.

| Method | Training ranges (part five) | Training ranges | Chaotic well | Asymmetric | Self-excited | Beating drive | Stiff far mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear | -0.04 | -0.06 | -0.13 | -0.09 | -0.03 | -0.02 | -0.08 |
| History net | -0.22 | -0.08 | -0.17 | -0.15 | -0.25 | -0.23 | -0.08 |
| Ridge | -1.67 | -2.49 | -3.26 | -2.21 | -2.63 | -1.77 | -2.06 |
| Recent cubic physics | -0.48 | -0.41 | -0.64 | -0.59 | -0.16 | -0.44 | -0.49 |
| Last value | -1.37 | -0.83 | -1.33 | -1.27 | -0.40 | -0.98 | -1.08 |
| Library shooting | -1.85 | -0.82 | -1.36 | -1.02 | -4.16 | -1.52 | -0.46 |

No method beats the zero forecast in any family. Part five's best audit
scores (+0.09 self-excited, +0.08 beating drive) are gone; the best is now
-0.02. Some methods score better on the training ranges than in part five
because the development stream is new; the best method still falls.

## Learnability calibration

The same privileged fit as part five (every change time known, the hidden
chain and tones shared across regimes, eight restarts of 1500 Adam steps),
restricted to the declared ranges, on 24 development episodes (seeds 777,
778, 779):

- All eight fits within 0.01 of the truth's loss forecast at -0.005 or
  better, and seven at 0.13 or better (up to 0.99).
- Two fits just outside that margin forecast badly: 0.010 above the truth's
  loss at -0.57, and 0.013 above at -0.74.
- The remaining fits stalled 0.02 to 2.1 above the truth's loss. Their
  forecasts range from 0.74 to -5.9; the loss itself flags them as failed
  searches.

A 32 to 48 reading outage failed this check: in its noisiest episode a fit
tied the truth's loss exactly (0.6262) and forecast at -0.63. With wide
bounds the same seed had two more counterexamples (-4.08 and -3.31).

A second rejected design switched both drive tones off when the forecast
starts. It kept the future identified, but every learner scored higher,
because the free ring-down is easier to continue than the driven motion.

**Status: harder than part five on every family, and well-posed inside the
declared model in every calibrated episode by part five's 0.01-loss margin.**
Two near-margin fits are the remaining doubt.

## Limits

The calibration knew every change time and covered 24 episodes, so it cannot
prove identifiability. Its random restarts stall often: averaged over all
episodes, even this privileged fit forecasts below zero. Solvable here means
the history pins down the future, not that a known method extracts it
reliably. The 0.01 margin follows part five and is not a statistical test.
No AI agent or pretrained forecaster has been evaluated yet.

## Reproduce

```bash
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp055 --override working_dir=/absolute/path/development
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp056 --override working_dir=/absolute/path/audit
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part six --declared --seed 777
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part six --declared --seed 778
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part six --declared --seed 779
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part five --declared --seed 777
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part five --declared --seed 778
```
