# Part five: a hidden chain, two drive tones, and a sensor that goes dark

Part four left one method above zero: DLinear explained about a quarter of the
future, and almost all of that came from continuing the newest readings for a
few dozen steps. Part five removes that shortcut and deepens the hidden state,
while keeping the future exactly determined by the history.

## Task

The observed mass is part four's driven, possibly double-well oscillator, now
coupled to a chain of two hidden masses and pushed by two unobserved tones:

```text
a1 = (part four's terms) + kc (x2 - x1) + F cos(phase) + G cos(tone)
a2 = -k2 x2 - c2 v2 - kc (x2 - x1) + kd (x3 - x2)
a3 = -k3 x3 - c3 v3 - kd (x3 - x2)
phase' = omega,  tone' = omega2
```

- Methods see only `x1`, through part four's sensor (noise, 2% glitches, 0.01
  quantization) that also freezes on its last reading in bursts: a free
  reading starts a burst with probability 0.01, and bursts last six readings
  on average.
- 384 observed states and 192 forecast states. The newest 16 to 24 readings are
  an outage, frozen at the last good value, so every forecast begins 16 to 24
  steps after the last live reading. Training labels are live; scores use the
  clean `x1`.
- Zero to four changes, with probabilities 0.1, 0.2, 0.3, 0.25 and 0.15. The
  last leaves 112 to 168 states, including the outage, so 88 to 152 live
  readings follow it. Earlier changes are 24 to 56 states apart. Only the
  observed mass's six coefficients change; the hidden chain and both tones
  persist.
- New training ranges: `kd` in `[0.05, 0.4]`, `k3` in `[0.3, 3]`, `c3` in
  `[0.01, 0.1]`, `G` in `[0.2, 0.6]`, `omega2` in `[1.7, 2.9]`. Part four's
  ranges are unchanged.

`exp053` evaluates the training ranges. `exp054` audits five reserved
families: part four's chaotic well, asymmetric restoring and self-excited
cycle, plus a beating drive (`omega2` in `[0.6, 1.6]`, the first tone's band)
and a stiff far mass (`k3` 4 to 6, `kd` 0.5 to 0.8).

The physics reference, `library_shooting`, now fits all sixteen coefficients
and six hidden states to the newest 64 readings with part four's Cauchy loss.
It is told nothing about the outage. Part four's generator is unchanged: its
random streams are bit-identical.

## Local run: 9 November 2026

Both campaigns completed on a local Apple CPU with one Torch thread, 32
episodes per family. Skill is `1 - MSE / MSE(zero forecast)`.

| Method | Training ranges | Chaotic well | Asymmetric | Self-excited | Beating drive | Stiff far mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear | -0.04 | -0.16 | -0.04 | -0.00 | 0.08 | -0.17 |
| History net | -0.22 | -0.32 | -0.14 | 0.09 | -0.05 | -0.32 |
| Ridge | -1.67 | -3.47 | -1.17 | -2.37 | -1.49 | -1.79 |
| Recent cubic physics | -0.48 | -0.28 | -0.23 | -0.27 | -0.14 | -0.25 |
| Last value | -1.37 | -0.58 | -0.74 | -0.62 | -0.85 | -1.10 |
| Library shooting | -1.85 | -1.04 | -0.67 | -0.32 | -0.70 | -1.65 |

No method beats the zero forecast on the training ranges, and the best audit
score is 0.09. Without an outage, DLinear kept 0.12: skill 0.66 over the first
16 forecast steps, falling below zero after step 144. The outage removes that
near-term persistence.

The future is pinned down by the hidden state. The exact continuation
reproduces every target in every family. With a 48-reading outage, which is
longer than the released 24 and so a conservative bound, the costs of small
errors are low. A 0.01 error in the state at the last live reading costs at
most 0.0008 skill on the training ranges and 0.049 on the chaotic well. A 5%
coefficient error costs 0.06, and at most 0.24 (chaotic well).

Learnability calibration, run outside the declared experiments. The fit
covers the whole live history, knows every change time, and shares the hidden
chain and tones across regimes. It used eight restarts of 1500 Adam steps on
eight development episodes:

- The fits that came closest to the truth's loss forecast with positive skill.
  Episode 1 reached 0.287 against 0.283 and scored 0.45. Episode 5 reached
  0.322 against 0.306 and scored 0.42. Episode 6, the noisiest, fit below the
  truth's loss (0.620 against 0.669) and still scored 0.35.
- Four fits stalled well above the truth's loss and forecast badly (-0.29 to
  -16.5). One diverged.
- Fits started at the truth with 1% jitter reached up to 0.91 (episode 3,
  loss 0.152 against 0.146) and 0.98.

A first version with a 32 to 48 reading outage failed this check. In episode
3, two fits matched the history equally well, yet one forecast at 0.90 and the
other at -1.78. Shortening the outage to 16 to 24 readings cleared those first
eight episodes. Eight more episodes (seed 778) did not hold up:

- Two random-start fits came within 0.01 of the truth's loss yet forecast at
  -4.17 (loss 0.439 against 0.436) and -3.71 (0.156 against 0.148).
- The other near-truth-loss fits forecast at 0.56 to 0.98.

**Status: harder than part four and no less well-posed.** In most calibrated
episodes, a fit that matches the history forecasts well. In about one episode
in eight, the history admits a near-equal fit with a very different future,
so no method could be scored fairly there. Part four has the same flaw, about
three times as often. Removing the far hidden mass did not help:
with it detached, one fit matched the history exactly as well as the truth
(loss 0.2832 against 0.2832) yet forecast at -0.19, and another forecast at
-3.09. Part four's 128-step horizon did not help
either. Episode 0 of seed 778 failed identically at both horizons (loss 0.439
against 0.436, skill -4.5). Two further fits beat the truth's loss and still
forecast at -1.9 and -1.2. So neither the far mass nor the horizon is the
cause. Some noisy episodes admit fits that explain the readings as well as the
truth but continue differently. The same calibration on part four, with the same sixteen seeds and settings
(`--part four`), shows the limitation is inherited, not new. There, six
episodes have a fit at or below the truth's loss that forecasts at -0.2 to
-3.4, against two of sixteen in part five. Part four's earlier claim that the
truth is always the global minimum came from eight episodes with one known
change, and does not survive this larger check.

## Limits

The calibration knew every change time and used only eight episodes, so it
cannot prove identifiability. It found counterexamples in two of sixteen
episodes, against six of sixteen for part four. The outage is detectable (exactly repeated
readings) but its length is not given. No AI agent or pretrained forecaster
has been evaluated yet.

## Reproduce

```bash
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp053 --override working_dir=/absolute/path/development
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp054 --override working_dir=/absolute/path/audit
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --seed 777
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --seed 778
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --restarts 1 --iterations 300 --near-truth
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part four --seed 777
uv --quiet run --frozen python -m priml.baselines.oscillator.scripts.joint_calibration --part four --seed 778
```
