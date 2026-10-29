# Part three: a hidden partner, sensor noise, and unannounced changes

Part two's best method fitted a cubic law to the newest 24 noiseless positions.
Part three removes every assumption that made that easy, while keeping the
future predictable from the observations.

## Task

The observed mass is coupled by a spring to a second mass that is never
observed:

```text
a1 = -k1 x1 - c1 v1 - a x1^3 - mu tanh(v1 / 0.05) + kc (x2 - x1)
a2 = -k2 x2 - c2 v2 - kc (x2 - x1)
```

- Methods see only `x1`, with per-episode Gaussian sensor noise (std 0.01 to
  0.04). Training labels are noisy too; scores use the clean `x1`.
- 160 observed states, 64 forecast states, interval 0.05, eight substeps.
- A quarter of the episodes have no change. The rest change the observed mass's
  coefficients with 48 to 112 states remaining. The partner and coupling persist.
  Methods are given neither the change time nor whether there was one.
- Training ranges: `k1, k2` in `[0.5, 2.0]`, `c1` in `[0.02, 0.15]`, `a` in
  `[0, 0.6]`, `mu` in `[0, 0.02]`, `kc` in `[0.05, 0.3]`, `c2` in `[0.01, 0.1]`.

`exp049` evaluates the training ranges. `exp050` audits four reserved families
generated after fitting: strong coupling (`kc` 0.6 to 1.0), dry friction (`mu`
0.06 to 0.12), a resonant lightly damped partner, and hardening (`k1` 2 to 3,
`a` 0.8 to 1.5).

The learnability reference, `library_shooting`, fits the full equation form and
the hidden partner's state to the newest 96 noisy observations with 16 restarts.
It is given no change time, coefficients, hidden state, or noise level.

Scoring, attempt records, and bootstrap intervals are unchanged from part two.
The only shared change scores against the clean position held by the evaluator.
That target is identical to part two's labels, so part two's results do not move.

## Local calibration: 27 October 2026

These figures come from one model seed, 32 episodes per family, on a local Apple
CPU. They are not results from the declared experiments.

DLinear's variance explained on the clean future position:

| Family | Variance explained |
| --- | --- |
| Training ranges | 96.4% |
| Strong coupling | 33.1% |
| Dry friction | 95.6% |
| Resonant partner | 95.1% |
| Hardening | -137.9% |

On the training ranges (scaled MSE, lower is better), ridge scored `0.019`. Part
two's cubic fit to the newest 24 states scored `1.45`, because noise breaks its
finite differences. The shooting reference scored `0.019` on all episodes. On the
episodes whose newest 96 states all follow any change, it scored `0.008`. On
stationary episodes fitted over the whole history, it scored `0.0011`.

The data therefore support forecasts far better than the baselines, but only
for a method that separates the old regime from the new one. Locating the change
under noise, with a hidden state, is the open problem the benchmark poses.

## Limits

Dry friction and the resonant partner barely affected DLinear in this
calibration. They may need stronger settings to add difficulty. No AI agent has
been evaluated on this task, so whether current frontier systems solve it is
unmeasured.

## Reproduce

```bash
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp049 --override working_dir=/absolute/path/development
uv --quiet run --frozen python -m priml priml.baselines.oscillator.experiments.exp050 --override working_dir=/absolute/path/audit
```
