# Craftax

Open-ended reinforcement learning in a 2D survival game. A player mines wood
and stone, crafts tools, fights creatures, eats, drinks, sleeps, and descends
through nine floors; 67 achievements form a tech tree, and the score rewards
breadth rather than depth. Nothing here is a wrapper -- the game itself is
reimplemented in PyTorch, so a rollout is one batched tensor program with no
simulator process and no host round trip.

It is the counterpart to `cifar10` for RL: where that one reads a supervised
pipeline end to end, this one shows what an on-policy learner needs that a
supervised loop does not -- an environment that generates its own data, a
score computed over whole episodes, and three ways of remembering the past.

## Attribution

This is a port. The game, its rules, its constants, and the baseline recipes
are the work of the Craftax authors; this repository contributes the PyTorch
implementation and the experiment chain built on it.

- **Craftax** -- Matthews et al. 2024, *Craftax: A Lightning-Fast Benchmark
  for Open-Ended Reinforcement Learning*. [arXiv:2402.16801][craftax-paper] ·
  [MichaelTMatthews/Craftax][craftax-repo] (MIT)
- **Craftax_Baselines** -- the PPO and PPO-RNN recipes `exp000`--`exp002`
  reproduce. [MichaelTMatthews/Craftax_Baselines][craftax-baselines] (MIT)
- **purejaxql** -- the PQN algorithm `exp003` reproduces, from Gallici et al.
  2024, *Simplifying Deep Temporal Difference Learning*.
  [arXiv:2407.04811][pqn-paper] · [mttga/purejaxql][pqn-repo]
- **transformerXL_PPO_JAX** -- the GTrXL architecture `exp013` reproduces.
  [Reytuag/transformerXL_PPO_JAX][gtrxl-repo]

Craftax is MIT-licensed, which permits this port and its redistribution; the
notice above is that license's attribution requirement. The game's sprites are
**not** vendored here -- `game/render/assets.py` downloads them from the
upstream repository at a pinned tag, so what ships is code.

Craftax itself builds on [Crafter][crafter] (Hafner 2021), from which it takes
the achievement-based scoring this baseline reports.

## Run it

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000
```

No preparation step: the environment generates its own worlds, so there is no
dataset to download. `exp_smoke` is `exp000` cut to a few updates, for checking
an installation.

`--override PATH=VALUE` adapts a run to the machine it lands on. Overriding a
*hyperparameter* is a different matter -- it produces a result whose config
exists nowhere in the code, so it can neither be rerun nor compared. Write a
fork instead; that is what they are for.

## Experiments

`exp000` is the baseline and is frozen: improvements land as forks, never as
edits, so a number measured against it stays comparable. Each fork applies one
named change, and its docstring carries a **Hypothesis**, **References**, and
**Results**.

| Experiment | Parent | Change |
|---|---|---|
| `exp000` | -- | PPO, 1M interactions, 256 envs x 16 steps |
| `exp001` | `exp000` | The published 1B recipe: 1024 envs x 64 steps, lr 2e-4 |
| `exp002` | `exp001` | A policy with memory, cheaply: a reset-aware GRU |
| `exp003` | `exp001` | No policy at all: Q-learning with an LSTM |
| `exp011` | `exp001` | The 1B geometry at 100M, as a screening budget |
| `exp013` | `exp001` | A policy with memory, expensively: GTrXL |

The three memory experiments are the point of the chain. `exp002` carries one
vector forward, so remembering costs the same at step 1 and step 10,000;
`exp013` keeps 128 steps individually addressable and pays for it; `exp003`
asks whether the policy gradient was needed at all.

Names match the JAX study these were ported from, so a torch number can be read
against its counterpart. Where a treatment did not survive the port, the
factory's docstring says so under **Not carried over from the JAX ...** -- all
of them were workarounds for XLA's static shapes, which do not exist here.

## Layout

| Path | Contents |
|---|---|
| `game/` | The game: rules, world generation, and the symbolic observation |
| `game/render/` | Watching it: sprites, a pygame viewer, keyboard play, video |
| `env.py` | The learner's view: batched reset, step, and auto-restart |
| `model.py`, `rnn.py`, `gtrxl.py`, `pqn.py` | The four networks |
| `*_train_step.py` | PPO, recurrent PPO, windowed PPO, and Q-learning |
| `metric.py` | Normalized return and the Crafter achievement score |
| `data.py` | The dataset seam for an environment that generates its own data |
| `experiments.py` | The configs above |

`game/` depends on nothing above it -- it advances a world and encodes what the
player sees, knowing nothing of policies or optimizers. Everything else depends
on it.

Watching a policy play needs the optional viewer dependencies:

```bash
uv --quiet run --frozen --group render python -c "
from priml.baselines.craftax.game.render.play import play; play(seed=0)"
```

WASD moves, space acts, digits craft. `record()` writes an mp4 from a trained
policy instead.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/craftax
```

Runs on CPU in half a minute, offline. Three kinds of evidence sit alongside
the unit tests:

- **Parity tests** compare tables, formulas, and generated worlds against the
  reference implementation, and skip cleanly when the optional `craftax`
  dependency is absent. They are what makes "this is the same game" checkable.
- **`testdata/`** holds a bit-for-bit snapshot of a complete PPO update --
  rollout, advantages, clipped loss, gradient clip, and Adam. A change that
  moves one floating-point bit fails it.
- **The learning rule** is compared against the reference implementation
  directly: identical inputs to both advantage recursions, both sets of PPO
  terms, and both Q(lambda) targets. End-to-end equality is not achievable --
  the two draw from different random streams and diverge from the first
  tile -- so the mathematics is isolated and checked exactly, which is the
  only honest cross-implementation comparison available.

Regenerate the golden only when a numeric change is intended, which means
`exp000` has changed and its recorded result no longer describes the code:

```bash
BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/baselines/craftax
```

[craftax-paper]: https://arxiv.org/abs/2402.16801
[craftax-repo]: https://github.com/MichaelTMatthews/Craftax
[craftax-baselines]: https://github.com/MichaelTMatthews/Craftax_Baselines
[pqn-paper]: https://arxiv.org/abs/2407.04811
[pqn-repo]: https://github.com/mttga/purejaxql
[gtrxl-repo]: https://github.com/Reytuag/transformerXL_PPO_JAX
[crafter]: https://arxiv.org/abs/2109.06780
