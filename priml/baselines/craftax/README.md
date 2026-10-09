# Craftax

PufferLib's Craftax trainer, ported: bit for bit what PufferLib computes, 1.46x
faster, with no PufferLib and no C. The game steps on CPU threads in Numba; the
policy and the learner run on the GPU in torch and Triton, captured in CUDA
graphs. Beside it are Craftax_Baselines' recipes and the recipes of our
[Craftax blog post](https://rekursiv.ai/blog/craftax/).

For the blog post, see [Our recipes](#our-recipes-exp100-exp113).

## Table of contents

- [Run the baseline](#run-the-baseline)
- [Results](#results)
  - [Baselines (exp000-exp008)](#baselines-exp000-exp008)
  - [Our recipes (exp100-exp113)](#our-recipes-exp100-exp113)
- [Reproducing the recipes](#reproducing-the-recipes)
- [Parity](#parity)
- [Speed](#speed)
- [Tests](#tests)
- [Files](#files)
- [Acknowledgements](#acknowledgements)

## Run the baseline

From the repository root, after `uv --quiet sync --frozen`:

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000
```

`exp000` is the Craftax baseline of [PufferLib 5.0][pufferlib-5.0]
([`config/craftax.ini`][pufferlib-craftax] at `6ffa5b10`): a four-layer MinGRU
policy trained with PPO and Muon on 2,048 environments, for 3.49B transitions
on one GPU.
`exp_smoke` checks a machine in four tiny epochs.

A run writes to `/opt/scratch/runs/craftax/{experiment_name}`, so the name
decides what a launch resumes: a launch under a used name continues from its
newest checkpoint and logs a `Resuming` warning naming it. Pass
`--override working_dir=...` to start over. A checkpoint holds the whole
pipeline, so a resumed run is bit-identical to one that never stopped. On a
machine with runs from before the experiments were renumbered, the `exp004` to
`exp008` directories hold the runs now named `exp100` to `exp104`.

## Results

Scores are perf: the achievement return as a share of its 226 points, over
about 10,000 episodes of the final evaluation. Every run trained on one H200 and
16 CPUs; Time is one seed's wall time. With several seeds, Score is their mean.

### Baselines (exp000-exp008)

| Experiment | Change | Score | Transitions | Time | Seeds |
|---|---|---:|---:|---:|---|
| [`exp000`](experiments.py) | PufferLib's recipe, from its seed-73 init | 48.45% | 3.49B | 31 min | 73 |
| [`exp001`](experiments.py) | 250M transitions | 15.22% | 250M | 2 min | 73 |
| [`exp002`](experiments.py) | Original Craftax's rules and observation | 12.48% | 3.49B | 64 min | 73 |
| [`exp003`](experiments.py) | Craftax_Baselines' 1B PPO recipe | 12.37% | 1B | 50 min | 42-44 |
| [`exp004`](experiments.py) | Its 1M-interaction geometry | 2.45% | 1M | 14 s | 42 |
| [`exp005`](experiments.py) | PPO-RNN: a reset-aware GRU | 16.69% | 1B | 1.6 h | 42 |
| [`exp006`](experiments.py) | PQN: Q-learning with an LSTM | 16.32% | 1B | 3.0 h | 42 |
| [`exp007`](experiments.py) | exp003 at 100M, a screen | 9.56% | 100M | 5 min | 42 |
| [`exp008`](experiments.py) | GTrXL: gated Transformer-XL memory | 17.44% | 1B | 2.6 h | 42 |

`exp000` and `exp001` equal PufferLib's runs bit for bit: every checkpoint's
weights and the final evaluation, to the last digit. `exp002` switches back
every way PufferLib differs from original Craftax (`docs/differences.md`).
For comparison, the reference implementations scored:

- `exp003`: 12.16%, Craftax_Baselines' JAX runs (their training return).
- `exp004`, `exp005`, `exp007`, `exp008`: 2.21%, 16.10%, 8.98% and 18.16%,
  the earlier JAX port at the same seed.
- `exp006`: about 16.0%, as purejaxql reports.

### Our recipes (exp100-exp113)

Our best experiment is [`exp103`](experiments.py), the blog post's recipe:
72.48% over seeds 73-75 at 20B transitions, about 11 hours per seed on one H200.

| Experiment | Change | Score | Transitions | Time | Seeds |
|---|---|---:|---:|---:|---|
| [`exp100`](experiments.py) | exp000 from the policy's own init | 47.20% | 3.49B | 31 min | 73 |
| [`exp101`](experiments.py) | The port's default kernels and fp32 | 50.27% | 3.49B | 31 min | 73 |
| [`exp102`](experiments.py) | Board encoder, injection, feasibility loss | 52.76% | 20B | 12 h | 73-76 |
| [`exp103`](experiments.py) | Frontier practice, self-imitation, rewards / 8 | 72.48% | 20B | 11 h | 73-75 |
| [`exp104`](experiments.py) | Width 2,048 on 1,024 environments x 512 steps | 65.10% | 20B | 16.5 h | 73 |
| [`exp105`](experiments.py) | A ConvNeXt trunk on the board | 62.01% | 20B | 23 h | 73 |
| [`exp106`](experiments.py) | exp105 for 250M, evaluated every 50M | 12.17% | 250M | 20 min | 73 |
| [`exp107`](experiments.py) | A frozen world model's feature | 42.92% | 250M | 1.6 h | 73 |
| [`exp108`](experiments.py) | That world model at its random init | 11.52% | 250M | 1.6 h | 73 |
| [`exp109`](experiments.py) | A new world at every reset | 60.66% | 6B | 4.4 h | 73, 74 |
| [`exp110`](experiments.py) | exp103's encoder replaced by a frozen world model trained on early-training play | 37.51% | 1B | 6.5 h | 73 |
| [`exp111`](experiments.py) | exp110 with the world model trained on a 20B exp103 agent's play | -- | 20B | -- | 73 |
| [`exp112`](experiments.py) | exp110's world model trained with the policy | 52.10% | 1B | 46 h | 73 |
| [`exp113`](experiments.py) | exp111's world model trained with the policy | -- | 20B | -- | 73 |

- The original runs of `exp102` scored 66.83% (seed 73, the best of 15 runs of
  that family) and 49.90% (seed 74); those of `exp103`, 73.25% (seeds 73-80).
  Same-seed runs spread by up to about 9 points.
- `exp106` to `exp108` score the mean of the 150M, 200M and 250M evaluations;
  the original runs scored 11.92%, 47.08% and 11.67%.
- `exp109` scores at 6B, on exp103's pool.
- `exp110` and `exp112` score their 1B arms (`--override max_steps=1907`). The
  20B runs of `exp110` to `exp113` are still training; their docstrings hold
  the latest evaluations.
- The world model `exp107` reads (`world_model.experiments.exp001`) predicts
  the next frame's cells at 98.83%, the mean of three seeds; see
  `world_model/README.md`.

Each experiment's docstring holds its result in full: every seed, the episode
count and the speed.

## Reproducing the recipes

- The GPU runs need Linux and an H200: the world model's attention is
  FlashAttention 4 (`flash-attn-4`, Linux only). Give each GPU 16 CPUs.
- Runs log to W&B; log in first, or drop the tracker.
- The RL recipes need no data: the game generates its worlds. `exp000` and
  `exp001` start from PufferLib's seed-73 initial weights, converted once to a
  `state_dict` at
  `/opt/scratch/datasets/craftax/goldens-v1/init_seed73.pt`.
- The world-model experiments read a frozen corpus of captured play
  (`world_model/README.md`, "The full loop"), and `exp107` and `exp110` to
  `exp113` read world-model checkpoints under
  `/opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints/`; point
  `step.feature.weights.checkpoint` at a port-trained one instead.

A seed overrides four fields; the default seed is 73:

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp103 --override seed=74 --override step.sampler.seed=74 --override step.env.seed=74 --override experiment_name=exp103_s74
```

Train the world model, one seed per run, then score the three together:

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp001 --override seed=0 --override dataset.sampler_seed=0 --override experiment_name=exp001_s0
w=priml/baselines/craftax/world_model/scripts
c=/opt/scratch/runs/craftax-world-model
$w/report.py $c/exp001_s0/checkpoints/step_00001525.pt $c/exp001_s1/checkpoints/step_00001525.pt $c/exp001_s2/checkpoints/step_00001525.pt --sampler-seeds 0,1,2 --output /opt/scratch/artifacts/craftax/world-model/report
```

`world_model/README.md` covers the report's W&B inputs, the engine `exp110` to
`exp113` run on, and `scripts/joint_smoke.py`, which smoke-tests a joint arm.

## Parity

The port was checked against fixtures PufferLib (pin `6ffa5b10`) minted, byte
for byte, with no tolerance. `exp000` selects the kernel classes of
`lib/compat.py`, which round as PufferLib's kernels do, and keeps PufferLib's
bf16 carry, outputs and stored rollouts; the defaults use Triton's standard
arithmetic and fp32. The fixtures are not distributed, so the suite that reads
them lives beside them; the goldens in `testdata/` run with this package's
tests.

| What | Checked |
|---|---|
| Full runs | exp000 (all 34 checkpoints and the final eval) and exp001 |
| Training | Epochs 1, 2, 4, 200 and 201; a resume; an eval between epochs |
| Learner | 18 minibatches at epochs 0, 1 and 200; Muon; each PPO kernel |
| Actor | The policy step, scan and embedding; the sampler; the rollouts |
| Environment | 2,048 envs x 10,000 steps; all 8,192 pool worlds; every step phase |
| Original Craftax | Against JAX on all nine floors, in distribution |
| Recipe policies | exp102's encoder and exp105's trunk, at init and at 20B, CPU and H200 |

The recipe-policy fixtures carry two deviations: the encoder's dead
previous-action input, and the gradient-scale wrapper at scale 1, whose
round trips are identities in real arithmetic but round the board in bf16.

## Speed

PufferLib and the port interleaved in one allocation: one H200 and 16 CPUs,
the median of two runs each.

| | PufferLib | Port | Port / PufferLib |
|---|---:|---:|---:|
| Training, transitions/s | 1,322,762 | 1,936,186 | 1.46x |
| Eval, 10,000 episodes, s | 30.37 | 23.00 | 0.76x |

On original Craftax's 1B PPO recipe (`exp003`), the port trains at about 332k
transitions/s against Craftax_Baselines' 155k in JAX: 2.1x.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/craftax
```

Unit tests run offline on CPU; those marked `gpu_triton` skip without CUDA.
The Triton classes refuse CPU tensors, so a CPU run selects their torch
counterparts (`TorchPPO`, `TorchPhiloxSampler`, `TorchScan`), whose bits
differ.

## Files

| Path | Contents |
|---|---|
| `experiments.py` | The configs |
| `game/` | The game: rules, world generation, the observation. Numba, CPU |
| `env.py` | The learner's view: 2,048 envs in four buffers on worker threads |
| `model.py` | The MinGRU policy |
| `rollout.py` | Action sampling and the actor's CUDA graphs |
| `train_step.py` | One PPO epoch |
| `metric.py`, `evaluation.py` | The score and the eval protocol |
| `data.py` | The loop's cadence: one tick per epoch |
| `policies/` | The other policies: exp102's encoder, exp003's MLPs, the GRU, PQN and GTrXL |
| `learners/` | The other learners, frontier practice and self-imitation |
| `lib/` | PufferLib's kernel arithmetic, Adam, typed array reads, cost helpers |
| `world_model/` | The world model, its full loop and the frozen feature; its own README |
| `ghosts/` | The blog post's ghost overlay; `ghosts/FORMAT.md` |
| `scripts/` | Golden minting and the joint world-model smoke |
| `testdata/` | Goldens |
| `docs/differences.md` | Every way PufferLib differs from original Craftax |

`isolation_test.py` checks that the port imports no other baseline and that
every public name is read inside the package.

## Acknowledgements

We thank [PufferAI](https://puffer.ai) and Joseph Suarez for
[PufferLib][pufferlib]. This package ports its Craftax environment and trainer,
and its recipe and speed are the bar the port was built to match.

This is a port. The game and the recipe are other people's work:

- **PufferLib** -- Joseph Suarez, the Craftax environment and trainer this
  package reproduces: release 5.0, pin `6ffa5b10`,
  [`config/craftax.ini`][pufferlib-craftax].
  [PufferAI/PufferLib][pufferlib] (MIT)
- **Craftax** -- Matthews et al. 2024, *Craftax: A Lightning-Fast Benchmark
  for Open-Ended Reinforcement Learning*, from which PufferLib's environment is
  ported. [arXiv:2402.16801][craftax-paper] ·
  [MichaelTMatthews/Craftax][craftax-repo] (MIT)
- **Craftax_Baselines** -- the same authors' PPO baseline, whose 1B recipe and
  actor-critic `exp003` restates, commit `7ce36fa`.
  [MichaelTMatthews/Craftax_Baselines][craftax-baselines] (MIT)

All three are MIT-licensed, which permits this port.

[pufferlib]: https://github.com/PufferAI/PufferLib
[pufferlib-5.0]: https://github.com/PufferAI/PufferLib/releases/tag/5.0-experiments
[pufferlib-craftax]: https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/config/craftax.ini
[craftax-paper]: https://arxiv.org/abs/2402.16801
[craftax-repo]: https://github.com/MichaelTMatthews/Craftax
[craftax-baselines]: https://github.com/MichaelTMatthews/Craftax_Baselines
