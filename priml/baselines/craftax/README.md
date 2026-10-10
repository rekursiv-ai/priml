# Craftax

PufferLib's Craftax trainer, ported: bit for bit what PufferLib computes, 1.46x
faster, with no PufferLib and no C. The game steps on CPU threads in Numba; the
policy and the learner run on the GPU in torch and Triton, captured in CUDA
graphs. Beside it are Craftax_Baselines' recipes and the recipes of our
[Craftax blog post](https://rekursiv.ai/blog/craftax/).

For the blog post, see [Our recipes](#our-recipes-exp101-exp113).

## Table of contents

- [Run the baseline](#run-the-baseline)
- [Results](#results)
  - [Baselines (exp000-exp008)](#baselines-exp000-exp008)
  - [Our recipes (exp101-exp113)](#our-recipes-exp101-exp113)
- [Reproduce](#reproduce)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
  - [The RL experiments](#the-rl-experiments)
  - [The world-model chain](#the-world-model-chain)
  - [The ghost overlay](#the-ghost-overlay)
  - [Smoke test of the chain](#smoke-test-of-the-chain)
  - [The base_dir knob](#the-base_dir-knob)
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
on one GPU. The policy draws its own init from PufferLib's distributions,
under PufferLib's seed.
`exp_smoke` checks a machine in four tiny epochs.

A run writes to `/opt/scratch/runs/craftax/{experiment_name}`, so the name
decides what a launch resumes: a launch under a used name continues from its
newest checkpoint and logs a `Resuming` warning naming it. Pass
`--override working_dir=...` to start over. A checkpoint holds the whole
pipeline, so a resumed run is bit-identical to one that never stopped.

## Results

Scores are perf: the achievement return as a share of its 226 points, over
about 10,000 episodes of the final evaluation. Every run trained on one H200 and
16 CPUs; Time is one seed's wall time. With several seeds, Score is their mean.

### Baselines (exp000-exp008)

| Experiment | Change | Score | Transitions | Time | Seeds |
|---|---|---:|---:|---:|---|
| [`exp000`](experiments.py) | PufferLib's recipe, from the policy's own init | 47.20% | 3.49B | 31 min | 73 |
| [`exp001`](experiments.py) | 250M transitions | 15.22% | 250M | 2 min | 73 |
| [`exp002`](experiments.py) | Original Craftax's rules and observation | 12.48% | 3.49B | 64 min | 73 |
| [`exp003`](experiments.py) | Craftax_Baselines' 1B PPO recipe | 12.37% | 1B | 50 min | 42-44 |
| [`exp004`](experiments.py) | Its 1M-interaction geometry | 2.45% | 1M | 14 s | 42 |
| [`exp005`](experiments.py) | PPO-RNN: a reset-aware GRU | 16.69% | 1B | 1.6 h | 42 |
| [`exp006`](experiments.py) | PQN: Q-learning with an LSTM | 16.32% | 1B | 3.0 h | 42 |
| [`exp007`](experiments.py) | exp003 at 100M, a screen | 9.56% | 100M | 5 min | 42 |
| [`exp008`](experiments.py) | GTrXL: gated Transformer-XL memory | 17.44% | 1B | 2.6 h | 42 |

`exp000` and `exp001` once loaded PufferLib's seed-73 init instead of drawing
their own. From it they scored 48.45% and 15.22% and equalled PufferLib's runs
bit for bit: every checkpoint's weights and the final evaluation, to the last
digit. `exp001`'s score is from that init. `exp002` switches back every way
PufferLib differs from original Craftax (`docs/differences.md`).
For comparison, the reference implementations scored:

- `exp003`: 12.16%, Craftax_Baselines' JAX runs (their training return).
- `exp004`, `exp005`, `exp007`, `exp008`: 2.21%, 16.10%, 8.98% and 18.16%,
  the earlier JAX port at the same seed.
- `exp006`: about 16.0%, as purejaxql reports.

### Our recipes (exp101-exp113)

Our best experiment is [`exp103`](experiments.py), the blog post's recipe:
72.48% over seeds 73-75 at 20B transitions, about 11 hours per seed on one H200.

| Experiment | Change | Score | Transitions | Time | Seeds |
|---|---|---:|---:|---:|---|
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
- `exp110` and `exp112` score their 1B arms: the config stopped after epoch
  1,907, 999,817,216 transitions, its schedule keeping the 20B horizon. The
  20B runs of `exp110` to `exp113` are still training; their docstrings hold
  the latest evaluations.
- The world model `exp107` reads (`world_model.experiments.exp001`) predicts
  the next frame's cells at 98.83%, the mean of three seeds; see
  `world_model/README.md`.

Each experiment's docstring holds its result in full: every seed, the episode
count and the speed.

## Reproduce

From a fresh clone, at the repository root, after
`uv --quiet sync --frozen`. Every command is copy-paste exact and runs its
experiment as the config states it. Where Seeds above lists one seed, the
config runs it; where it lists several, the config runs the first, and the
others were the same config with every seed field set to that seed.

### Prerequisites

- Linux and an NVIDIA GPU: the policy's kernels are Triton, so every run,
  `exp_smoke` included, needs CUDA. Every record is from one H200 with 16
  CPUs per run.
- FlashAttention 4 (`flash-attn-4`, Linux only) for the world model and for
  exp107 to exp113, whose feature runs it.
- A W&B login (`wandb login`) for every run but the smoke runs, or
  `WANDB_MODE=offline` to log locally.
- Disk: a run keeps its newest two checkpoints, 2.6 GB for exp000 and 18 GB
  for exp002; a capture archive takes about 0.7 bytes per decision, a few GB
  for the 3.5B decisions of `archive-v1`.

### Data

Nothing to download or prepare. The game generates its worlds, and the world
model's corpus is captured from the RL policies (below).

### The RL experiments

Each command trains one experiment into
`/opt/scratch/runs/craftax/{experiment}/`: its checkpoints,
`checkpoints/step_{epoch:08d}.pt` (the newest two, the last at the final
epoch), and `metrics.json`, the final evaluation.

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp001
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp002
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp003
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp004
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp005
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp006
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp007
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp008
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp101
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp102
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp103
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp104
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp105
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp106
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp107
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp108
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp109
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp110
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp111
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp112
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp113
```

Every experiment reads nothing but its config, except:

| Experiment | Reads | Run first |
|---|---|---|
| `exp107` | `/opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt` | world-model `exp001` ([the chain](#the-world-model-chain), step 4) |
| `exp110`, `exp112` | `/opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints/early-fit-s73/step_00013135.pt` | none here: no experiment trains this fit |
| `exp111`, `exp113` | `/opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints/mature-fit-s73/step_00012738.pt` | none here: no experiment trains this fit |

exp107's record read world-model exp001's seed-0 run. The records and times
are the Results tables'.

### The world-model chain

The world model trains on play captured from three RL policies. Four steps,
in order:

1. **Train the policies.** `exp103` (11 h), `exp102` (12 h) and `exp000`
   (31 min), as above, one H200 each. The published archive's policies were
   exp103's seed-74 run, exp102's seed-73 run and exp000's run.
2. **Capture.** Each of the four arms runs four workers, one per seed range,
   beside one verifier that replays a sample of every shard; each job takes
   one GPU. A launch id names the launch's markers and verifier log, and is
   never reused:

   ```bash
   m=priml.baselines.craftax.world_model.capture.experiments
   id=v1-$(date -u +%Y%m%dT%H%M%SZ)
   uv --quiet run --frozen python -m priml $m.verifier --override launch=$id --override workers=16 &
   for arm in arm0 arm1 arm2 arm3; do for worker in 0 1 2 3; do uv --quiet run --frozen python -m priml $m.$arm --override worker=$worker --override launch=$id; done; done; wait
   ```

   `arm0` and `arm1` read exp103's final checkpoint,
   `/opt/scratch/runs/craftax/exp103/checkpoints/step_00038146.pt`; `arm2`
   exp102's, `step_00038146.pt`; `arm3` exp000's, `step_00006662.pt`. They
   write `/opt/scratch/datasets/craftax/world-model/archive-v1`. One H200
   captured 831k decisions/s from exp103 and 1.16M/s from exp000 (rollout
   only), about 1.2 GPU-hours for the 3.5B decisions.
3. **Freeze the base corpus**, 50M training decisions in the arms' shares,
   into `archive-v1/corpora/base.json`:

   ```bash
   priml/baselines/craftax/world_model/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/archive-v1 --corpus base --decisions 50_000_000
   ```

4. **Train the world model** into
   `/opt/scratch/runs/craftax-world-model/{experiment}/`:

   ```bash
   uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.craftax.world_model.experiments.exp000
   uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp001
   uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp002
   uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp010
   uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp011
   uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp012
   uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp013
   ```

   Each reads the base corpus. exp000 runs on one node of 8 GPUs; the rest
   on one H200, where an update of exp001 takes about 4.1 s, its 1,525
   updates about 1.7 hours, and exp010, exp012 and exp013 ran in 7,051 s,
   6,593 s and 6,579 s. Their recorded results, in each docstring, were
   measured on the original implementation's capture of the same mixture.
   The corpora that exp014, exp015, exp020 and the flat comparison (exp003
   to exp005) read are not produced by this chain; their docstrings hold
   their records.

Then `exp107` above reads world-model exp001's final checkpoint.
`world_model/README.md` covers the corpus tools, the evaluation report, and
`scripts/joint_smoke.py`, which smoke-tests a joint arm.

### The ghost overlay

The blog post's ghost overlay plays three policies' episodes on one world
(`ghosts/experiments.py`, `ghosts/FORMAT.md`). Its tiers read the final
checkpoints of `exp001`, `exp102` and `exp103`, so those train first. The
published page's medium and high tiers were exp102's seed-73 and exp103's
seed-74 runs. Its boss tier reads a policy no experiment here trains.

```bash
m=priml.baselines.craftax.ghosts.experiments
g=priml/baselines/craftax/ghosts
a=/opt/scratch/artifacts/craftax/ghosts/capture
id=pilot-$(date -u +%Y%m%dT%H%M%SZ)
uv --quiet run --frozen python -m priml $m.pilot_verifier --override launch=$id --override workers=3 &
for tier in pilot_early pilot_medium pilot_high; do uv --quiet run --frozen python -m priml $m.$tier --override launch=$id; done; wait
$g/select_world.py $a/pilot --tier early:0:/opt/scratch/runs/craftax/exp001/metrics.json --tier medium:1:/opt/scratch/runs/craftax/exp102/metrics.json --tier high:2:/opt/scratch/runs/craftax/exp103/metrics.json --output $a/world.json
id=w15-$(date -u +%Y%m%dT%H%M%SZ)
uv --quiet run --frozen python -m priml $m.capture_verifier --override launch=$id --override workers=3 &
for tier in capture_early capture_medium capture_high; do uv --quiet run --frozen python -m priml $m.$tier --override launch=$id; done; wait
$g/build.py $a/w15 /opt/scratch/artifacts/craftax/ghosts/site-data --tier early=0 --tier medium=1 --tier high=2
```

The captures play world 15, the world `select_world.py` chose from the
pilots; `extra_<tier>` and `extra_verifier` add the episodes of the short set
the same way, and `endings.py` reports how each tier's episodes end.

### Smoke test of the chain

The whole chain at minimum size, on one CUDA GPU in minutes: a four-epoch
policy, a 20,000-decision capture of it, its corpus, and the world model's
four-step `exp_smoke` on the CPU:

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp_smoke
uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.capture.experiments.smoke
priml/baselines/craftax/world_model/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/smoke --corpus smoke --all
uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp_smoke
```

The capture reads
`/opt/scratch/runs/craftax/exp_smoke/checkpoints/step_00000004.pt` and writes
`/opt/scratch/datasets/craftax/world-model/smoke`, whose corpus `smoke` the
world model reads.

### The base_dir knob

Every path above is `/opt/scratch` joined with a logical path: runs under
`/runs`, captures and corpora under `/datasets`, reports under `/artifacts`.
`--override base_dir=DIR` moves all of one job's paths beneath `DIR`, its
inputs included, so give every job of a chain the same one. The scripts take
their directories as arguments: substitute `DIR` for `/opt/scratch` in them.

## Parity

The port was checked against fixtures PufferLib (pin `6ffa5b10`) minted, byte
for byte, with no tolerance. `exp000` selects the kernel classes of
`lib/compat.py`, which round as PufferLib's kernels do, and keeps PufferLib's
bf16 carry, outputs and stored rollouts; the defaults use Triton's standard
arithmetic and fp32. The fixtures, PufferLib's seed-73 init among them, are
not distributed, so the suite that reads them lives beside them; the goldens
in `testdata/` run with this package's tests.

| What | Checked |
|---|---|
| Full runs | exp000 (all 34 checkpoints and the final eval) and exp001, from PufferLib's seed-73 init |
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
| `scripts/` | The references the game's kernels are checked against, and the joint world-model smoke |
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
