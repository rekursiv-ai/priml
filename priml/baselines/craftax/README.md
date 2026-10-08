# Craftax: PufferLib's trainer, ported

Open-ended reinforcement learning in a 2D survival game, trained the way
PufferLib trains it -- and bit-for-bit what PufferLib computes. The game steps
on CPU threads in Numba; the policy and the learner run on the GPU in torch
and Triton, captured in CUDA graphs. There is no PufferLib and no C.

`exp000` is PufferLib's Craftax baseline: a four-layer MinGRU policy trained
with PPO and Muon on 2,048 environments, for 3.49B transitions on one GPU. It
matches PufferLib bit for bit and trains faster (below): it selects the
kernel classes of `lib/compat.py`, which round as PufferLib's kernels do, where
the defaults use Triton's standard arithmetic, and it keeps PufferLib's bf16
for the MinGRU carry, the decoder's logits and value, and the stored
log-probabilities, values and rewards, which the defaults keep in fp32
(`state_dtype`, `output_dtype`, `Rollout.Config.dtype`). A checkpoint holds
the whole pipeline, so a resumed run is bit-identical to one that never
stopped; one is written every 200 epochs and the newest two are kept.
`docs/differences.md` catalogs every way PufferLib differs from original
Craftax; `CraftaxEnv.Config` switches each back, and `exp002` runs the
original setup.

`exp102` and `exp103` port the recipe of the
[Craftax blog post](https://rekursiv.ai/blog/craftax/): a board encoder, an
action-effect head, then frontier practice, reward scaling and a stall cap.
The original runs of those recipes trained on an earlier implementation; the
port reproduces their recipes, not their random streams.

## Results

One row per experiment. The port's score was measured on this code, one H200
and 16 CPUs per run; the reference is PufferLib for exp000 and exp001, the
published recipe for exp003 to exp008, and the original runs of each recipe
from exp102 on. The two are never mixed: `--` means no reference exists, and
"Not run" that the port has not trained it. Scores are perf, the achievement
return as a share of its 226 points, over about 10,000 episodes of the final
evaluation unless the row says otherwise; several seeds give their mean, then
each seed.

| Experiment | Change | Score (port) | Reference | Budget, seeds |
|---|---|---|---|---|
| `exp000` | PufferLib's recipe, from its seed-73 init | 48.452%, bit for bit | 48.452% (PufferLib) | 3.49B, seed 73 |
| `exp001` | 250M transitions | 15.2224%, bit for bit | 15.2224% (PufferLib) | 250M, seed 73 |
| `exp002` | Original Craftax's rules and observation | 12.48% | -- | 3.49B, seed 73 |
| `exp003` | Craftax_Baselines' 1B PPO recipe | 12.37% (12.79 / 11.99 / 12.34) | 12.16% (Craftax_Baselines' JAX runs, their training return) | 1B, seeds 42-44 |
| `exp004` | Its 1M-interaction geometry | 2.45% | 2.212% (the earlier JAX port) | 999,424, seed 42 |
| `exp005` | PPO-RNN: a reset-aware GRU | 16.69% | 16.099% (the earlier JAX port) | 1B, seed 42 |
| `exp006` | PQN: Q-learning with an LSTM | 16.32% (7.15% before the BatchRenorm and loss fixes) | About 16.0% (purejaxql, as reported) | 1B, seed 42 |
| `exp007` | exp003 at 100M, a screen | 9.56% | 8.976% (the earlier JAX port) | 100M, seed 42 |
| `exp008` | GTrXL: gated Transformer-XL memory | 17.44% | 18.159% (the earlier JAX port); 18.3% reported | 1B, seed 42 |
| `exp100` | exp000 from the policy's own init | 47.20% | -- | 3.49B, seed 73 |
| `exp101` | The port's defaults, not PufferLib's bits | 50.27% | -- | 3.49B, seed 73 |
| `exp102` | Board encoder, injection, feasibility loss | 52.76% (56.24 / 54.77 / 50.01 / 50.02) | 66.83% (seed 73; a rerun 57.62%), 49.90% (seed 74) | 20B, seeds 73-76 |
| `exp103` | Frontier practice, self-imitation, rewards / 8 | 72.48% (67.51 / 77.66 / 72.27) | 73.25% (seeds 73-80); its first run 75.30% (seed 73) | 20B, seeds 73-75 |
| `exp104` | Width 2,048 on 1,024 environments x 512 steps | Not run | 65.10% (seed 73) | 20B |
| `exp105` | A ConvNeXt trunk on the board | Not run | 62.01% (seed 73) | 20B |
| `exp106` | exp105 for 250M, evaluated every 50M | 12.17% | 11.92% (seeds 73-75) | 250M, seed 73; mean of the 150M, 200M and 250M evaluations |
| `exp107` | A frozen world model's feature | 42.92% | 47.08% (seeds 73-75) | As exp106 |
| `exp108` | That world model at its random init | 11.52% | 11.67% (seeds 73-74) | As exp106 |
| `exp109` | A new world at every reset | Not run | 60.80 / 60.51% (seeds 73 / 74) on the pool at 6B | 20B |
| `exp110` | A frozen world model, early fit, the sole encoder | 37.51% at 1B; 20B training | -- (ended before any evaluation) | 1B of the 20B schedule, seed 73 |
| `exp111` | The mature fit | 20B training | -- (ended before any evaluation) | 20B, seed 73 |
| `exp112` | exp110's world model trained with the policy | 52.10% at 1B; 20B training | -- (ended before any evaluation) | 1B of the 20B schedule, seed 73 |
| `exp113` | exp111's world model trained with the policy | 20B training | -- (ended before any evaluation) | 20B, seed 73 |

The reference's exp102 run at seed 73 was the best of its 15 runs of that
family. The experiments' docstrings hold each result in full, with its
episodes, speed and the 20B runs' latest evaluations.

The world model that exp107 reads, `world_model.experiments.exp001`:

| Experiment | Run | Port | Reference |
|---|---|---|---|
| World model, the base model | exp001, 50M decisions, seeds 0 / 1 / 2 | Not trained on the port yet | Next-frame cell accuracy 98.87 / 98.75 / 98.88% (copying the frame: 82.5%); action NLL 0.504 / 0.547 / 0.485 nats (empirical frequencies: 2.06) |

Its reference row is the port's `world_model/scripts/report.py` scoring the
original implementation's three checkpoints, which reproduces that
implementation's own report to every digit; the port has not trained one
itself. The world model's parity, at every layer below a full run, is in
`world_model/README.md`.

## Reproducing the recipes

### Setup

- From the repository root, `uv --quiet sync --frozen`. The GPU runs need
  Linux and an H200: the world model's attention is FlashAttention 4
  (`flash-attn-4`, a Linux-only dependency).
- An RL run steps the game on CPU worker threads: give each GPU 16 CPUs, as
  every speed figure below assumes.
- Runs log to W&B; log in first, or drop the tracker.
- Data:
  - The RL recipes need none: the game generates its worlds.
  - `exp000` and `exp001` start from PufferLib's seed-73 initial weights,
    saved by PufferLib (`puf_save_weights`) and converted once to a
    `state_dict`, at
    `/opt/scratch/datasets/craftax/goldens-v1/init_seed73.pt`. The other
    experiments draw their own.
  - World-model training reads a frozen corpus of captured play: record one
    with `world_model/capture/` and freeze it with
    `world_model/scripts/freeze_corpus.py` (`world_model/README.md`, "The
    full loop"), then point `dataset.working_dir` at its archive. The
    experiments name the original corpus, under
    `/opt/scratch/datasets/craftax/world-model-reference/`.
  - The frozen feature (exp107) and the sole world-model arms read world-model
    checkpoints in the port's layout under
    `/opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints/`; point
    `step.feature.weights.checkpoint` at a port-trained one instead.

Each RL seed overrides four fields; the default seed is 73:

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp103 --override seed=74 --override step.sampler.seed=74 --override step.env.seed=74 --override experiment_name=exp103_s74
```

### 1. Our recipe (exp103)

One H200 and 16 CPUs per seed, about 12 hours for 20B transitions at about
509k transitions/s. It evaluates every 1B transitions and ends with a
10,000-episode evaluation; the seeds share the W&B group `exp103`. Expect the
row above: 67-78% per seed. Same-seed runs of these recipes spread by up to
about 9 points.

### 2. The world model (exp001), then its report

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp001 --override seed=0 --override dataset.sampler_seed=0 --override experiment_name=exp001_s0
```

exp001 is the 8-GPU recipe accumulated on one H200: 1,525 updates of about
32,760 decisions each, about 4.1 s per update (about 2 hours). A seed run sets
both `seed` and `dataset.sampler_seed`. Each run writes
`/opt/scratch/runs/craftax-world-model/exp001_sK/checkpoints/step_00001525.pt`.
Then score the three seeds together (about 82 minutes on one H200):

```bash
w=priml/baselines/craftax/world_model/scripts
c=/opt/scratch/runs/craftax-world-model
$w/report.py $c/exp001_s0/checkpoints/step_00001525.pt $c/exp001_s1/checkpoints/step_00001525.pt $c/exp001_s2/checkpoints/step_00001525.pt --sampler-seeds 0,1,2 --output /opt/scratch/artifacts/craftax/world-model/report
```

The stability criteria read the training curves from W&B: add the three runs'
IDs as `--wandb-runs ID0,ID1,ID2` (project `craftax-world-model` of the
default entity, or `--wandb-project ENTITY/PROJECT`), or leave them for a
later `--summarize` from a host logged in to W&B. `report.md` gives the four
criteria's verdicts and the numbers of the world-model row above;
`world_model/README.md` describes each step and its options.

### 3. RL with a frozen world-model feature

exp106 is exp105's policy for 250M transitions, about 20 minutes at 213k
transitions/s; exp107 adds the trained world model's feature, and exp108 the
same model at its random init, each about 1.6 hours at 43k transitions/s.
Each evaluates at 50M, 100M, ..., 250M. The comparison is the mean of the
150M, 200M and 250M evaluations, per arm, against the row above. To feed
exp107 a world model from recipe 2, set `step.feature.weights.checkpoint` to
its checkpoint.

### 4. RL with a frozen world model as the sole encoder

Each arm (exp110, exp111) is exp103's recipe with its encoder removed: the
policy reads only the frozen world model's 1,152-wide state, projected to
1,024. Both run the original implementation's engine, refill: a context of
256 to 512 decisions, a row that would overflow restarting from its last 256
(`world_model/README.md`). The reference runs read exactly the last 512
decisions; the engine keeps that as a setting (`Sliding`, `DonorHistory`),
but it recomputes every row past 512 decisions each step and slowed to
0.5-1.3k transitions/s as episodes grew, so refill is the method. The 1B run
is exp110 stopped after 1,907 epochs (999,817,216 transitions) on the 20B
schedule, as the reference's was. Every arm evaluates 10,000 episodes every
250M transitions and at the end, on 1,024 environments: training's caches
leave no room for 2,048 (`exp110`'s docstring has the measurements). From
their init the arms train at 52-61k transitions/s; expect 37-45k once
contexts fill, about 6 days for 20B.

### 5. RL training the world model with the policy

Each arm (exp112, exp113) forks a frozen arm of recipe 4 with one change, the
train step's `feature_training`: the learner trains the world model's feature
weights with the policy, in one Muon group after the policy's. Each window
recomputes its features from the contexts the actor read
(`world_model/context.py`), steps whose contexts begin at the same decision
sharing one causal pass, and before each rollout the trained weights are
published into the actor's model and every row's history rebuilt under them.
`scripts/joint_smoke.py` runs a few epochs of an arm, then one evaluation,
and reports each epoch's learner, rollout and rebuild seconds, the
actor-to-learner feature gap and peak memory (`--sliding` runs the arm on
exact windows); `world_model/scripts/joint_speed.py` times the learner's
epoch alone on synthetic contexts. From their init the arms trained at 6-11k
transitions/s, peaking at 113 GiB reserved and 137 GiB with an evaluation of
1,024 beside it; the reference's joint learner took 6,214 s an epoch, where
the port's took 88 s on refill contexts.

## Parity

The port was checked bit for bit against PufferLib's trainer: every row below
is byte-equal, with no tolerance, unless it says otherwise. The checks compare
against fixtures PufferLib (pin `6ffa5b10`) minted, and against the original
recipe runs' own code. The fixtures are not distributed, so the suite that
reads them is kept beside them, outside this package; the goldens in
`testdata/` need no fixture and run with this package's tests.

| What | Checked |
|---|---|
| exp000, a full run | 3.49B transitions: the fp32 masters at all 34 checkpoints, and the final eval, 48.452% |
| exp001, a full run | 250M transitions: the masters at epochs 200, 400 and 476, and the final eval, 15.2224% |
| Training epochs | Epochs 1, 2, 4, 200 and 201 from PufferLib's seed-73 init; a resume after epoch 2; an eval between epochs |
| Learner | 18 minibatches at each of epochs 0, 1 and 200; the Muon step; each PPO kernel |
| Actor | The policy step, scan and embedding; the sampler; exp000's rollouts |
| Environment | 2,048 envs x 10,000 steps, masked-random and under the 20B policy; all 8,192 pool worlds; every step phase |
| Final eval | 100 episodes from the init and 300 from the 20B weights: every rollout's aggregate and every env's log |
| Original Craftax (exp002's options), against JAX | JAX transitions on all nine floors; 11,200 random episodes a side and world counts, compared in distribution (`docs/differences.md`) |
| The recipe runs' policies: exp102's board encoder and exp105's ConvNeXt trunk | From each run's init and its 20B checkpoint, CPU and H200: the learner window's logits, values, carry, feasibility targets and loss, every gradient, and an actor step |
| Goldens in `testdata/`, no fixture needed | 37 files. Any CPU: the tiny model, Muon, PPO and sampler, the tiny pipeline's rollouts, train epochs (exp000's and exp002's recipes) and evals, and the configs of exp000, exp003 and exp103. glibc x86-64: env trajectories. Any GPU: the exact sampler. RTX 5090: exp000's kernels at the same size: model, rollouts, train steps and Muon |

The recipe-run fixtures were minted by the original implementation's own code,
with the port's two deviations applied: D1, the encoder's dead previous-action
input, and D2, the gradient-scale wrapper at scale 1, whose `t + (b - t)`
round trips are identities in real arithmetic but round the board in bf16. A
second ConvNeXt set keeps D2's round trips, and the port matches it once those
round trips are restored. On the 20B checkpoint they move 797 of 903 logits,
so neither form has the other's bits. The ConvNeXt CPU fixtures are minted and
compared on one thread: at width 2,048 the CPU's bf16 reductions split by
thread count.

## Speed

PufferLib and the port interleave A B B A in one allocation, so CPU placement
cancels out: one H200 (NUMA node 0) and 16 CPUs. Each figure is the median of
two runs.

| Row | PufferLib | Port | Port / PufferLib |
|---|---|---|---|
| Training, transitions/s (30 epochs, the first 5 skipped) | 1,322,762 | 1,936,186 | 1.46x |
| Eval, 10,000 episodes from the 20B weights, s | 30.37 | 23.00 | 0.76x |

Both sides scored perf 0.4860696 over 10,039 episodes. Each kernel was timed
the same way, and each component's throughput compared with the one PufferLib
recorded beside its fixtures.

The refactored tree keeps that speed: exp000 at 1.928M transitions/s, 0.998x
the pre-refactor tree in one allocation with byte-equal masters, and the
default configuration (standard kernels, fp32 carry, value head and storage)
at 1.996M, 1.03x exp000.

Against original Craftax, on its rules and its published 1B PPO recipe
(exp003), one H200 each, seeds 42/43/44:

| Row | Craftax_Baselines (JAX) | Port (exp003) |
|---|---|---|
| Normalized return, mean of 3 seeds | 12.16% | 12.37% |
| Training, transitions/s | ~155k | ~332k (2.1x) |

The JAX figures are its training return over the last 20 updates, as
`ppo.py` reports it; the port's are its final evaluation.

## Attribution

This is a port. The game and the recipe are other people's work:

- **PufferLib** -- Joseph Suarez, the Craftax environment and trainer this
  package reproduces, pin `6ffa5b10`, `config/craftax.ini`.
  [PufferAI/PufferLib][pufferlib] (MIT)
- **Craftax** -- Matthews et al. 2024, *Craftax: A Lightning-Fast Benchmark
  for Open-Ended Reinforcement Learning*, from which PufferLib's environment is
  ported. [arXiv:2402.16801][craftax-paper] ·
  [MichaelTMatthews/Craftax][craftax-repo] (MIT)
- **Craftax_Baselines** -- the same authors' PPO baseline, whose 1B recipe and
  actor-critic `exp003` restates, commit `7ce36fa`.
  [MichaelTMatthews/Craftax_Baselines][craftax-baselines] (MIT)

All three are MIT-licensed, which permits this port; the notices above are
those licenses' attribution requirement.

## Layout

| Path | Contents |
|---|---|
| `game/` | The game: rules, world generation, the observation. Numba, CPU |
| `env.py` | The learner's view: 2,048 envs in four buffers on worker threads |
| `model.py` | The MinGRU policy; its scan and embedding are `priml/model/{min_gru,embedding}.py` |
| `rollout.py` | Action sampling and the actor's CUDA graphs |
| `train_step.py` | One epoch; the learning rule is `priml.loss.policy_gradient`, the optimizer `priml.optimizers.fused_muon` |
| `metric.py`, `evaluation.py` | The score and the eval protocol |
| `data.py` | The loop's cadence: one tick per epoch, what builds an evaluator per eval |
| `experiments.py` | The configs |
| `testing.py` | Test support: tiny configs, a fake env, packed observations |
| `policies/encoder.py` | exp102's board encoder (`BoardEncoder`: cells, status MLPs, `BoardCNN`); the board's residual blocks are a slot, which `ConvNextTrunk` fills for exp105 |
| `policies/actor_critic.py` | Craftax_Baselines' actor-critic MLPs, exp003's policy |
| `policies/rnn.py`, `learners/rnn_update.py` | exp005's policy and learner: the reset-aware GRU actor-critic, and PPO epochs over shuffled whole trajectories, each replayed from its starting carry |
| `policies/pqn.py`, `learners/pqn_train_step.py` | exp006's Q-learner: the recurrent Q-network, its epsilon-greedy and greedy samplers and previous-action feature, and the step that regresses each rollout toward its Q(lambda) targets |
| `policies/gtrxl.py`, `learners/gtrxl_train_step.py` | exp008's policy and learner: the gated Transformer-XL over a memory of past steps, and PPO epochs over windows of shuffled whole trajectories, each window started from the memory the actor held there |
| `learners/update.py`, `lib/adam.py` | exp003's learner: standard PPO epochs over shuffled transitions, and Adam behind a global clip |
| `learners/practice.py`, `learners/imitation.py` | Frontier practice and the branch self-imitation, exp103's |
| `lib/compat.py` | PufferLib's kernel arithmetic: the kernel classes `exp000` selects |
| `lib/arrays.py`, `lib/costs.py` | Typed reads of numpy arrays, and the shared cost helpers each model's `cost()` prices with |
| `world_model/` | The world model, its full loop, and the frozen feature a policy can read; its own README |
| `ghosts/` | The blog post's ghost overlay: many captured episodes of each policy on one world; `ghosts/FORMAT.md` |
| `scripts/` | The references the game's kernels are checked against live: glibc's `rand_r` and the libm light level (`mint_goldens.py`); a joint world-model arm's CUDA smoke (`joint_smoke.py`) |
| `testdata/` | Goldens: the reference and headline configs, bits at test size and on the GPU, env trajectories, the C state layout |
| `docs/differences.md` | Every way PufferLib differs from original Craftax, and the option for each |

`game/` depends on nothing above it. The parity suite imports the port and the
port imports nothing of it; `isolation_test.py` checks that, and that every
public name is read inside the package, so the parity suite needs no dead code
here.

## Running

```bash
uv --quiet run --frozen python -m priml priml.baselines.craftax.experiments.exp000
```

`exp001` is the same recipe at 250M transitions, `exp002` the recipe on
original Craftax's setup, rules and 8,268-float observation, `exp003` that
setup under Craftax_Baselines' 1B PPO recipe (its MLP, Adam and shuffled
minibatches), `exp100` the recipe from the policy's own init, `exp101` that
run on the port's defaults (standard kernels, fp32 carry, value head and
storage), and `exp_smoke` checks a machine in four tiny epochs. `exp004`
is exp003 at Craftax_Baselines' one-million-interaction geometry (256
environments x 16 steps, Adam at 3e-4), `exp005` exp003 with
Craftax_Baselines' PPO-RNN policy (a reset-aware GRU, learned from whole
trajectories), `exp006` exp003's setup learned by PQN (an LSTM Q-learner, no
replay buffer or target network), `exp007` exp003 at 100M, a screening
budget, and `exp008` exp003 with transformerXL_PPO_JAX's gated Transformer-XL
(128 steps of memory, learned from 64-step windows of whole trajectories).
The port's own recipes start at `exp100`.

`exp102` and `exp103` port the board-encoder recipe and our recipe (20B
transitions). `exp104` is exp102 at width 2,048 on 1,024 environments of 512
steps (the original runs' capacity arm). `exp105` puts a ConvNeXt trunk on
the board: a 1 x 1 lift to 64 channels, two ConvNeXt blocks (5 x 5
depthwise, per-cell LayerNorm, 256-wide GELU MLP, layer scale from 1e-6) and
a zero-initialized projection back onto the 16-channel board. `exp106` stops
exp105 after 250M transitions, inside its 20B schedule's linear warmup,
evaluating every 50M. `exp107` adds a frozen world model's per-step feature
(the train step's `feature` slot and the policy's zero-initialized
`proj_feature`; `world_model/README.md`), and `exp108` reads it from the
world model's random init: the three arms of the frozen-feature experiment.
`exp109` trains exp103 in a newly generated world at every reset and
evaluates it on exp103's pool. `exp110` removes exp103's encoder: its policy
reads only a frozen world model's feature (the first stage `NoEncoder`, so no
`proj_in`, and a `proj_feature` drawn as every projection is), fitted to
early play, on the original engine's refill; `exp111` swaps in the mature
fit. `exp112` and `exp113` train each one's world model with its policy (the
train step's `feature_training`). Seeds 74 and 75 override `seed`,
`step.sampler.seed`, `step.env.seed` and `experiment_name` (`exp105_s74`),
which files the run under the recipe's W&B group.

A run's `working_dir` defaults to `/opt/scratch/runs/craftax/{experiment_name}`,
so the name decides what a launch resumes: a launch under a used name
continues that run from its newest checkpoint, and says so in a `Resuming`
warning naming the checkpoint, its step and when it was written. Pass a fresh
`working_dir` (`--override working_dir=...`) to start over. The experiments
were renumbered once: on a machine holding runs from before, the `exp004` to
`exp008` directories hold the runs now named `exp100` to `exp104`, which a
launch of today's `exp004` to `exp008` there would resume.

One H200 and 16 CPUs train exp102 at 512k transitions/s, exp104 at 341k and
exp105 at 211k (the original runs measured about 337k and 240k for the last
two). The world-model feature brings exp106's 213k to about 43k (the
original trainer, about 256k and 46-49k).

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/craftax
```

Unit tests run offline on CPU; those marked `gpu_triton` skip without CUDA.
The Triton classes refuse CPU tensors, so a CPU run selects their torch
counterparts (`TorchPPO`, `TorchPhiloxSampler`, `TorchScan`), whose bits
differ.

[pufferlib]: https://github.com/PufferAI/PufferLib
[craftax-paper]: https://arxiv.org/abs/2402.16801
[craftax-repo]: https://github.com/MichaelTMatthews/Craftax
[craftax-baselines]: https://github.com/MichaelTMatthews/Craftax_Baselines
