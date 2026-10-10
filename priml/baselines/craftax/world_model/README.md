# Craftax world model: a trajectory model, ported

A hierarchical world model of Craftax trajectories, trained on token frames of
recorded episodes: a frame encoder reads each decision's board and HUD, a
global transformer runs over the episode's frames and actions, and a local
decoder generates the next frame, its reward and its done flag. It was first
built on episodes captured from PufferLib's C game; here it runs on the port's
Numba game, with no PufferLib and no C.

The data path needs no C either. The port's game is PufferLib's byte for
byte, so a recorded episode (world seed, actions, a state hash every 256
decisions) replays here, and the original implementation's snapshots load as
the State's bytes plus the uint32 environment stream. `capture/` records new
episodes from the port's own trained RL policies.

## The full loop

Train an RL policy, capture its play, train the world model on it, then score
it and dream with it:

1. RL: the port's trainer, `priml/baselines/craftax` (exp000-exp008 and
   exp101-exp113).
2. Capture: run each arm of `capture/experiments.py` and its verifier, which
   record the behaviour mixture into an archive, every closed shard
   replay-verified ([Capture](#capture)).
3. Corpus: `scripts/freeze_corpus.py` freezes it in the arms' shares
   ([Corpora](#corpora)).
4. Train: `python -m priml` on an experiment of `experiments.py`.
5. Report: `scripts/report.py` runs the baselines, the natural-mix
   evaluation, the eval-only matrix, the engine checks and the dreams, and
   writes `summary.json` and `report.md` ([Evaluation](#evaluation-and-the-report)).

`../README.md`, "Reproduce", gives the full-scale steps in order, with the
runs each reads, and the chain at minimum size. At the end-to-end check's
scale (the `e2e_` capture factories), the loop from captured
exp000/exp102/exp103 play to a trained `exp_smoke` takes about 3.5 minutes on
one node, once those three policies are trained. Each capture job is one
`python -m priml` run of a factory, on one GPU: the verifier, given the
launch's id and its worker count, and each arm's worker, given the launch's id
and its seed range:

```bash
w=priml/baselines/craftax/world_model
m=priml.baselines.craftax.world_model.capture.experiments
id=e2e-$(date -u +%Y%m%dT%H%M%SZ)
uv --quiet run --frozen python -m priml $m.e2e_verifier --override launch=$id --override workers=4 &
gpu=0
for arm in e2e_arm0 e2e_arm1 e2e_arm2 e2e_arm3; do
  CUDA_VISIBLE_DEVICES=$gpu uv --quiet run --frozen python -m priml $m.$arm --override worker=0 --override launch=$id &
  gpu=$((gpu + 1))
done; wait
$w/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/e2e --corpus smoke --decisions 1000000
$w/scripts/coverage_report.py /opt/scratch/datasets/craftax/world-model/e2e --corpus smoke --output /opt/scratch/artifacts/craftax/world-model/e2e/coverage.json
uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp_smoke --override dataset.working_dir=/datasets/craftax/world-model/e2e
```

A launch's id must be new, as the UTC time makes it: a rerun of the recipe, or
a top-up of an archive with a higher budget, is a new launch whose verifier
verifies every shard under the root.

## Parity

The port was checked byte for byte (no tolerance) against the original
implementation, through fixtures that implementation's own code minted:

| What | Scale |
|---|---|
| Training | 3 exp013 updates on one H200: all 657 model and optimizer tensors at every checkpoint |
| Modules | exp013's init, outputs and gradients: CPU, and GPU in fp32 and bf16 |
| Data stream | exp013's first 24 training and every validation micro-batch, every field, on three archives |
| Replay | Every hash and stored snapshot of every 16th episode of three archives' shards; the frames of 20 episodes |
| Recording | 61 random-play episodes: receipt, actions, hashes and frames |
| Exact games | The original viewer's bundles from their resets: every frame and creature-health record of a 3,888-decision death on the Ice floor and a 4,777-decision victory; on glibc, the State's hash after each of 3,888 and 1,000 decisions |
| Capture | Four archives of the original capture worker: fresh, epsilon 0.25-0.75, stall cap 40, and 24 branches; every shard's records, snapshots and summaries, manifests but for provenance |
| Verifier | The original verdicts on those four and two damaged copies, but for wall time |
| Corpus tools | `freeze_corpus` corpora and `data_derive` shards; `coverage_report` and `data_stats` JSON. A capped derived episode also holds its final hash and is truncated, so it replays; the original dropped both |
| Engine rollouts | The base model, seed 0: from new worlds, from real prefixes under forced actions with a mid-run re-prefill, under a policy reading decoded observations, teacher-forced through a terminal; every tensor, on CPU fp32 and on an H200 under deterministic algorithms in fp32, bf16 and through `GraphedStep` |
| Dream report | Episode statistics, the dream evaluation's features and tests, rule departures, the dream summary, every viewer bundle file |
| Engine checks | Equal under deterministic algorithms; in default mode three numbers of the report-only bf16 entry vary, as between two runs of the original |
| Evaluation | The base model's three seeds: baselines, 512 natural tiles (2.1M decisions), the 3 x 3 eval-only matrix, the summary and its criteria; every digit but `zstd19_bpb`, which the port compresses per micro-batch |
| Frozen feature | The engine's features on the base model and the sole world model's early and mature fits, CPU fp32, across resets, mid-episode windows and re-prefills; their weights and the random-init weights tensor by tensor |
| Fidelity | Equal wherever nothing is sampled; bootstrap bounds within 2 ulps under the suite's pinned BLAS kernel (equal under MKL's default). Sampled dreams differ, within each rate's 95% interval |

These checks read a private implementation's fixtures, so they are not part of
this package. A GPU policy capture is not compared bit for bit: the original
sampled its actions in its C loop. The same capture logic is held to it under
the same random-legal streams, and every policy capture is checked by replay
(every live hash and token, and the verifier).

## Speed

exp014 for 300 updates on one H200, the port and the original implementation
alternating (port, original, original, port) in one allocation, 16 CPUs. Both
train with FlashAttention 4 and one prefetch thread.

| | Port | Original |
|---|---|---|
| Median update | 0.975 s, 0.976 s: 8,396 decisions/s | 0.973 s, 0.977 s: 8,402 decisions/s |
| Validation pass | 60.1 s, 59.3 s | 58.7 s, 58.8 s |
| Start-up, warm caches | 27.9 s | 27.5 s |
| Start-up, cold | 72.2 s (Numba compiles replay) | 55.4 s |
| Validation bpb after 300 updates | 0.2848, 0.2836 | 0.2836, 0.2864 |

Training speed is equal: the two implementations differ by 0.1%, the
original's own two runs by 0.4%. The training losses differ from the
original's by 0.0046 per logged update, against 0.0033 and 0.0042 between two
runs of either, which GPU nondeterminism sets.

## Data

| Corpus | Holds |
|---|---|
| `archive-v1` | the base corpus as frame shards |
| `archive-v1-replay` | the same episodes as records and snapshots; replay gives the same micro-batches |
| `archive-v2`, `archive-v2-branch` | dataset v2, stall-capped, and its branches |
| `e2e` | 10.6M verified decisions of the four `e2e_` arms; a 1.44M-decision corpus |
| `smoke` | The `smoke` capture of the RL `exp_smoke`'s policy; corpus `smoke` |

Every root lies under `/opt/scratch/datasets/craftax/world-model`. A
`ReplayStream` left at its default reads `archive-v1`'s corpus `base`, which
the capture arms and `freeze_corpus.py` write; exp000-exp013 read it.
`exp_smoke` reads `smoke`'s. The original implementation's corpora are not
distributed and no step here produces them, yet four experiments name them
under `/opt/scratch/datasets/craftax/world-model-reference/`: exp014 and
exp015 read `archive-v1-replay`'s base corpus, and the flat comparison,
exp003, exp004 and exp005, its small corpus. exp020 reads `archive-v1`'s
`scaleup-v1`, every verified shard of one node of the original capture.

Speed settings are slots on the pieces they change:
`WorldModel.Config.embed_board = multi_hot_board` (default `gathered_board`),
each SwiGLU's `split_gate_projection`, each local block's `recompute_policy =
keep_attention`, the local streams' `cast_stream = to_autocast_dtype`, and an
untied `output_table = Embedding.Config(...)`. None changes a state-dict key,
the initialization or a computed value. The validation set is the slot
`ReplayStream.Config.validation`: `StratifiedWindows.Config(batches=...)` by
default, or `EvalSpans.Config(...)`. An experiment sets
`dataset.micro_batches_per_step` beside `step.accumulate_grad_batches`, and
the loop refuses the two when they differ.

## Capture

`capture/` records the port's trained RL policies into the world model's
archive. Every episode `CaptureEnv` hands over is a record (seeds, actions, a
state hash every 256 decisions and after the last), a floor trace and a
summary line. Its frames are not taken live: game and replay are one
implementation, so the shard writer replays each record, checking every live
hash, and stores the snapshots; an episode that does not replay fails its
shard.

| Module | Does |
|---|---|
| `capture/seeds.py` | World seeds, the 1-in-20 validation split, sampling and rollout seeds |
| `capture/env.py` | `CaptureEnv`: `CraftaxEnv`'s five buffers over a recording game; epsilon range, stall cap, branch starts, budget drain |
| `capture/step.py` | The buffer step: one `nogil` Numba kernel per buffer (play, observe, record) |
| `capture/source.py` | `PolicySource` (an RL experiment's step config plus a checkpoint, driven by `Rollout`) and `RandomSource` (uniform legal play) |
| `capture/worker.py` | `CaptureWorker`: training and validation shard streams, budget, resume, markers |
| `capture/shards.py` | Encode episodes on a thread pool, publish in order |
| `capture/verify.py` | `ReplayVerifier`: replay a sample of every closed shard, halt the archive on a mismatch |
| `capture/control.py` | `HALT.json` and the launch's start and completion markers |
| `capture/branches.py` | Branch pools, and `BranchFeeder`, which replays parents to their branch states |
| `capture/experiments.py` | The behaviour-mixture arms, their v2 fresh and branch arms, and the verifier |

A launch's jobs meet only through one archive root -- the arm policies, the
archive, and the start, completion and halt markers -- so they run on one
machine, or on machines sharing that filesystem; `scripts/gather.py` copies
shards between roots. The arms, of a 3.5B-decision target, read their policy
runs' final checkpoints, at the paths those runs write; the published archive
played exp103's seed-74 run (77.66%), exp102's seed-73 run (56.24%) and
exp000's run (48.45%, from PufferLib's init):

| Arm | Factory | Policy | Share |
|---|---|---|---:|
| 0 | `arm0` | exp103 at 20B | 70% |
| 1 | `arm1` | the same, epsilon 0.05 | 10% |
| 2 | `arm2` | exp102 at 20B | 10% |
| 3 | `arm3` | exp000 at 3.5B | 10% |

One H200 captures 831k decisions/s from exp103 at 1,024 environments in 4
buffers and 1.16M/s from exp000 (rollout only; the encoders run beside it). A
worker over budget records no new episode but finishes the ones in flight, one
per environment, so no episode is cut, recorded or dropped for its length. It
overshoots its budget by those episodes, and long episodes are the likeliest
to be in flight: in the e2e archive (64 environments, budgets of 700k, 100k,
100k and 100k decisions) the four arms published 6.27M, 0.21M, 2.30M and
1.79M decisions, a 59/2/22/17 mix rather than 70/10/10/10.
`freeze_corpus.py --decisions` draws the arm shares back out of an archive
(72/9/9/9 on e2e); `--all` and `--combine` keep the archive's realized mix.

## Corpora

| Script | Does |
|---|---|
| `scripts/freeze_corpus.py` | Freeze whole published shards in the arm shares (`--decisions`), every shard (`--all`), or join corpora (`--combine`) |
| `scripts/coverage_report.py` | Index a corpus, count its coverage (`coverage.py`), and write the sufficiency report |
| `scripts/data_stats.py` | Episode statistics per split and arm, and idle measures of drawn shards |
| `scripts/data_derive.py` | A corpus of whole or capped episodes drawn at random |
| `scripts/gather.py` | Copy a worker's published shards between archive roots, manifests last |
| `scripts/branch_pool.py` | Select and shuffle the branch points of an arm's late floors |

## Inference

`engine.Engine` decodes many episodes at once, one per row, and `dream.py` runs
them open-loop from new worlds or real prefixes, with the model's action head,
a policy fed the decoded observation, or recorded actions. Every frame's
tokens and per-slot log-probabilities are kept, so a rollout becomes
statistics (`episodes.py`, the dream evaluation) and viewer bundles alike.

| Module | Does |
|---|---|
| `engine.py` | Batched per-row decoding: a static global KV cache per row, Gumbel-max sampling under the schema's masks, re-prefill of full rows, and `GraphedStep`, the step as CUDA graphs on any device. It runs its own masked SDPA in place of the attention kernel slots, so it refuses a global kernel other than `SdpaVarlen` or `Flash4Varlen` and decoder kernels other than `SdpaFused` |
| `dream.py` | Open-loop rollouts and the invalid-cell rule |
| `episodes.py` | Episodes cut at a horizon; summaries, total-variation comparison, and rule departures (HUD maxima, floor changes without a ladder) |
| `codec.py` | The game's 843-float observation to token values and back |
| `session.py` | One engine row played decision by decision; its token stream, save and reload |
| `rules.py` | The game's legal actions read off a decoded frame |
| `viewer/` | `bundle.py` writes model bundles and `exact.py` exact games (their docstrings define the formats); `games.mjs build` inlines both into one self-contained page; `render_check.mjs` loads every game in headless Chrome ([Exact games](#exact-games)); `policy_view.py` and `games.mjs panel` make the policy-view panel ([The policy-view panel](#the-policy-view-panel)) |
| `scripts/dream.py` | Dreams from new worlds beside real episodes from their world's reset (branches are skipped): `report.json`, `samples.pt`, bundles |
| `scripts/dream_eval.py` | Free-running dreams and prefix continuations of `--rows / 2` real windows, with bootstrap intervals and permutation nulls; an odd `--rows` or an arm too short for the windows is refused before anything is generated |
| `scripts/engine_checks.py` | The five engine checks on a trained checkpoint and a real segment |
| `scripts/games.py` | A capture archive's average and best episode (`rank`), and one episode as an exact game (`save`) or a policy-view bundle (`panel`) |

```bash
w=priml/baselines/craftax/world_model
$w/scripts/dream.py CHECKPOINT --rows 256 --decisions 4000 --reference 512 --bundles 32 --output OUT/dreams
node $w/viewer/games.mjs build --output OUT/dreams.html OUT/dreams/bundles/generated-{0..31}
node $w/viewer/render_check.mjs OUT/render-check OUT/dreams.html
```

Each script takes `--experiment` (default `experiments.exp001`), repeatable
`--override PATH=VALUE` (the run's launch overrides) and `--device`; dreams
sample in bfloat16 on CUDA and in float32 elsewhere (`dream_eval.py --dtype`
overrides it). The page opens from disk; a render check exits 1
when a page fails to load, offers no games, logs an error or draws a blank
board, in the policy view or, for an exact game, the full map.

## Exact games

The same page plays games the policy played, decision by decision, exactly as
the game played them: the policy view, the full 48x48 map with every creature
and its health, heatmaps of where the player went and acted, the resources
and reward charts, and the inventory, with a picker across checkpoints that
shows each one's average beside its game. The original implementation's
replay viewer recorded games with a C++ build of PufferLib's game;
`viewer/exact.py` instead replays a capture archive's record (world seed,
actions, a state hash every 256 decisions and after the last) on the port's
game and writes the viewer's layout, a 7,858-byte frame and a 120-byte
creature-health record per decision, byte for byte. The capture archive is
the record the original's instrumented evaluation trace was, so its `rank`,
`extract` and `save` become `scripts/games.py rank` and `save`; `games.mjs
build` is unchanged.

From a checkpoint to one page:

1. Capture: a capture factory records the checkpoint's play into an archive,
   as the ghost overlay's `capture_<tier>` do (`../ghosts/experiments.py`): a
   budget of one decision records each environment's first episode and no
   other, unbiased by length, and one arm gives every checkpoint the same
   worlds. The factory's policy config must be the checkpoint's experiment's.
2. Rank and save: `scripts/games.py rank` prints the archive's mean
   achievement return over its complete episodes and its best episode (the
   highest return, then the fewest decisions); `save` replays the best, or
   `--episode NAME`, its first `--limit N` decisions if given, checks every
   hash its record holds, and writes a bundle whose manifest carries that
   mean as the checkpoint's average. Save on the libm the capture ran on:
   world generation and the daylight curve go through it.
3. Build and check, where Node and Chrome are.

```bash
w=priml/baselines/craftax/world_model
a=/opt/scratch/artifacts/craftax/games
$w/scripts/games.py rank ARCHIVE
$w/scripts/games.py save ARCHIVE $a/NAME/best --title "TITLE"
node $w/viewer/games.mjs build --output $a/games.html --page-title "Craftax across training" --picker-label Checkpoint --default-game last $a/NAME/best
node $w/viewer/render_check.mjs $a/render-check $a/games.html
```

The page is one file with no external asset; it opens from disk. A recorded
page of four policies on the same 256 worlds took 7 minutes on one H200: each
capture about 85 s, each save 6 to 39 s (5,288 to 97,432 decisions). Their
averages, 51.66% (exp000 from PufferLib's init), 58.21% (exp102 seed 73),
69.68% (exp103 seed 73) and 76.43% (exp103 seed 74), lie within 3.3 points of
their runs' final evaluations, and the 14.0 MB page renders all four games. A policy trained to stop at the necromancer's
fall plays on here, to its death or the 100,000-tick timeout: exp103 s73's
best game is 97,432 decisions long.

### The policy-view panel

A page that shows a game beside other figures embeds one panel instead: the
window and HUD the policy saw before each decision, on one canvas scaled by a
whole number of device pixels to its container's width. `scripts/games.py
panel` replays an episode as `save` does and writes a policy-view bundle
(`viewer/policy_view.py`): each exact frame cut before its map, 885 bytes, so
`hud.js` draws it as the viewer does. `games.mjs panel` writes the panel's
script, `hud.js` and `policy_view.js` in one function, which defines
`CraftaxPolicyView` and nothing else:

```bash
w=priml/baselines/craftax/world_model
$w/scripts/games.py panel ARCHIVE OUT --episode val/arm3/w0/shard-000000/34 --limit 3822
node $w/viewer/games.mjs panel --output OUT/policy_view.js
```

```js
const view = await CraftaxPolicyView.mount(container, { bundleUrl, spritesUrl });
view.show(t);  // the frame before decision t; from t = view.decisions on, the last
view.caption.textContent = '…';
```

`show` draws nothing for the frame already shown and redraws the HUD only when
it changes: 0.15 to 0.22 ms a call in Chrome, over a 3,822-decision game. A
page whose host `fetch` cannot reach passes `loadBundle(url)`, which resolves
to the bundle's bytes or their base64 text.

## Evaluation and the report

`scripts/report.py` evaluates the checkpoints of one experiment (the seeds of
a recipe, or one run) and writes each step's JSON, `summary.json` (the
criteria verdicts and seed spreads) and `report.md`:

| Step | Code | Writes |
|---|---|---|
| Trivial baselines on the run's own validation windows | `baselines.py`, `scripts/baselines.py` | `baselines-NAME.json` |
| Natural-mix NLL on fixed validation tiles, per stratum, arm, timeout and repeated-frame class | `scripts/data_eval.py` | `data-eval-NAME.json` |
| The in-loop validation metric on every sampler seed's windows | `report.validation_metric` | `evalonly-NAME-on-vV.json` |
| Engine checks | `scripts/engine_checks.py` | `engine-checks-NAME.json` |
| Open-loop dreams beside real episodes | `scripts/dream.py` | `dreams-NAME/` |
| Training stability and trends from W&B (`--wandb-runs`) | `scripts/curves.py` | `wandb-report.json` |

Scoring uses the experiment's own kernels and autocast on CUDA (`scoring.py`).
Every flag is checked before OUTPUT is created, and W&B is read before
anything is scored. An engine check that crashes counts as a failed criterion
4, not an unmeasured one. `--summarize` rebuilds the summary from an existing
directory, whose `settings.json` names the checkpoints and their seeds (it
refuses checkpoints, `--names`, `--sampler-seeds` or `--window-seeds` beside
it), so a host logged in to W&B can add the training criteria to a GPU node's
evaluation. Beside the report: `scripts/fidelity.py` and
`scripts/fidelity_compare.py` (teacher-forced NLL on fixed spans, rule
violations, recorded-action continuations), `scripts/data_compare.py` (seeded
`data_eval` reports against a control) and `scripts/curves.py` (training
objectives against a reference).

```bash
w=priml/baselines/craftax/world_model/scripts
$w/report.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt --output /opt/scratch/artifacts/craftax/world-model/report
$w/report.py --summarize --wandb-runs RUN --output /opt/scratch/artifacts/craftax/world-model/report
```

`RUN` is the training run's W&B id. The recorded report scored three seeds of
exp001, 0, 1 and 2, each its own run: `report.py` takes several checkpoints and
their `--sampler-seeds`.

On the base model's three seeds the whole report takes 82 minutes on one
H200, most of it the dreams (256 x 4,000 decisions per checkpoint).

## The frozen feature for RL

`feature.py` feeds the RL policy a trained world model's state of the
episode: each step, the global stack's final-normed hidden state at `obs_t`,
reading the episode as training does (`start, obs_0, act_0, obs_1, ...`). Its
weights stay frozen. On exp105's policy at 250M transitions the original
implementation measured 47.1% averaged over the 150M, 200M and 250M
evaluations, against 11.9% without it and 11.7% with random frozen weights;
the RL package's exp106-exp108 replicate those arms.

`WorldModelFeature` loads the weights once per process from a slot
(`TrainedWeights`, a checkpoint in the port's layout; `InitialWeights`, an
experiment's init under a seed) and makes a `FeatureEngine` per actor buffer.
The engine keeps a static KV cache per row over the global stack up to the
tap, encodes frames from the float32 observation (the policy's bf16 copy
loses the health grid above 10 HP), and runs one two-query pass per step,
`[start or act_{t-1}, obs_t]`, through the world model's own modules. A step
reads no device value and draws no random number, so the rollout's step graph
captures it between the ingest and the policy's forward, and the rollout
stores each step's feature for the learner. Every `hook_interval` steps,
outside the graph, `ensure_room` raises on its sticky counters (a step
without room, an out-of-schema frame, a non-finite feature), then makes room
for the next block as the `history` slot says:

- `Refill(t_max, keep)`, the default and every RL experiment's: a row that
  cannot take the next block re-prefills from its last `keep` decisions at
  position 0, as a mid-episode training window. A context holds `keep` to
  `t_max / 2` decisions.
- `Sliding(decisions)`: every decision reads exactly its last `decisions`
  decisions, as the original sole world-model policies did. While those
  extend the episode's `start` (or a mid-episode window's first `obs`) the
  cache serves the step. Once a row's window slides, its context is `obs, act,
  ..., obs_t` from position 0, no `start`, recomputed every step from a ring
  of its last `decisions` decisions: one full forward of `2 decisions - 1`
  positions per sliding row. `ensure_room` lists the rows that can slide
  within the block and sizes that fallback batch to whole chunks of
  `fallback_rows`, the fewest that hold them, so the recompute has one shape
  and pads less than a chunk; each of the rollout's `(slot, buffer)` step
  graphs is captured once per batch size, the captures sharing one memory
  pool. Telemetry: `feature/sliding_fraction`, `feature/fallback_capacity`.

A compiled kernel (`compile`) has one budget of shapes, dynamo's
`recompile_limit` of 8, which the actor, an evaluation and a joint learner
share, and a full-graph compile past it raises; `FeatureKernels` counts a
joint `Sliding` source's: six per global-block function, four for the frame
encoder.

`FeatureEngine.context()` names each row's context after its last step: its
decisions and whether `start` leads them. `context_inputs` lays a context out
as the global stack's inputs from position 0, for the engine's prefills and
sliding windows. The cache is not checkpointed: a resumed run begins a window
in every row.

A practice restore puts a donor's world into a row; the `practice` slot says
what the row's history becomes. `FreshWindow` begins a mid-episode window at
its next step. `DonorHistory` resumes the donor's: the saving rows write their
histories (rings, counters, the action that led to the next frame) into an
archive in the step graph, beside their carries and with the same timing, and
a restore copies one in, re-prefills the row's context and hands the rollout
the donor's previous action for the row's next step. Under `Sliding` the
context is the donor's own; under `Refill` it is what a re-prefill at the save
would have kept. The archive is not checkpointed: after a resume, a row
restored from an entry its donor has not saved again begins a fresh window.

With `joint`, the learner trains the weights and publishes them into the
source's `model` between rollouts, copied into its tensors in place (the
captured graphs address them). Each row's rings then hold its longest
context's raw frames (the codec's uint8 cells and int16 aux) as well, and
`Rollout.rebuild_features` (`FeatureEngine.rebuild`) re-encodes them and
re-prefills every row's context under the new weights; a `DonorHistory`
archive keeps raw frames, which a restore encodes afresh. The rollout stores
the learner's inputs per slot (`RolloutStorage.frame_cells` and the rest):
each step's frame, the action that led to it, and its context's decisions
`L_t` and anchor `A_t`; and before step 0, each row's last decisions, as far
back as its first contexts reach. Step `t`'s feature is the final-normed
hidden at `obs_t` of the last `L_t` decisions of the prefix and steps,
`start, obs, act, ..., obs_t` when anchored, else `obs, act, ..., obs_t`,
from position 0. The learner lays those contexts out itself, packed
(`context.plan_replay`); `scripts/feature_gates.py`'s `window_segment` and
`context_features` run them through the training forward, the reference. An
evaluation reads the same live weights through a frozen view of the source
(`WorldModelFeature.frozen`): no frames, no learner's inputs.

A feature refuses the symbolic view, and, like the engine, a frame-encoder
kernel other than `SdpaFused` or a global one other than `SdpaVarlen` or
`Flash4Varlen`.

| Gate | Result |
|---|---|
| Engine against training: CPU fp32 engine vs the training forward on the history it holds | Within 2e-6 on unit-RMS hiddens, across resets, windows and re-prefills |
| The same, `Sliding`: each step vs the training forward of its own window | Within 2e-6, before, at and after a row slides, across resets, windows and both practice slots' restores |
| The same, `joint`: features after new weights and a rebuild; a slot's stored inputs replayed through the training forward | Within 2e-6 of the forward under the new weights; each stored feature within 2e-6 of its stored context's, both histories, across restores, re-prefills and slides |
| Parity with the original engine | Byte-equal on the base model (x86) and the sole world model's early and mature fits (Mac), CPU fp32 |
| Production precision: FA4, bf16, compiled and graphed vs the fp32 training forward, 8 validation episodes x 640 steps | Relative RMS 0.879%, min cosine 0.99989, argmax agreement 99.84% (the original: 0.819%, 0.99991, 99.90%) |
| Graphs: graph replay vs eager; one row's reset | Bit for bit; the other rows unchanged |
| Graphs, `Sliding`: graph replay vs eager, one graph per fallback plan | Bit for bit across plans 0, 1, 2 and 4 of 4 rows; a rollout's in-graph donor saves and restores equal the eager engine's |
| Init: `proj_feature` at zero | Parameters, generator state, outputs and every other gradient equal the control's |
| Feature off | Every RL golden unchanged, CPU and H200 |
| Speed, 256 rows | Step 5.54 ms (the original 5.66), re-prefill 76.8 ms per 128 rows (79.0) |
| Speed, `Sliding`, 512 rows, exp001's size, eager kernels | Step 17.1 ms with no row sliding (`Refill` 17.1), then about 2.1 ms more per sliding row: 50.7 ms at 16, 289 ms at 128, 1,118 ms at 512 |

`scripts/feature_gates.py` runs the production-precision gate, the engine
gate and the step timing on a trained checkpoint and real frames. A buffer's
first re-prefill compiles its chunk shape, and Inductor's autotuning
synchronizes the device, so `ensure_room` holds the rollout's capture lock.

## Layout

| Module | Does |
|---|---|
| `replay.py` | Token frames read from the game State; replay and record episodes; snapshots |
| `capture/` | Record the port's RL policies into an archive ([Capture](#capture)) |
| `archive.py`, `snapshots.py`, `index.py` | Shards, replay shards and their stratum index |
| `coverage.py` | Coverage counters and the corpus sufficiency report |
| `data.py`, `batch.py` | The replay stream: stratified windows, packed |
| `schema.py` | The token vocabulary and each slot's allowed IDs |
| `model.py`, `attention.py`, `grid_encoder.py` | The hierarchical model and its varlen global attention; the local blocks use priml's `Attention` for self- and cross-attention |
| `flat.py` | The flat model: every slot one global position |
| `loss.py`, `metric.py`, `train_step.py` | Restricted-softmax loss, bits per byte, the step |
| `experiments.py` | exp000-exp020 (the flat comparison is exp003-exp005) and exp_smoke |
| `checkpoint.py`, `scoring.py` | Rebuild a trained model; score it with its experiment's kernels |
| `engine.py`, `dream.py`, `episodes.py`, `codec.py`, `session.py`, `rules.py`, `viewer/` | Inference ([Inference](#inference)) |
| `baselines.py` | The trivial validation baselines |
| `feature.py` | The frozen feature for RL ([The frozen feature](#the-frozen-feature-for-rl)) |
| `context.py` | The feature recomputed from each step's stored context with a learner's weights, and its gradient: the RL learner trains the world model with the policy |
| `scripts/` | Corpus tools, evaluation, dreams and the report |
| `scripts/joint_speed.py` | Time the joint learner's epoch at exp103's geometry on synthetic contexts (CUDA) |
| `testing.py` | Tiny models and fixtures the tests share |

Not ported: the original's text layout of a decision, its browser export and
play server with the viewer's live mode, the viewer's C++ capture tool
(`viewer/exact.py` replays on the port's game instead), and the tools that
converted its first archive (capture here writes replay shards directly).

## Running

```bash
uv --quiet run --frozen python -m torch.distributed.run --standalone --nproc_per_node=8 -m priml priml.baselines.craftax.world_model.experiments.exp000
uv --quiet run --frozen python -m priml priml.baselines.craftax.world_model.experiments.exp013
```

The global stack runs on FlashAttention 4 (`flash-attn-4`, Linux only);
`exp_smoke` trains a minimum-size model on the CPU. Training the base model,
scoring it and feeding it to the RL policy, step by step, with the results to
expect: `../README.md`, "Reproduce".

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/craftax/world_model
node --test priml/baselines/craftax/world_model/viewer/games.test.mjs
node --test priml/baselines/craftax/world_model/viewer/policy_view.test.mjs
node --test priml/baselines/craftax/world_model/viewer/render_check.test.mjs  # needs Chrome
```
