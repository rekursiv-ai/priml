# Cross-subject motor-attempt EEG

`exp000` retains the complete strongest publicly released notebook found on
2026-10-15, with an unbounded outer time horizon. Its public score is **0.56**.
`exp006` is the separate five-minute candidate. The leaderboard leader scored
**0.90**; no corresponding public implementation was available. The organizer's
0.5308 starter is a separate reference.

| Source fact | Pin |
| --- | --- |
| Author | Avik Das |
| Notebook | [Cross-Subject EEG Spatio-Temporal Rehab Decoding](https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding?scriptVersionId=353208444) |
| Kaggle version | V1, `353208444` |
| Repository | [Kaggle-Notebooks-Avik](https://github.com/avikds/Kaggle-Notebooks-Avik/blob/e2d892f158c68afa92ecda8229b9ad255c7d0aaf/cross-subject-eeg-spatio-temporal-rehab-decoding.ipynb) |
| Commit | `e2d892f158c68afa92ecda8229b9ad255c7d0aaf` |
| Notebook SHA-256 | `0225f99382e41558a2b318a76d3c093b32934bfd80f0ace302586ee350ab909e` |
| Competition data license | CC BY-NC-SA 4.0 |
| Source runtime | 291.8 seconds; two T4s visible, code uses one CUDA device |
| Source environment | PyTorch `2.10.0+cu128` |
| Released weights | None identified; the notebook trains all models |

The released run logs 50 training files, nine test files, 17 training people,
1,795 labeled cues and 12,524 windows. Test people are S008, S013 and S015.
Filenames and actual markers govern ingestion; these source observations do
not assert the contents of future data revisions.

## Stage and run

[DATA.md](DATA.md) gives the one-time staging steps and describes the small
test inputs.

Stage the archive using existing Kaggle authentication. This does not log in,
join the competition, accept terms or submit predictions.

```sh
uv --quiet run --frozen python -m priml.baselines.motor_imagery.scripts.prepare_data
uv --quiet run --frozen python -m priml.baselines.motor_imagery.scripts.prepare_data --print-config
uv --quiet run --frozen python -m priml priml.baselines.motor_imagery.experiments.exp000
uv --quiet run --frozen python -m priml priml.baselines.motor_imagery.experiments.exp006
uv --quiet run --frozen python -m priml priml.baselines.motor_imagery.experiments.exp_smoke
```

An already authorized download can be extracted with the staging command's
`--archive` and `--directory` flags. Staging records raw SHA-256 hashes in
`manifest.json`. It performs no filtering or training and requires a fresh
destination, including after incomplete staging. Downloads reuse
`/opt/scratch/caches/kaggle/motor-imagery`.

Declare environment paths in a named experiment factory that calls its parent.
Set `cfg.validation.dataset.working_dir` to the staged raw directory: a
logical path resolved beneath the loop's `base_dir`, or a literal `Path` when
`base_dir=None`. Set `cfg.base_dir` and `cfg.working_dir` to the run root and
logical output path. Finalization expands string `{study_name}` and
`{experiment_name}` patterns and gives both loops the campaign's names. Each
run needs a fresh output directory. `--print-config` shows the complete
finalized loop, preprocessing, model and budget tree without data/GPU.

The official release was staged at
`/opt/scratch/datasets/motor-imagery-release-2026-09-14`; its archive SHA-256 is
`8a02e5cc7504addf74cd4d27ed7a327e5b71c32e31d121baad452dcf1c4ebf3c`.
Point the declared data field there to reuse its 50 training and nine test CSVs.

`exp001` changes only the validation protocol to fixed people S002/S006/S019.
`exp002` reduces validation and retraining to three complete passes;
`exp003` changes stride to 250; `exp004` keeps only MultiScaleEEGNet;
`exp005` halves its temporal/projection widths to 4/8; `exp006` adds a 300-second
outer budget. Each factory calls its named parent and has a complete-tree
change test. `exp_smoke` retains all three models at smaller dimensions and
three epochs; it carries no competition result.

## Preserved recipe

| Stage | Public recipe and native implementation |
| --- | --- |
| Input | Eight named EEG electrodes, 250 Hz, sorted CSV filenames |
| Filtering | 50 Hz notch, Q=30; order-four Butterworth 1–45 Hz; zero-phase filtering on axis 0 |
| Alignment | Each move/rest or masked cue_start through the next nonempty marker; final unpaired marker ignored |
| Normalization | Per-cue NumPy population zscore with epsilon 1e-8; no fitted cross-trial state |
| Windows | 500 samples, stride 125; short training cues yield none; short test cues zero-pad |
| Diagnostics | Construct all three disposable models, draw four examples, run each before CV |
| Initialization | CPU construction; train-mode dummy updates BatchNorm and consumes dropout; then transfer |
| Validation | Five subject-disjoint GroupKFold splits; MultiScaleEEGNet, 18 epochs each |
| Selection | First strict improvement from zero window accuracy; retain that epoch's probabilities |
| Training | Batch 64, shuffle, drop_last; float32 inputs; mean off-target-smoothed loss; norm clip 1.0 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-3; source default betas/epsilon |
| Schedule | Shared RecursiveCosine, exact absolute Torch recurrence once per epoch; eta_min 1e-5 |
| Full cohort | Construct all three models together; train each for 16 epochs |
| Inference | Float32 softmax P(move), NumPy mean over every model/window |
| Submission | Sorted file/cue order; ID 0..359; move for probability >=0.5, otherwise rest |

The [overview](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/overview)
lists 1–100 Hz as a permitted minimally filtered representation. The complete
[rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules)
fetched 2026-10-15 contain no prohibition on the source's 1–45 Hz filter. This
port retains it; organizer certification is not inferred. Diagnostic plots,
balanced accuracy and AUC are excluded reports; their model/random draws are
retained where they affect training. The objective is trial accuracy.

The campaign declares both loops it runs: `validation` fits one fold and
`ensemble_loop` fits each member on the whole cohort for 16 epochs without
evaluating, each with its own epoch budget and cosine horizon. A `folds` slot
holds the validation people policy, `GroupedFolds` or `FixedCohort`. The
declared dataset can reconstruct raw inputs itself. Campaign execution shares
one cold preprocessing result through runtime constructor arguments, without
putting modules, trials or tensors in Config; each loop receives only its
model, the trials and the fold's people. All full-cohort models
are constructed before any member trains. The step retains
`loss.mean().backward()`, uses the shared update path and epoch schedule slot,
and evaluates through `call_eval`. State export scopes only the EMA swap.
Compiler, autocast/cache, timers, clipping, nonfinite skipping, EMA, placement,
quantization and activation slots remain available. Accumulation must be 1;
closure optimizers are unsupported. The shared recursive schedule is indexed by
the dataset's epoch timer. The networks name their normalizations `norm_*` and
their output projection `proj_out`; state saved under the source's `bn_*` and
`classifier` names still loads.

A zero-score fold has no selected epoch. Native execution raises before
ensemble training or submission; the source later raises `TypeError` while
thresholding `None`. Zero ties, first-positive selection and campaign failure
are tested. No fallback epoch is part of `exp000`.

## Evidence and provenance

No test reads a notebook output. Every golden is minted in-tree from this
package by the test that replays it (`--regenerate-b4b` or
`--regenerate-golden`). Tensor goldens compute under `host_agnostic_numerics()`
and pinned samplers; every stored key, shape, dtype and bit is compared.

| Golden | Contents and minting test |
| --- | --- |
| multiscaleeegnet.pt, shallowconvnet.pt, attentioneegnet.pt | `init/`: initial state and forward logits (`model_test.py`); `train/`: three epochs of losses, final state, gradients, AdamW state, evaluation outputs and sampler-site counts (`train_step_test.py`) |
| exp000.txt | Complete finalized exp000 configuration (`experiments_test.py`) |

A real Priml loop runs the same three epochs and must equal the final state,
AdamW state and sampler counts in the `train/` part of multiscaleeegnet.pt.
Sampler sites are numbered in encounter order within
each file, so moving a line does not change a golden. Pinned samplers intercept
uniform initialization, Bernoulli dropout and DataLoader integer seeds and
permutations beneath the original sampling algorithms; an unknown seeded
primitive raises. Loop construction restores the campaign's Python and Torch
streams and the pending first-use CUDA seed.

Filters, folds, ingestion and submission have hand-checkable tests: coefficient
structure, band-edge and center gains, steady states, literal fold members and
windows composed from the documented steps. Preprocessing designs the notch
from its closed form and the bandpass by prewarped bilinear transform of the
analog Butterworth prototype. Each filter starts from the steady state of a
unit step at its DC gain. Zero-phase filtering runs each filter forward, then
backward, over an odd extension of three filter lengths at each end.

## Timing and remaining measurements

The shared WallBudget clock starts before cold preparation and includes fit,
evaluation, export and durable receipt publication. Downloading is excluded.
`exp000` is uncapped; `exp006` arms an outer alarm for 300 seconds. Native calls
may defer Python signals; over-budget completion cannot report success.
`result.json` contains shared `status`, `budget_seconds`, `elapsed_seconds` and
failure `error`; its `metadata` holds run identity, source pins, hardware,
submission count, parity status and `eval/accuracy`. The latter is
subject-disjoint trial accuracy with same-heldout source epoch selection,
not nested hyperparameter evaluation. Catchable failures and interrupts after
output creation produce failed receipts. Caller runtimes survive cleanup;
SIGKILL, power loss and unwritable storage cannot guarantee a final receipt.

Historical GPU component measurements predate this native layer composition.
On 2026-10-15, GB10/Torch 2.11cu128 matched all 50 training CSVs, 1,795 cues and
12,524 windows; each full-sized model matched three batch 64 updates, including
complete initial/updated tensors, logits, gradients, AdamW and random outputs.
A 193-window corpus exercises dropped remainders. Primitive counts were 20/12/22;
a fresh process preserved CPU state, CUDA seed 42 and the first 17 CUDA draws.
The verifier took 28.858 seconds excluding interpreter startup. H200 components
matched the same complete scope in 37.393 seconds, also excluding startup.

Native runs at `90ac943b1a88323d667b06fe517d7f960a2b18b3` used one H200 with
Torch 2.11cu128. Outer walltime included startup, raw preprocessing, fitting,
evaluation, inference, checkpoints and submission; acquisition was excluded.

| Historical recipe | Walltime | Local trial accuracy | Scope |
| --- | --- | --- | --- |
| exp000 | 237.53 s | 1,208/1,795=0.6729805014 | Five 18-epoch folds; three 16-epoch fits; 360 rows |
| Combined candidate, then exp001, now exp006 | 47.16 s | 164/305=0.5377049180 | Fixed holdout; three passes; 360 rows; 300-second cap |

These scores are neither Kaggle scores nor nested estimates. These timings do
not establish source/native full trajectories. Open measurements are real test
CSV parity, complete fold trajectories, native T4/PyTorch 2.10 and H100 execution,
H100/H200 full-run CUDA lockstep, production-batch random/branch coverage and
H100 outer walltime. No tolerance-based or metric-only result closes those gaps.
