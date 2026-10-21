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

Original definitions, ingestion statements and source-only minters remain
external. Native replay loads only complete source-minted numeric PT records,
renaming their `bn_*` and `classifier` keys to the current nouns; every stored
key, shape, dtype and bit is compared. The external definition
fixture SHA-256 is
`173eef407c6dd75d96e721c1d781dd7618d05ba26331ab8eb87a6a49f4b6c4ef`;
decoded definitions:
`e115694dd3ce0140b24048a988a0a57d2228e6d7589a35deb41480567651d4fb`.
The original-ingestion fixture SHA-256 is
`5c948cc66a582f09b4d29ca46fa71ed1d9f6c275f050cd4e0a1dabc19e97bc56`.

| Complete control | SHA-256 |
| --- | --- |
| multiscaleeegnet.pt | `d50d87638bd4f0952831649417187dad642969e069a3f81696bafcbfc2d37003` |
| shallowconvnet.pt | `e7c4312ee80d96d729396105819d5b6bb568f56aadf8b9852e9c8e905f65b431` |
| attentioneegnet.pt | `b944b96da8f2982219eea6af3aab2b269998a9cf3d9ee2f4328c07e27a2dad54` |
| filtering.pt | `16efd9c52c23c20600b7dbfafaf7502a0c21ec8e8fe8ef9cb1372125ee5ad2fc` |
| ingestion.pt | `2cdc27af9e919fc45925f306615a02c6fd17bfe0da674eca5e7e0b8e26c604d7` |
| subject_folds.pt | `6a41ff0a2297569433f1d45868afe8d891cf52791666a03c41e5c243b8426f43` |

The three neural controls are 14,450/4,192/8,233 bytes in ensemble order. They
contain complete initial/final tensors, three mean losses, final gradients,
all AdamW tensors, logits/probabilities and sampler-site counts. A real Priml
loop separately checks complete model/optimizer/epoch-schedule/random state.
Filtering records compare complete float64 input/output arrays. Ingestion records cover
parsed decimals, missing markers, cue boundaries, windows, labels and IDs;
fold records cover every validation index for unequal counts and tied counts.
Test-marker/padding and submission tests check their complete native outputs.

All six controls were captured from unchanged pinned originals on x86_64 on
2026-10-17, using the same BFB implementation as native replay (capture SHA
`87f7880ea686cfd85bcb6c598d1721ca17b173453fb4b2843f3d21c7ab8a17e7`).
Original ARM controls remain external; the three current neural files retain
the bytes above. Current ARM replay is unmeasured.

The shared `host_agnostic_numerics()` testing boundary fixes the adjacent-pair
CPU float64 convolution weight reduction tree internally. Input/bias gradients
and non-target kernels retain their ordinary dispatch. The same reduction
applies to original minting and native replay;
production arithmetic remains native. It replaced five differing
ShallowConvNet gradient/AdamW fields after replay of all six complete backward
inputs isolated the weight reductions. Thread/MKL controls did not resolve
them. Earlier ARM/x86 traces retained 17 differing float64 scratch fields in
BatchNorm, elementary functions and a norm; none reached float32 outputs or
stored fields. Archived before/after source controls remain external.

A later shared testing-context repair copied legacy `native_batch_norm`
running-buffer updates back despite its incomplete mutation schema. Before
replacement, source and native loaded every archived old initial field and
matched through three updates. Corrected original-source captures changed
only initial running variances to 0.9; other initial parameters/buffers retained
their bits. Constructor regressions require the dummy pass to update every
BatchNorm counter/variance. Complete old controls remain external.

Preprocessing designs both filters without SciPy, in plain Python arithmetic
that repeats SciPy's NumPy operations with x86 glibc's hypot and x86 OpenBLAS's
complex dot order, so coefficients and filtered records match on x86 and
aarch64. It preserves the unfused IIR recurrence and row-major buffers. Layout
fixes the subsequent NumPy reduction order; all 90 ingestion values match after
standardization and
window extraction. [Filtering provenance](testdata/filtering.md) records the
original input cases, source definition hash and capture environment.

Shared RNG pins intercept uniform initialization, Bernoulli dropout, DataLoader
integer seeds and permutations beneath the original sampling algorithms.
Unknown seeded primitives raise. Existing site keys/counts and full values are
preserved; initializer locations use encounter order within each layer type.
Additional full-file site/generator-state metadata does not change the old PT
keys. Python/Torch source seeds and pending first-use CUDA seed are preserved;
complete NumPy/hash-environment state parity remains open.

Sensitivity tests reject one ULP in all six model-record families for every
architecture and added uniform/Bernoulli or omitted dummy-dropout draws. A
uniform logit shift preserving argmax still fails every changed bit. Historical
full/reduced source path measurements cover 140 lines/156 arcs identically,
with zero native line/arc changes in the then-current model/layer/training/loss
paths. They measure constructors, three updates and evaluation, not a full
production run.

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
