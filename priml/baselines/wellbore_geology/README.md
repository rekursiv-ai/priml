# Wellbore geology native public-best study

[ROGII: Wellbore Geology Prediction][kaggle]
predicts true vertical depth along horizontal wells from visible depth anchors,
gamma-ray observations and paired reference curves. The final deadline was
**August 5, 2026, 23:59 UTC**; Kaggle now offers late submissions. The official
metric is pooled original-row RMSE. The prize pool was $50,000.

`exp000` implements the complete six-family, three-repeat, five-fold recipe:
**90 native trained models**. The [winning solution][winner]
reports private RMSE 5.639, public RMSE 5.980 and OOF RMSE 4.627. Full native
score reproduction and H100/H200 five-minute qualification remain unmeasured.

## Prepare inputs

[DATA.md](DATA.md) gives the Kaggle download and one-time staging steps. It
also describes the small test fixtures.

Use the frozen repository environment. In a worktree, set `UV_NO_SYNC=1` and
`WORKERS_DEFAULT=1` so validation preserves the shared environment.

The staging command copies paired train/test CSVs and the official sample
submission, excludes images, and builds the geographic centroid map. Each
validation prior uses training-fold suffixes and the query's visible prefix.
Train-only geology columns do not enter model features.

Download the [ConvNeXt-small initialization][weights] at the pinned revision:

```sh
uv --quiet run --frozen hf download timm/convnext_small.in12k_ft_in1k_384 \
  --revision e9f96b3e3e8be16706d721c30d85112a87b82a8b \
  --include model.safetensors \
  --local-dir /opt/scratch/datasets/rogii-ruby-fade6112/initialization
ln -s model.safetensors \
  /opt/scratch/datasets/rogii-ruby-fade6112/initialization/convnext_small.safetensors
sha256sum /opt/scratch/datasets/rogii-ruby-fade6112/initialization/model.safetensors
```

Expected SHA-256:
`476bbb20b9d3b2b47e2a98228b216b8c912fbad5fd7b2052f703e3117797a605`.
The native constructor loads every local backbone tensor before converting
normalization. The winner used an unversioned initializer identifier; this pin
provides reproducible setup without establishing historical weight identity.

Generate the particle-feature family caches of an experiment through native Priml:

```sh
uv --quiet run --frozen python -m priml.baselines.wellbore_geology.scripts.prepare_pf \
  priml.baselines.wellbore_geology.experiments.exp000 --check-only
uv --quiet run --frozen python -m priml.baselines.wellbore_geology.scripts.prepare_pf \
  priml.baselines.wellbore_geology.experiments.exp000
```

The experiment names its dataset and cache roots; the default cache root is
`/opt/scratch/caches/wellbore_geology/exp000-pf`. Complete records are isolated
by family and split, with an exact configuration manifest. Existing records are
reused; a different configuration is rejected. Use one writer per cache
directory. A family with `prepare_particles=True` (`exp004` onward) generates its
own missing records inside the run, charged to the wall allowance; every other
family requires this cache to be complete before the run.

## Run the study

```sh
uv --quiet run --frozen python -m priml \
  priml.baselines.wellbore_geology.experiments.exp000 --pprint
uv --quiet run --frozen python -m priml \
  priml.baselines.wellbore_geology.experiments.exp_smoke
uv --quiet run --frozen python -m priml \
  priml.baselines.wellbore_geology.experiments.exp000
```

Runs default to `/opt/scratch/runs/wellbore_geology/{experiment_name}`; particle
caches default to `/opt/scratch/caches/wellbore_geology/{experiment_name}-pf`.
Declare a new experiment function with its own `experiment_name` for another run.
Set `working_dir`, `pf_directory` or `dataset_directory` to explicit `Path`
values in that function when using other locations. Finalization resolves string
templates without accessing data; an explicit `Path` stays literal. Run directories
must be new, and a family that prepares its own particles requires an empty particle cache.

`exp000` retains the complete 200/300-epoch schedules, raw simulation, filtering
and backward paths, geographic priors, all encoder/decoder stages and heads,
compound loss, AdamW, cosine decay, gradient clipping and whole-state EMA.
Repeated OOF predictions are
aggregated after original-row smoothing. Final model predictions are averaged
in float32 bins in a fresh FP32 interpreter, then expanded and blended.

`exp_smoke` retains every stage and family with one epoch, one repeat, two folds
and narrower model/data geometry. The separate candidate chain changes one recipe
choice at each named parent edge:

| Experiment | Parent | Change |
| --- | --- | --- |
| `exp001` | `exp000` | Latest family and its required single-family publication |
| `exp002` | `exp001` | One repeat |
| `exp003` | `exp002` | First fold |
| `exp004` | `exp003` | 285-second cold allowance, including 30 seconds for evaluation |
| `exp005` | `exp004` | 64 particles |
| `exp006` | `exp005` | Eight particle seed runs |

The complete 200-epoch cosine horizon remains unchanged. Budgeted candidates force
final validation after wall truncation. Measure the entire command:

```sh
/usr/bin/time -o /opt/scratch/artifacts/wellbore-geology-exp006-wall.txt -f '%e' \
  timeout --signal=TERM --kill-after=10 300 \
  uv --quiet run --frozen python -m priml \
  priml.baselines.wellbore_geology.experiments.exp006
```

Downloads precede timing. Cold preprocessing, training, fresh inference,
evaluation and export count toward 300 seconds. A nonzero exit, missing metric
or elapsed time above the allowance fails qualification.

Family directories contain complete CPU checkpoints, their ordered manifest,
serialized native configuration, fold selection history, `oof_df.pqt` and
`inference/submission_details.pqt`. Final publication writes `config.txt`,
`run.json`, `oof.parquet`, `test.parquet` and `submission.csv`. Official sample
IDs must match the complete prediction set exactly. Failed execution records
its error and elapsed time through shared `train.budget.WallBudget`; domain
qualifiers and the published metric live in its receipt's `metadata`.

The package is flat. `model.py` holds every network, `data.py` the well
preparation, `folds.py` the geographic folds and priors, `particles.py` the
particle priors and caches, `train_loop.py` the fold loop, `inference.py` row
prediction and publication, and `pipeline.py` the study with its family
training and fresh inference. `scripts/` owns preparation
commands. Fresh inference runs as
`uv --quiet run --frozen python -m priml priml.baselines.wellbore_geology.pipeline.family_prediction`
with `--configuration`, `--directory` and `--output`; training launches it
itself. `testing.py` holds the replay and fixture tools; production does not
import them. Goldens live under `testdata/`.

## Numerical controls

CPU tests compare against goldens minted in-tree by this package:

```sh
uv --quiet run --frozen pytest priml/baselines/wellbore_geology -n0 -pno:randomly
```

A golden test writes a missing record and fails, so the new file is inspected
before a rerun accepts it. `--regenerate-b4b` (tensors) or `--regenerate-golden`
(config text) rewrites an existing record. Tensor records compute inside
`host_agnostic_numerics()` from `priml.testing.bfb`.
No test loads the scientific source (revision
`fade6112ffebd9e27c08ac7f053e8d4c3d5c319a`) or imports another implementation.
Two computations follow outside implementations: the quadratic features solve
their 3-by-3 normal equations in LAPACK's left-looking LU order with fused
multiply-adds, and the particle tracker's random stream reproduces NumPy's
legacy `RandomState`.

Only `exp000`, the nexus, has records, five in all. The study has a config
print (`exp000_study.txt`), and so does its latest family (`exp000_0803_v2.txt`).
That family also has three augmented model inputs with their random streams
(`augmentation.pt`) and a three-update record (`train_step.pt`) of a two-stage,
345-parameter CPU float32 network: decoder, heads, losses, AdamW state, EMA and
validation history, starting from the loop's own initialization. The tiny
four-stage model has a construction record (`constructor.pt`). Every other
family and every fork has a delta test against its parent.
Data preparation, folds, geographic and particle priors and prediction columns
have unit tests with hand-checkable values. Every tensor record stays below
32 KiB. Full-width GPU numerics and public-score reproduction need separate
measurements.

[kaggle]: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
[winner]: https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby
[weights]: https://huggingface.co/timm/convnext_small.in12k_ft_in1k_384
