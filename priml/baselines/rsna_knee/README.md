# RSNA Knee Abnormality Detection

`exp000` composes the complete released inference recipe with native Priml
models and independently implemented preprocessing. It uses both Raptor
CoAtNet models, all four R7 ConvNeXt-small folds, ResNet34 laterality, VNet
anatomy, spatial/state/extent/synovitis heads, both calibrated stackers, and
percentile-rank blending. The 384-trained Raptor scores a 320 crop with
anatomical mirror averaging. The second model uses its 224 crop, 0.8 field
crop, and float32 retry. R7 retains its compile checks and all-fold eager
fallback. Failure caps are 3% for each Raptor arm and 50% for R7; R7 keeps the
8.3-hour deadline from recipe start.

The selected public recipe is
[heliosli's version-two 0.957 notebook](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-957-new-one-you-can-use).
The displayed score is reported by that source. Official test labels are
unavailable locally. Historical private weak targets and complete checkpoint
training histories are unavailable, so `exp000` covers released-weight
inference. It does not reconstruct historical training.

## Layout

- `experiments.py`: inference `exp000`, the released recipe (`inference.PublicRecipe`), and the
  cold-training chain `train_exp000`, `train_exp001`.
- `inference.py`: `RaptorStage` and `R7Stage`, execution modes, scoring, stackers, rank
  blend, `ReleasedState` checkpoints and the pipeline.
- `model.py`: `R7`, `FindingAttention`, `Raptor`, `TinyConvNeXt`, `ImageNetEncoder` and
  `ReleasedVNet` (the released anatomy key map). The side network is `priml.model.resnet.ResNet`,
  the Raptor backbone is `priml.model.coatnet.CoAtNet` and the anatomy network is
  `priml.model.vnet.VNet`, all configured in `exp000`.
- `imaging.py`: DICOM geometry, side check, anatomy, anchors, fluid, rendering.
- `train_loop.py`: cold protocol. `train_exp000` is the budgeted recipe, `train_exp001`
  forks it with four studies per batch. The acquisition receipts, cold data, loop and
  budgeted run live here.
- `data.py`: the twelve finding names.
- `scripts/`: `acquisition` and `adopt_receipt` jobs.
- Every path is a logical path under a `base_dir` (default `/opt/scratch`).
- Tests share `support_test.py`.

## Measured scope

- Released-source parity is a port-time check recorded in git history. Every
  checked-in golden is minted in-tree from this package by the test that reads
  it: tensors with `--regenerate-b4b` inside `host_agnostic_numerics`,
  configurations with `--regenerate-golden`. No test compares against bits
  produced by another implementation.
- Goldens cover the nexus only: the finalized `exp000` and `train_exp000`
  configurations; tiny-width forwards of every `exp000` model family (R7,
  Raptor, ResNet34 laterality, VNet anatomy) with literal weights; the
  `train_exp000` Tiny encoder's seeded construction and forward; and three
  updates through `train_exp000`'s own train step and optimizer stack. Every
  binary golden is at most 32 KiB.
- DICOM decoding, physical preparation, rendering, anchors, side correction,
  fluid volumes, windows, scoring, stackers, blending and CSV output are unit
  tests with hand-checkable expectations: small synthetic DICOM files in
  native, deflated, RLE, JPEG and JPEG 2000 transfer syntaxes, literal arrays
  and closed-form values. Frozen model boundaries do not prove learned
  end-to-end GPU inference.
- Complete learned native GPU inference, mixed-precision convolution parity,
  and current cold-training timing remain unmeasured. A historical H200
  source-runtime run took 168.29 seconds for three test studies and matched
  its unchanged notebook control. That result predates this independent
  rewrite and does not qualify it.
- A historical cold-training implementation took 200.29 external H200 seconds
  with 16 CPUs and 64 GB: 9,874 updates, 44 training studies, 14 validation
  studies, and macro AUC 0.620330512741227. All 58 PatientIDs were available
  and groups were disjoint. That receipt does not establish current
  constructor initialization, current numerical parity, or current timing.
- Original programs, loaders, notebook extraction and runtime setup stay
  outside Priml.

Released-source SHA-256 identities:

- `compact_infer.py`: `8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d`.
- `raptor_run.py`: `663e33fcf6022c7470fc0fa3d23e8d57292dc147e2c07609ad5e960588a3162a`.
- `raptor224_run.py`: `2e17e7a2849a38920d9467f508c51ce14f0e6275531ac4fae5c098713c25e380`.

## Inputs and provenance

[DATA.md](DATA.md) gives the one-time acquisition steps and describes the tiny
local test inputs.

Place official CSVs and complete acquired DICOM series beneath
`/opt/scratch/datasets/rsna-knee/raw`. Required CSVs are `train.csv`,
`test.csv`, `sample_submission.csv`, `train_series.csv`, and
`test_series.csv`. Native inference reads these files directly.

Acquire these releases beneath `/opt/scratch/datasets` before timing:

| Directory | Kaggle dataset | Version |
|---|---|---|
| `rsna-knee-r7-v2` | `heliosli/rsna-knee-r6-frozen-fe055406` | 2 |
| `rsna-knee-raptor-swa-v1` | `nartaa/rsna-knee-publication-swa-weights-20261007` | 1 |
| `rsna-knee-orthofoundation-v1` | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | 1 |

`RaptorStage` and `R7Stage` pin complete SHA-256 identities for both Raptor
checkpoints, four R7 checkpoints, laterality, anatomy, state specification,
and both stackers. Every checkpoint is verified before deserialization and
loaded through a strict name/shape/dtype bijection. Native execution requires
no original notebook, original Python module, isolated project, or source
runtime. The recorded notebook SHA-256 is
`85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24`;
the released R7 bundle identity is `8c7b666e27dc0e8f`.

Architecture and recipe provenance includes dreaddevelopment's Raptor work,
nartaa's released weights and anatomical mirror, and
varshithakolupuri's 224-model blend. The released R7 assets preserve MONAI 0.6
VNet anatomy. Public source credits weak-label work by stevenleehans,
pilkwang, and riadmohamed42. Released weights and competition data retain
their publishers' terms. The original work used Osteoarthritis Initiative
resources distributed through the NIMH Data Archive, with NIAMS support:
[study 3407](https://nda.nih.gov/study.html?id=3407),
[dataset DOI](https://doi.org/10.15154/0hcg-f676).

## Run native inference

```bash
UV_NO_SYNC=1 uv --quiet run --frozen python -m priml priml.baselines.rsna_knee.experiments.exp000
```

Set directory fields for another host. The output directory must be new and
separate from all inputs. It contains both Raptor raw/ordinal-rank CSV pairs,
R7 calibrated and scored-membership CSVs, final `submission.csv`, complete
configuration, stage diagnostics, and success/failure receipts. Final rows
follow `test.csv`; finding columns follow `sample_submission.csv`. R7 failure
means are accumulated in sample order, written at nine significant digits,
and read back before the rank blend. Final values are float32.

## Independent cold experiment

`train_exp000` is a separate ImageNet-initialized ConvNeXt-tiny
and finding-attention protocol using official gold labels. Its CSV contains
4,407 studies but only 58 complete twelve-finding gold labels. Missing labels
are not negatives. Knee-selected checkpoints and reports are excluded from
this experiment's features. Its deterministic split preserves PatientID
groups and both classes for every finding in each partition. Missing patient
identity falls back to study grouping and records that patient-disjoint
qualification is unavailable.

The declared tree uses Priml `TrainLoop`, `TrainStep` over `Raptor` with an
`ImageNetEncoder` backbone, a path-built cold dataset,
and final-only probability-based ROC AUC. Its shared WallBudget receipt includes raw
rendering, model construction, training, the complete holdout and export.
The receipt records complete/failed status and elapsed seconds at the root;
RSNA hardware, split, steps and final metric evidence live under metadata.
Qualification requires complete status and metadata.qualified_hardware=true.
`train_exp001` forks this protocol with four studies per batch.
Public `exp000` remains released-checkpoint inference.

```bash
UV_NO_SYNC=1 uv --quiet run --frozen python -m priml priml.baselines.rsna_knee.experiments.train_exp000
```

Acquire the [ConvNeXt-tiny ImageNet-1K initializer](https://download.pytorch.org/models/convnext_tiny-983f1562.pth)
at `/opt/scratch/datasets/rsna-knee-imagenet/convnext_tiny-983f1562.pth`.
Its complete SHA-256 is checked before strict state loading.

The shared loop constructs the model before cold data preparation. The measured
job includes complete-series verification, uncached raw rendering, training,
every holdout study and export. It chooses one fluid-preferred series per plane
and four slices per plane, producing twelve normalized 128-pixel RGB views.
This is the separate cold recipe's rendering. Training has a 180-second limit and reserves
45 seconds for evaluation/export within a 300-second total target. CPU
fixture completion does not qualify GPU hardware or timing. Qualification
requires external launch-to-exit timing as well as the durable receipt.

## Acquire complete DICOM series

Place the acquisition-only selected training-series CSV at
`/opt/scratch/datasets/rsna-knee/selected-series.csv`, with official study and
series UID columns. `test_series.csv` selects every test acquisition.

```bash
UV_NO_SYNC=1 uv --quiet run --frozen python -m priml priml.baselines.rsna_knee.scripts.acquisition.acquire
```

Acquisition uses the separately installed, authenticated Kaggle CLI's
`auth print-access-token` command. Priml imports no Kaggle SDK. The token and
authenticated archive redirect stay in memory; the initial streaming request
reads no archive payload. Acquisition then uses ZIP64 HTTP ranges, verifies
full lengths and CRC32 values, records SHA-256 identities, and publishes files
only after every selected member verifies. Interrupted transfers reuse verified
files. Four requests run concurrently by default. Shared ensure_data verifies the full
SHA256 manifest and seals staging completion before any member is published. Credentials and signed URLs
stay in memory.

The receipt beneath `/opt/scratch/artifacts/rsna-knee/acquisition` binds the
resolved input root and both inventory manifests. The cold renderer rechecks
every selected complete series and member identity. A benchmark requires
`complete=true`; partial selected series invalidate timing inputs.

To bind a complete transferred inventory to another host, preserve its
original receipt/manifests beneath
`/opt/scratch/artifacts/rsna-knee/acquisition-transferred`, then run:

```bash
UV_NO_SYNC=1 uv --quiet run --frozen python -m priml priml.baselines.rsna_knee.scripts.adopt_receipt.adopt_receipt
```

This job verifies every original manifest digest, member length, CRC32,
SHA-256, and complete inventory total before recording destination identity.
