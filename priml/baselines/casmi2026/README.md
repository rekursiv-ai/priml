# CASMI 2026

1. Public `experiments.exp000` reproduces released-checkpoint inference.
2. `experiments.train_exp000` declares a separate five-minute training protocol.
3. Scientific preparation runs explicitly outside the project environment.
4. Goldens minted in-tree pin the `exp000` networks and FPNet training steps.

## Competition

[Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)
identifies molecules from tandem mass spectra. The metric is MRR@25. Entry and
team formation close December 7, 2026; final submissions close December 14,
2026 at 23:59 UTC. This is a notebook code competition with a nine-hour runtime
limit. Downloading competition data requires accepted rules and authenticated
Kaggle access. Running this package never submits a leaderboard entry.

## Layout

| Role | Defining modules |
| --- | --- |
| Shared base modules | `custom_types`, `scores`, `boosting`, `model` (every network that is not a promoted architecture) |
| Main and independent second ranking | `main_engine`, `second_engine` |
| Numerical molecular tables and reaction declarations | `chemistry` |
| External mass-store screening | `pubchem` |
| DREAMS tails and fragment network | `dreams`, `fragnet` |
| Simulator data, scheduling and backends (their networks are in `model`) | `simulators` |
| Fusion policy and complete inference controller | `fusion`, `recipe` |
| Offline generation and cache construction | `scripts` |
| New optimization protocol | `train_step`, `data`, `metric`, `experiments.train_exp000` |
| Test helpers and fixture objectives (not shipped in the wheel) | `testing` |

Runtime inference and training consume Arrow/NumPy tables without RDKit,
pandas, scikit-learn or original implementations. Missing prepared rows raise;
explicit rejected rows retain their recorded outcome. Native histogram boosting
remains local because scikit-learn is absent from the public Priml dependencies.

## Acquire and prepare data

[DATA.md](DATA.md) gives the one-time download, environment, and preparation
steps. Ordinary tests use tiny local inputs and never provision RDKit.

[assets.tsv](assets.tsv) pins seventeen public archive versions and complete
SHA256 digests. Extract them beneath
`/opt/scratch/datasets/casmi2026/inputs/datasets/<owner>/<slug>/`.
Competition files belong beneath
`/opt/scratch/datasets/casmi2026/inputs/enveda-CASMI26-molecule-id-mass-spectra/`.
Place each downloaded archive as `<owner>/<slug>/<slug>.zip` beneath
`/opt/scratch/datasets/casmi2026/archives/`, then verify them against the manifest
before extracting:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen -m priml \
  priml.baselines.casmi2026.scripts.acquire_assets.acquire_assets
```

The job names the first absent archive and moves a corrupt one aside; a verified
directory carries a completion marker.

The manual commands in [DATA.md](DATA.md) create isolated Python 3.14
environments under
`/opt/scratch/caches/rdkit/casmi-preparation/`: RDKit 2026.03.3 for molecular
identities/generation and RDKit 2025.03.6 for simulator graphs. Both use NumPy
2.5.3. Ordinary pytest does not create or run these interpreters.

The independent generator is the existing `scripts` executable:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen --no-project --no-sync \
  --python /opt/scratch/caches/rdkit/casmi-preparation/2026.03.3/bin/python \
  -m priml.baselines.casmi2026.scripts --help
```

Its typed CLI has one subcommand per asset family (`structures`, `admission`,
`ice`, `glacier`, `reactions`, ...; `--help` lists them), each taking its input
table and `--output`; the simulator families default `--output` to the directory
their reader uses. Generation calls standard
RDKit primitives and never imports a notebook or original recipe. Outputs must
be new scratch directories. Every manifest records scientific versions, full
input digests and generator digests; invalid and duplicate rows remain explicit.

Prepare complete data in this order:

1. Generate training-only identities from the supplied training structures.
2. Build the spectral cache with `data.prepare_data`.
3. Generate second-engine fingerprints, pool, library and fragments.
4. Run `scripts.recipe_preparation.prepare_requests` for all public candidates.
5. Generate reaction results, then run `prepare_raw_products`.
6. Generate product identities, then run `prepare_product_requests`.
7. Generate final identities, engine identities, PubChem rows and analog families.
8. Generate main fragments, intrinsic descriptors, Fragnet and derivation rows.
9. Generate fragment-gap graphs and both simulators' admission/root tables.

| Prepared family | Complete stored boundary |
| --- | --- |
| Molecular identities | Raw/standardized key, formula, mass, property and fingerprint decisions; official stereo-free identity |
| Fingerprints | Packed complete 16,551-bit scientific rows; each model selects its declared bits |
| Reactions | Ordered requests, products, first plans, every application and rejection; 46-rule and 71-rule declarations remain distinct |
| Analog/Fragnet | Ordered atom/bond observations, masses, capabilities, scaffold and mobile-H records |
| Derivation | Every terminal path, reacting-site environments, raw/standardized identity and application provenance |
| Main fragments | Independent complete F32 depth-one/two vectors and pre-cast F64 MAIN94 masses |
| Second fragments | Distinct unrounded F64 vectors and its own admission/fallback policy |
| Simulator admission | Distinct ICE/GLACIER decisions, canonical strings and explicit rejection reasons |
| ICE roots | Canonical 32-channel atoms, bonds, isotope-aware masses and complete formulas |
| GLACIER roots | Complete atom/bond features, shortest-path/random-walk inputs and physical fields |
| Fragment gap | Ordered atom masses, bonds and component counts |

The runtime reaction vocabulary and `v6_reactions.tsv` live in `chemistry`.
Preparation/cache builders stay in `scripts`. Tests use small synthetic rows;
complete generator execution is a manual data step.

## Run public inference

Run from a gated checkout after preparing all declared assets:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen -m priml \
  priml.baselines.casmi2026.experiments.exp000
```

The printed recipe retains Full1, A/B/D views, all five main feature families,
four main rankers, twelve independently fitted second rankers, the V3 PubChem
bank, optional DREAMS tails (enabled only when `dreams_np_mass.npy` exists under
`inputs/datasets/hengck23/hengck23-dreams-enveda-casmi26`; `dreams.json` in the
output records whether they were used) and engine-two rank-two insertion. ICE and GLACIER
retain their separate 5,400/4,000-second schedules, 96/32 prediction batches and
3,000-job chunks. Legacy, control and final fusion policies are separate printed
configuration fields. Each resource closes before the next stage allocates.

The output directory must be new and disjoint from every protected input.
Outputs include configuration, complete score/health maps, intermediate lists,
full audits and the submission in original sample order. Failed stages preserve
their records and prevent successful publication. Full native GPU scoring and
the historical public score of 0.443 remain unmeasured.

## Run the independent training chain

```sh
UV_NO_SYNC=1 uv --quiet run --frozen -m priml \
  priml.baselines.casmi2026.experiments.train_exp000
```

This new protocol trains width-128/two-block FPNet from scratch with AdamW
1e-5 and per-bit BCE. Its real Priml TrainLoop owns dataset, optimization, final
metrics and checkpoints. The seed remains unset. `experiments.train_exp001`
forks this parent and changes only the peak cap from 160 to 128;
`experiments.train_exp002` forks it and charges cold cache preparation to the
shared wall budget.

The shared 300-second wall budget includes cold preparation/verification,
construction, training, complete evaluation, export and durable receipt
publication. Training/evaluation targets are 240/45 seconds. Use
`experiments.train_exp002` when preparing data inside that budget. Official
molecular identities define disjoint splits; evaluation scores every candidate
in the 10 ppm window and retains unusable queries in its denominator. `budget.json` records completion/failure, elapsed
wall time and local preparation/step metadata. An over-budget completion raises.

Native checkpoints retain optimizer/RNG/cursor state. `fpnet.pt` exports the
actual evaluated EMA view and complete model configuration. Historical released
checkpoint-training recipes were unavailable. The first cold H200 attempt spent
300.76 seconds indexing identities, completed zero updates and exported nothing;
it supplies no completed MRR measurement.

## Source provenance and proof limits

The reference notebook is [Antriksh Arya, kernel 138524934 version 1](https://www.kaggle.com/code/antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert),
licensed Apache 2.0. Its complete SHA256 is
`81ebe7b6972bdebdf3d1227d77565e35a71b4b3418b155e3035ccaaff9d53005`.
The independent FPNet reference SHA256 is
`0ea6e372d1f045a877d350234c2ccf5d93fdd98cb23bd4c22376bf7e99547608`.
The V4b version-four manifest SHA256 is
`e64b960037769971e146a35932e29ed3f042de1716e4dbe281ebd84a7cfc3b6a`;
all 64 declared files matched. Historical mounted archive versions were not
identified; the seventeen downloadable archive versions are pinned separately.

Attribution remains with canqiang, flexonafft, obstacledeveloper, bobthebot369,
skomuro, prvsiyan, megayak, Dmitrii Gluzdov, hengck23, ms-pred,
Huseyin Emre Aksoy and Ahmed Berat Özer. Public asset terms remain applicable.
In particular, the Prvsiyan fingerprint dataset carries CC-BY-NC-SA-4.0; the
notebook license does not relicense data or checkpoints. No original source or
additional license file ships in this package.

`testdata/` holds the printed `exp000` and `train_exp000` configs, three
`train_step` updates of the tiny FPNet in `fpnet.pt`, and one initialization
and forward golden for each other network `exp000` builds: DreamsFP,
SecondFPNet, Glacier and the ICE generator and intensity models. Each PT golden
is minted by its test under `pytest --regenerate-b4b` with size-only reductions
inside `host_agnostic_numerics`; `fpnet.pt` is keyed by native state names and
the network goldens by released checkpoint names. Float64 outputs are rounded
once to float32 values and keep their float64 dtype. Every other stage is
checked on small literal inputs against hand-derived values and structural
identities. Released checkpoint names map bijectively to native state names,
with collision, missing, unexpected, shape and dtype validation before loading.

Tiny component and stage checks and the complete native entry run qualify their
declared boundaries. They do not establish complete production-cache generation,
full hidden-workload GPU parity, historical checkpoint training or leaderboard
score reproduction. Source timing examples used different workloads: twelve
molecules on two T4s took 20m6s; a 400-molecule H200 run took 2,859.85s. Python
set order changed one molecule's tied merged-adduct ordering in another run.
These observations are limitations of source replay, not new training results.
