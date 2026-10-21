# Motor imagery EEG data

Accept the competition rules linked in the [README](README.md)
and authenticate with Kaggle. From the repository root, stage the archive once:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen python -m \
  priml.baselines.motor_imagery.scripts.prepare_data
```

If the authorized archive is already downloaded, supply `--archive` and a fresh
`--directory` instead. The stage contains raw CSVs under `Training set/` and
`Cross participant test set/`, plus `manifest.json` with each file's SHA-256.
The default download cache is `/opt/scratch/caches/kaggle/motor-imagery`.
Point the experiment's `validation.dataset.working_dir` at the staged directory;
the [README](README.md#stage-and-run) gives the run commands. Staging does not
filter EEG or train a model.

Tests create a ZIP with one tiny training CSV and one tiny test CSV under
`tmp_path`, then call `stage_archive` directly. Ingestion and filter tests use
small generated arrays; only the model and step goldens are checked-in `.pt`
records. They need no Kaggle account or full competition archive.
