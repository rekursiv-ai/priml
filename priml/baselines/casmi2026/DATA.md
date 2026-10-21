# CASMI 2026 data

Run these preparation steps once from the repository root. Training and inference
reuse the prepared files under `/opt/scratch/datasets/casmi2026`; pytest uses
small local fixtures and does not download data or install RDKit.

Use `/opt/scratch/datasets/casmi2026` as the data root:

1. Accept the competition rules linked in the [README](README.md) and
   authenticate with Kaggle. Download its files into
   `inputs/enveda-CASMI26-molecule-id-mass-spectra/`:

   ```sh
   CASMI_ROOT=/opt/scratch/datasets/casmi2026
   COMPETITION="$CASMI_ROOT/inputs/enveda-CASMI26-molecule-id-mass-spectra"
   kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra \
     -p "$COMPETITION"
   unzip -n "$COMPETITION/enveda-CASMI26-molecule-id-mass-spectra.zip" \
     -d "$COMPETITION"
   ```

2. For each row in [assets.tsv](assets.tsv), download the listed version from
   `https://www.kaggle.com/datasets/<owner>/<slug>/versions/<version>` and
   save its ZIP as `archives/<owner>/<slug>/<slug>.zip`. The Kaggle CLI
   `datasets download` command has no version option, so use the version page.
3. Run the digest check below. It names the first missing or corrupt archive.
   Extract verified ZIPs under `inputs/datasets/<owner>/<slug>/`, preserving
   their internal paths.

```sh
UV_NO_SYNC=1 uv --quiet run --frozen -m priml \
  priml.baselines.casmi2026.scripts.acquire_assets.acquire_assets
```

The scientific generators need two RDKit releases. Create their environments
explicitly; ordinary tests do not create them. `treekle` supplies the independent
generator's serialization functions.

```sh
IDENTITY=/opt/scratch/caches/rdkit/casmi-preparation/2026.03.3
SIMULATOR=/opt/scratch/caches/rdkit/casmi-preparation/2025.03.6
uv --quiet venv --python 3.14 "$IDENTITY"
uv --quiet pip install --python "$IDENTITY/bin/python" \
  'rdkit>=2026.03.3,<2026.03.4' 'numpy>=2.5.3,<2.5.4' \
  'pyarrow>=25.0.1,<25.0.2' 'wrapt>=2,<3' 'treekle>=0.2.0'
uv --quiet venv --python 3.14 "$SIMULATOR"
uv --quiet pip install --python "$SIMULATOR/bin/python" \
  'rdkit>=2025.03.6,<2025.03.7' 'numpy>=2.5.3,<2.5.4' \
  'pyarrow>=25.0.1,<25.0.2' 'wrapt>=2,<3' 'treekle>=0.2.0'
```

The [README](README.md#acquire-and-prepare-data) lists the preparation order.
It does not define a single full-dataset command: each generator needs a source
table from the downloaded archives. Inspect each family before running its
subcommand; its help lists the required arguments:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen --no-project --no-sync \
  --python /opt/scratch/caches/rdkit/casmi-preparation/2026.03.3/bin/python \
  -m priml.baselines.casmi2026.scripts structures --help
```

Use the 2025.03.6 interpreter for `graphs`, `admission`, `ice`, and `glacier`;
the remaining scientific families use 2026.03.3. Each command writes a new
scratch output directory with input and generator digests. Build the spectral
cache through `data.prepare_data` after the training identities are available.

Tests create a few Arrow/CSV rows in `tmp_path` and check readers and
preparation functions directly against hand-derived values. They do not
validate a complete production data build; check each generated manifest when
preparing the full dataset.
