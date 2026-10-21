# ROGII wellbore geology data

Accept the competition rules linked in the [README](README.md)
and authenticate with Kaggle. Download and extract the competition archive,
then stage the complete paired CSVs once from the repository root:

```sh
kaggle competitions download -c rogii-wellbore-geology-prediction \
  -p /opt/scratch/caches/kaggle/rogii
unzip -n /opt/scratch/caches/kaggle/rogii/rogii-wellbore-geology-prediction.zip \
  -d /opt/scratch/caches/kaggle/rogii/extracted
UV_NO_SYNC=1 uv --quiet run --frozen python -m \
  priml.baselines.wellbore_geology.scripts.prepare_data \
  --source /opt/scratch/caches/kaggle/rogii/extracted \
  --directory /opt/scratch/datasets/rogii-ruby-fade6112
```

The stage copies paired train/test CSVs and the official sample submission,
excludes images, and writes the geographic centroid map. Download the pinned
ConvNeXt-small initialization and prepare particle caches using the commands
in the [README](README.md#prepare-inputs). Those jobs run once before an
experiment that requires the cache.

Tests write a few synthetic well CSV rows under `tmp_path` and call the staging
and reader functions directly. Pytest needs no Kaggle archive, downloaded
weights, or particle cache.
