# RSNA knee data

Accept the competition rules and use an authenticated Kaggle account. Put the
official `train.csv`, `test.csv`, `sample_submission.csv`, `train_series.csv`,
and `test_series.csv` in `/opt/scratch/datasets/rsna-knee/raw`.

Download the three pinned Kaggle datasets listed in the
[README](README.md#inputs-and-provenance) beneath `/opt/scratch/datasets/`.
Their checkpoints are verified by SHA-256 before loading.

For the cold training recipe, put a selected training-series CSV at
`/opt/scratch/datasets/rsna-knee/selected-series.csv`, then acquire complete
DICOM series once:

```sh
UV_NO_SYNC=1 uv --quiet run --frozen python -m priml \
  priml.baselines.rsna_knee.scripts.acquisition.acquire
```

This explicit job obtains a token from the authenticated Kaggle CLI, downloads
selected ZIP members, verifies lengths, CRC32, and SHA-256, and writes an
acquisition receipt. The [acquisition section](README.md#acquire-complete-dicom-series)
explains receipt transfer and revalidation. No test downloads DICOM or requests
Kaggle credentials. Tests construct small CSV/DICOM inputs in `tmp_path`; the
checked-in goldens are minted in-tree from this package.
