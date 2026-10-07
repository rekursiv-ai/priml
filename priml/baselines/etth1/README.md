# DLinear on ETTh1

This is a starting point for forecasting experiments in PRIML. It uses DLinear
with the past 336 hours of ETTh1 data to predict the next 96 hours across all
seven variables.

`exp000` follows the reference setup: batch size 8, Adam at `1e-4`, seed 2021,
and a limit of 10 epochs / 10,260 updates. It stops early after three epochs
without matching or improving the best validation score. The full settings
are in `experiments.py`.

## Run locally

Run these commands from the PRIML repository. Data and checkpoints go under
`/opt/scratch`, the default resource root. To use another root, override
`base_dir` when training and use the matching paths for preparation and
evaluation. Use a fresh run directory if the checkpoint path contains an older run.

Install the dependencies and download the data:

```sh
uv sync --frozen
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.prepare_data \
  --directory /opt/scratch/datasets/etth1
```

Train the baseline:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml \
  priml.baselines.etth1.experiments.exp000
```

Evaluate the checkpoint with the best validation score:

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate \
  --directory /opt/scratch/datasets/etth1 \
  --checkpoint /opt/scratch/runs/etth1/exp000/checkpoints
```

The download script checks the file's checksum. If you already have the CSV,
add `--source /path/to/ETTh1.csv` to the preparation command. If downloading
fails with a certificate error on macOS, try adding
`SSL_CERT_FILE=/etc/ssl/cert.pem` before that command.

Normalization uses only the training data. The split and batch ordering match
the reference, including dropping incomplete batches. Checkpoints save the
state needed to resume training. Older checkpoints without that state can
still be evaluated, but cannot be resumed.

## Tests and reference check

Run the baseline tests:

```sh
uv --quiet run --frozen pytest priml/baselines/etth1 -o addopts= -q
```

These tests use small fixtures, so they do not need the downloaded dataset.
They cover data loading, training, checkpoint resume, and saved reference
outputs (called goldens) that catch changes to the results.

The reference is [ql-denoising/DLinear](https://github.com/ql-denoising/DLinear)
at commit `da9e67442b95af76488b8e4e1806cc3185723dd9`. To compare against a local
checkout, run the following with its pandas, scikit-learn, and matplotlib
dependencies available:

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.verify_reference \
  --reference /path/to/DLinear --directory /opt/scratch/datasets/etth1
```

This checks the source files and compares small test runs plus three training
updates on the real data, bit for bit. It allows the NumPy 2 compatibility
change from `np.Inf` to `np.inf` in `utils/tools.py`.

The goldens were recorded from the reference after the comparison passed.
To deliberately regenerate them, use pytest so PRIML's numerical settings
are loaded before the source capture:

```sh
uv --quiet run --frozen pytest \
  priml/baselines/etth1/scripts/mint_reference_test.py -o addopts= \
  --etth1-reference /path/to/DLinear \
  --etth1-directory /opt/scratch/datasets/etth1
```

Adding `--mint` to the verifier command above runs the same pytest step.
Source details are in `testdata/source.json`.

## Local results

The CPU run on 2026-10-02 stopped after 5 epochs / 5,130 updates. The best
checkpoint was step 2,052:

| Metric | Value |
| --- | --- |
| Validation MSE | 0.6461290121 |
| Test MSE | 0.3748246133 |
| Test MAE | 0.3994735181 |

A full run of the reference matched the validation losses, saved parameters,
final Torch RNG state, and every test prediction exactly. The comparison is
recorded in `results/local_cpu.json`.

These results were measured on Apple Silicon with Python 3.12.3, PyTorch 2.11.0,
NumPy 2.5.3, and one Torch thread. GPU/reference-hardware results are still TBD.

For the next experiment, derive `exp001()` from `exp000()` and change one thing.
Keep the data split, scoring, seed, and maximum compute budget fixed so results
can be compared. Use validation scores to choose changes; leave the test split
for final evaluation.
