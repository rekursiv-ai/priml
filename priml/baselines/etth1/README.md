# DLinear on ETTh1

This is a starting point for forecasting experiments in PRIML. It uses DLinear
with the past 336 hours of ETTh1 data to predict the next 96 hours across all
seven variables.

`exp000` follows the reference setup: batch size 8, Adam at `1e-4`, seed 2021,
and a limit of 10 epochs / 10,260 updates. It stops early after three epochs
without matching or improving the best validation score. The full settings
are in `experiments.py`.

## Reproduce

Run every command from the root of a fresh clone. `uv` installs the locked
dependencies on first use.

### Prerequisites

- Hardware: a CPU. `exp000` pins its runtime to the CPU; no GPU is used.
- Credentials: none; ETTh1 is a public download.
- Disk: under 10 MB for the CSV and `exp000`'s checkpoints.

### Data

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.prepare_data
```

Downloads `ETTh1.csv` from the ETDataset repository at a pinned commit, checks
its SHA-256, and writes it to `/opt/scratch/datasets/etth1`. An existing file
is verified instead. If you already have the CSV, add
`--source /path/to/ETTh1.csv`; it must pass the same check. If downloading
fails with a certificate error on macOS, prefix the command with
`SSL_CERT_FILE=/etc/ssl/cert.pem`.

### Experiments

```sh
uv --quiet run --frozen python -m priml priml.baselines.etth1.experiments.exp_smoke
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE uv --quiet run --frozen python -m priml priml.baselines.etth1.experiments.exp000
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate --checkpoint /opt/scratch/runs/etth1/exp000/checkpoints
```

The thread and MKL settings on `exp000` are the ones the recorded result used;
they fix the floating-point reduction order. `evaluate` scores the checkpoint
with the best validation score on the held-out test split; it reads the
`best.json` that `exp000` writes, so it runs after `exp000`. Use a fresh run
directory if `exp000`'s checkpoint path holds an older run.

| Step | Reads | Run after | Output | Result | Hardware |
|---|---|---|---|---|---|
| `exp_smoke` | `datasets/etth1` | Data | Log only | Integration check only | CPU |
| `exp000` | `datasets/etth1` | Data | `runs/etth1/exp000/checkpoints` | Validation MSE 0.6461290121 | Apple Silicon and x86-64 CPUs |
| `evaluate` | `datasets/etth1`, `runs/etth1/exp000/checkpoints` | `exp000` | Printed JSON | Test MSE 0.3748246133, MAE 0.3994735181 | Apple Silicon and x86-64 CPUs |

Paths in the table are beneath `base_dir`. Normalization uses only the
training data. The split and batch ordering match the reference, including
dropping incomplete batches. Checkpoints save the state needed to resume
training. Older checkpoints without that state can still be evaluated, but
cannot be resumed.

### The `base_dir` knob

`base_dir` defaults to `/opt/scratch`. Data and run outputs resolve beneath it,
so one override relocates a whole run; point the Data step and the evaluation
at the same root:

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.prepare_data --directory /data/priml/datasets/etth1
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE uv --quiet run --frozen python -m priml priml.baselines.etth1.experiments.exp000 --override base_dir=/data/priml
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate --directory /data/priml/datasets/etth1 --checkpoint /data/priml/runs/etth1/exp000/checkpoints
```

## Tests and reference check

Run the baseline tests:

```sh
uv --quiet run --frozen pytest priml/baselines/etth1 -o addopts= -q
```

These tests use small fixtures, so they do not need the downloaded dataset.
They cover data loading, training, checkpoint resume, and saved reference
outputs (called goldens) that catch changes to the results.

The reference is [ql-denoising/DLinear](https://github.com/ql-denoising/DLinear)
at commit `da9e67442b95af76488b8e4e1806cc3185723dd9`. Check out that commit
next to this repository and apply the NumPy 2 spelling of `np.Inf`, the one
edit the verifier allows:

```sh
git clone https://github.com/ql-denoising/DLinear ../DLinear
git -C ../DLinear checkout da9e67442b95af76488b8e4e1806cc3185723dd9
sed -i.orig 's/np\.Inf/np.inf/g' ../DLinear/utils/tools.py
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.verify_reference --reference ../DLinear --directory /opt/scratch/datasets/etth1
```

The verifier runs the reference's own pandas, scikit-learn, and matplotlib
code, so those must be importable in the environment. It checks the source
files and compares small test runs plus three training updates on the real
data, bit for bit.

The goldens were recorded from the reference after the comparison passed.
To deliberately regenerate them, use pytest so PRIML's numerical settings
are loaded before the source capture:

```sh
uv --quiet run --frozen pytest priml/baselines/etth1/scripts/mint_reference_test.py -o addopts= --etth1-reference ../DLinear --etth1-directory /opt/scratch/datasets/etth1
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
NumPy 2.5.3, and one Torch thread. A run of the Reproduce steps from a fresh
clone on 2026-10-10, on an x86-64 Linux CPU with Python 3.14.7, PyTorch 2.11.0,
and NumPy 2.5.3, gave the same three values to every printed digit and the
same best step. GPU/reference-hardware results are still TBD.

For the next experiment, derive `exp001()` from `exp000()` and change one thing.
Keep the data split, scoring, seed, and maximum compute budget fixed so results
can be compared. Use validation scores to choose changes; leave the test split
for final evaluation.
