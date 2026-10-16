# NanoChat

Time-budgeted language-model training with n-gram memory, local attention,
layer pooling, and sparse embedding updates. Experiment factories expose the
model, optimizer, data, and evaluation settings in one configuration tree.

For our [(auto)²-research blog post](https://rekursiv.ai/blog/autoautoresearch/),
see [Reproduce](#reproduce) for the commands and [Results](#results) for the
measured checkpoints.

## Table of contents

- [Reproduce](#reproduce)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
  - [Experiments](#experiments)
  - [Parity checks against the reference](#parity-checks-against-the-reference)
  - [The base_dir knob](#the-base_dir-knob)
- [Results](#results)
  - [H200, 300 seconds](#h200-300-seconds)
  - [H200, 525 seconds](#h200-525-seconds)
  - [B200, 300 seconds](#b200-300-seconds)
  - [References](#references)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Files](#files)

## Reproduce

Every input below is either downloaded at a pinned revision and checked
against its SHA-256, or written by an earlier step here. Run the steps in
order from a checkout with its dependencies installed.

### Prerequisites

- **GPU.** `exp000` and `exp004`-`exp024` run FlashAttention-4, which needs
  an SM90 or SM100 GPU (H100, H200, B200). `exp001`-`exp003` run PyTorch
  attention on any CUDA GPU. `exp_smoke` runs on whatever device is present,
  CPU included. The recorded results use one H200, and one B200 for `exp024`.
- **FlashAttention-4.** Install the `flash-attn-4` package (Linux only) into
  the same environment, e.g. `uv add flash-attn-4`, if it is not there yet.
- **Network.** The data steps download Karpathy's ClimbMix shards from
  Hugging Face. No credentials are needed. W&B is optional: `exp000` and its
  forks report to the W&B project `nanochat` when you are logged in, and
  continue without it otherwise. Metrics always land in the run directory.
- **Disk.** Each ClimbMix shard is about 92 MB. The BPE corpus holds 8 shards
  (0.7 GB). The prepared corpus holds 15 raw shards, their donor-moved copy,
  and 1.3 GB of packed rows; budget 6 GB for it, and as much again for the
  ConvexTok corpus.

### Data

Three preparations, one line each. Each writes beneath
`/opt/scratch/datasets`, where the experiments read by default.

```bash
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --num-train-shards 7
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --stage all
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data priml.baselines.convextok.prepare.donor_convextok16k --stage all
```

| Step | Writes | Feeds |
|---|---|---|
| BPE corpus | `datasets/nanochat/shard_0000{0..7}.parquet`, `datasets/nanochat/tokenizer/` | `exp000`-`exp011`, `exp_smoke` |
| Prepared corpus | `datasets/nanochat/{raw,donor-original,sample,unigram16k,reference-bpe,reference-eval}/` | `exp012`-`exp022`, `exp024` |
| ConvexTok corpus | `datasets/nanochat-convextok/` in the same layout, `convextok16k/` in place of `unigram16k/` | `exp023` |

- The BPE corpus fetches shards 0-7 and fits an 8K byte-pair vocabulary on
  shards 0-6; shard 7 is held out for validation.
- The prepared corpus fetches shards 0-14 into `raw/` and fits the same BPE
  vocabulary there. It writes `donor-original/`, the training shards with 490
  documents moved forward, then fits a 16K Unigram vocabulary on a train-only
  sample, packs training and evaluation rows, and builds the byte-matched
  reference evaluation. Fitting never reads validation text.
- The ConvexTok corpus is the prepared recipe with ConvexTok's 16K vocabulary
  in the tokenizer slot, fitted on raw shards 0-6. It needs an 80 GB GPU and
  about 60 GiB of host memory; see `../convextok/README.md`.
- `--print-config` prints a recipe without preparing anything, and `--stage`
  runs one step of it: `fetch`, `baseline`, `corpus`, `sample`, `tokenizer`,
  `rows`, `reference`, or `verify`. Prepared outputs are never overwritten, so
  a rerun of a finished step needs a fresh `--directory`.

### Experiments

One launch per experiment. Each run writes `metrics.json` to
`/opt/scratch/runs/nanochat/<experiment>`; `exp_smoke` and `exp000`-`exp008`
also save `checkpoints/` there. The launch log prints the resolved
configuration.

```bash
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp_smoke
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp000
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp001
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp002
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp003
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp004
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp005
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp006
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp007
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp008
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp009
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp010
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp011
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp012
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp013
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp014
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp015
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp016
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp017
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp018
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp019
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp020
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp021
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp022
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp023
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp024
```

What each reads, and its recorded result. Paths are beneath
`/opt/scratch/datasets`. Every result is a mean BPB over the seeds named, at
the factory's own budget; lower is better.

| Experiment | Reads | Run after | Budget | Recorded result |
|---|---|---|---|---|
| `exp_smoke` | `nanochat/` shards 0-7, `tokenizer/` | BPE corpus | 4 steps | Installation check; no score |
| `exp000` | `nanochat/` shards 0-7, `tokenizer/` | BPE corpus | 300 s | H200: 1.005432 (seeds 42-44) |
| `exp001` | as `exp000` | BPE corpus | 300 s | H200: 1.056776 (42-44) |
| `exp002` | as `exp000` | BPE corpus | 300 s | H200: 1.074704 (42-44) |
| `exp003` | as `exp000` | BPE corpus | 300 s | H200: 1.030769 (42-44) |
| `exp004` | as `exp000` | BPE corpus | 525 s | H200: 0.972782 (42-44) |
| `exp005` | as `exp000` | BPE corpus | 525 s | H200: 0.961005 (42-44) |
| `exp006` | as `exp000` | BPE corpus | 525 s | H200: 0.957160 (42-44) |
| `exp007` | as `exp000` | BPE corpus | 525 s | H200: 0.949300 (42-44) |
| `exp008` | as `exp000` | BPE corpus | 525 s | H200: 0.938595 (42-44) |
| `exp009` | as `exp000` | BPE corpus | 525 s | H200: 0.934975 (42-44) |
| `exp010` | as `exp000` | BPE corpus | 525 s | H200: 0.934865 (42-44) |
| `exp011` | as `exp000` | BPE corpus | 525 s | H200: 0.927435 (42-44) |
| `exp012` | `nanochat/raw/` shards 0-14, `raw/tokenizer/` | Prepared corpus | 525 s | H200: 0.927194 (42-44) |
| `exp013` | as `exp012` | Prepared corpus | 525 s | H200: 0.910094 (42-44) |
| `exp014` | as `exp012` | Prepared corpus | 525 s | H200: 0.907300 (42-44) |
| `exp015` | as `exp012` | Prepared corpus | 525 s | H200: 0.906465 (42-44) |
| `exp016` | as `exp012` | Prepared corpus | 525 s | H200: 0.906208 (42-44) |
| `exp017` | as `exp012` | Prepared corpus | 525 s | H200: 0.903610 (42-44) |
| `exp018` | as `exp012` | Prepared corpus | 525 s | H200: 0.900618 (42-44) |
| `exp019` | `nanochat/donor-original/`, `raw/tokenizer/` | Prepared corpus | 525 s | H200: 0.897725 (42-44) |
| `exp020` | `nanochat/unigram16k/prepared/`, `reference-eval/unigram.npz` | Prepared corpus | 525 s | H200: 0.894827 (42-44) |
| `exp021` | as `exp020` | Prepared corpus | 525 s | H200: 0.888283 (42-44) |
| `exp022` | as `exp020` | Prepared corpus | 525 s | H200: 0.887457 (42-51) |
| `exp023` | `nanochat-convextok/convextok16k/prepared/`, `nanochat-convextok/reference-eval/unigram.npz` | ConvexTok corpus | 525 s | H200: 0.890578 (42-44) |
| `exp024` | as `exp020` | Prepared corpus | 300 s | B200: 0.887791 (42-51) |

- Each factory pins seed 42. For another seed of `exp004` onward, launch
  through the preparation, which binds the same inputs and reports to
  `metrics.json` only:

  ```bash
  uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --stage train --experiment exp022 --seed 43 --run-directory /opt/scratch/runs/nanochat/exp022-seed43
  ```

  It reads the prepared corpus's layout, so it needs `--stage all` first,
  and `--run-directory` must not exist yet.
- `exp000`-`exp015` were measured on FlashAttention-3, before `exp000`
  switched to FlashAttention-4; they now run FlashAttention-4.
- The [Results](#results) tables also list each recipe at the other budget.
  To reproduce those, change the one budget line the rungs inherit:
  `train_budget_sec` in `exp000` (300 s, inherited by `exp001`-`exp003`) or in
  `exp004` (525 s, inherited by `exp005`-`exp023`).

### Parity checks against the reference

Three scripts compare this package with Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) at commit
`b11d6f283f866eb7e10fb776a4b8553fef873fd5`. Each clones it at that commit into
`/opt/scratch/karpathy-autoresearch` and reads the BPE corpus.

```bash
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.karpathy_data_parity
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.karpathy_parity
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.karpathy_race
```

- `karpathy_data_parity` compares every packed token of both dataloaders.
- `karpathy_parity` steps both models side by side and compares every
  parameter and gradient with `torch.equal`.
- `karpathy_race` runs the reference's own `train.py` for a full budget, the
  counterpart of an `exp001` run.

### The base_dir knob

Every default above sits beneath `/opt/scratch`. To move everything, pick a
root and pass it in two places:

- **Experiments:** add `--override base_dir=/data/root` to a launch. It reads
  from `/data/root/datasets/...` and writes to `/data/root/runs/nanochat/...`.
- **Preparation:** add `--directory /data/root/datasets/nanochat` to the BPE
  and prepared steps, and `--directory /data/root/datasets/nanochat-convextok`
  to the ConvexTok step.

`--stage train` has no default output: its `--run-directory` names it, and its
inputs come from `--directory`.

## Results

These experiments checkpoint the major improvements from the research
campaigns described in [(auto)²-research: SoTA on Karpathy's NanoChat
Benchmark](https://rekursiv.ai/blog/autoautoresearch/). The code collects those
improvements into a cumulative sequence of runnable recipes. Each checkpoint
records the changes from its named parent; some combine several campaign
discoveries. The tables below report the measured checkpoints at each hardware
and training budget.

As explained in the blog post, we ran the majority of our experiments on H200s.
This is why you will see 525s instead of 300s time limits, which closely matches
the steps that you would get on a B200, but on an H-series GPU.

`exp004` starts from `exp000`; later factories inherit their predecessor.
`exp004`-`exp022` default to 525 seconds. To reproduce the final experiment,
`exp022`, on B200, run `exp024`, its 300-second fork.

Δ BPB compares each recipe with its parent at the same budget; negative is
better. `--` means no measured comparison is available.

### H200, 300 seconds

| Experiment | Change | Mean BPB | Δ BPB | Seeds |
|---|---|---:|---:|---|
| `exp000` | Eight-layer reference (measured on FlashAttention-3; now runs FlashAttention-4) | 1.005432 | -- | 3 (42-44) |
| `exp001` | PyTorch attention | 1.056776 | +0.051344 | 3 (42-44) |
| `exp002` | Remove value embeddings | 1.074704 | +0.017928 | 3 (42-44) |
| `exp003` | Remove local windows | 1.030769 | -0.043935 | 3 (42-44) |
| `exp004` | 525-second training budget | 1.002721 | -0.002711 | 3 (42-44) |
| `exp005` | Five wider layers and larger microbatches | 0.988665 | -0.014056 | 3 (42-44) |
| `exp006` | Input bigram embeddings | 0.969540 | -0.019125 | 3 (42-44) |
| `exp007` | Input trigram embeddings and separate optimizer groups | 0.960311 | -0.009229 | 3 (42-44) |
| `exp008` | Larger memory and adjusted learning-rate schedule | 0.956233 | -0.004078 | 3 (42-44) |
| `exp009` | Eight layers and native BF16 weights | 0.962389 | +0.006156 | 3 (42-44) |
| `exp010` | Output normalization and thresholded squared ReLU | 0.961917 | -0.000472 | 3 (42-44) |
| `exp011` | Residual/head gates and two-layer pooling | 0.955083 | -0.006834 | 3 (42-44) |
| `exp012` | Fourteen training shards, excluding validation | 0.955053 | -0.000030 | 3 (42-44) |
| `exp013` | Per-layer hashed n-gram values and optimizer schedules | 0.946772 | -0.008281 | 3 (42-44) |
| `exp014` | Nonuniform feed-forward expansion | 0.943577 | -0.003195 | 3 (42-44) |
| `exp015` | Narrower model and row-wise RMSProp | 0.935564 | -0.008013 | 3 (42-44) |
| `exp016` | FlashAttention-4 (now inherited from exp000) and CUDA graphs | 0.935105 | -0.000459 | 3 (42-44) |
| `exp017` | Fused QK/RoPE and n-gram accumulation | 0.931969 | -0.003136 | 3 (42-44) |
| `exp018` | Wider model and larger microbatches | 0.936150 | +0.004181 | 3 (42-44) |
| `exp019` | Reuse attention inputs and reorder training documents | 0.932429 | -0.003721 | 3 (42-44) |
| `exp020` | Prepared 16K Unigram and reference-byte evaluation | 0.930738 | -0.001691 | 3 (42-44) |
| `exp021` | Sparse table updates and bounded-logit loss | 0.923441 | -0.007297 | 3 (42-44) |
| `exp022` | Zero-initialize memory tables | 0.922382 | -0.001059 | 10 (42-51) |
| `exp023` | ConvexTok vocabulary | 0.925462 | -- | 3 (42-44) |

### H200, 525 seconds

| Experiment | Change | Mean BPB | Δ BPB | Seeds |
|---|---|---:|---:|---|
| `exp000` | Eight-layer reference (measured on FlashAttention-3; now runs FlashAttention-4) | 0.973908 | -- | 3 (42-44) |
| `exp001` | PyTorch attention | 1.011005 | +0.037097 | 3 (42-44) |
| `exp002` | Remove value embeddings | 1.028092 | +0.017087 | 3 (42-44) |
| `exp003` | Remove local windows | 0.995873 | -0.032219 | 3 (42-44) |
| `exp004` | 525-second training budget | 0.972782 | -0.001126 | 3 (42-44) |
| `exp005` | Five wider layers and larger microbatches | 0.961005 | -0.011777 | 3 (42-44) |
| `exp006` | Input bigram embeddings | 0.957160 | -0.003845 | 3 (42-44) |
| `exp007` | Input trigram embeddings and separate optimizer groups | 0.949300 | -0.007860 | 3 (42-44) |
| `exp008` | Larger memory and adjusted learning-rate schedule | 0.938595 | -0.010705 | 3 (42-44) |
| `exp009` | Eight layers and native BF16 weights | 0.934975 | -0.003620 | 3 (42-44) |
| `exp010` | Output normalization and thresholded squared ReLU | 0.934865 | -0.000110 | 3 (42-44) |
| `exp011` | Residual/head gates and two-layer pooling | 0.927435 | -0.007430 | 3 (42-44) |
| `exp012` | Fourteen training shards, excluding validation | 0.927194 | -0.000241 | 3 (42-44) |
| `exp013` | Per-layer hashed n-gram values and optimizer schedules | 0.910094 | -0.017100 | 3 (42-44) |
| `exp014` | Nonuniform feed-forward expansion | 0.907300 | -0.002794 | 3 (42-44) |
| `exp015` | Narrower model and row-wise RMSProp | 0.906465 | -0.000835 | 3 (42-44) |
| `exp016` | FlashAttention-4 (now inherited from exp000) and CUDA graphs | 0.906208 | -0.000257 | 3 (42-44) |
| `exp017` | Fused QK/RoPE and n-gram accumulation | 0.903610 | -0.002598 | 3 (42-44) |
| `exp018` | Wider model and larger microbatches | 0.900618 | -0.002992 | 3 (42-44) |
| `exp019` | Reuse attention inputs and reorder training documents | 0.897725 | -0.002893 | 3 (42-44) |
| `exp020` | Prepared 16K Unigram and reference-byte evaluation | 0.894827 | -0.002898 | 3 (42-44) |
| `exp021` | Sparse table updates and bounded-logit loss | 0.888283 | -0.006544 | 3 (42-44) |
| `exp022` | Zero-initialize memory tables | 0.887457 | -0.000826 | 10 (42-51) |
| `exp023` | ConvexTok vocabulary | 0.890578 | -- | 3 (42-44) |

`exp023` has no Δ against `exp022`'s ten-seed mean. Its `exp022` controls,
run on the same node and code, scored 0.888754 at 525 s and 0.923618 at
300 s.

### B200, 300 seconds

| Experiment | Change | Mean BPB | Δ BPB | Seeds |
|---|---|---:|---:|---|
| `exp024` | `exp022` at a 300-second budget | 0.887791 | -- | 10 (42-51) |

### References

- Karpathy. [autoresearch](https://github.com/karpathy/autoresearch), commit
  `b11d6f283f866eb7e10fb776a4b8553fef873fd5`.
- So et al. [Primer: Searching for Efficient Transformer for Language
  Modeling](https://arxiv.org/abs/2109.08668).
- Jordan et al. 2024. [Muon: an optimizer for hidden
  layers](https://kellerjordan.github.io/posts/muon/).
- Beltagy et al. [Longformer: The Long-Document
  Transformer](https://arxiv.org/abs/2004.05150).
- Zhou et al. [Value Residual Learning](https://arxiv.org/abs/2410.17897).
- Tempus et al. [Tokenisation via Convex
  Relaxations](https://arxiv.org/abs/2605.22821).

## Evaluation

BPE experiments use the base evaluator. Unigram experiments replay the rows
selected by the BPE reference tokenizer, preserving their source bytes and
document boundaries. Padding and boundary tokens are excluded from scoring.
A replay longer than the model context is never truncated: it continues in
extra windows, each scoring only targets not yet scored, with context from its
own row. The 16K Unigram rows all fit, so their archive has no such windows.

`bpb` uses the reference evaluator's decoded-token byte denominator.
`literal_bpb` uses the literal UTF-8 byte count. These denominators can differ
for tokens that contain partial UTF-8 characters; compare like with like.
Matching bytes does not make the token-level prediction tasks identical.

The 16K experiments select ATen matrix multiplication inside the compiled
model. This avoids large-matrix indexing errors observed with PyTorch 2.11 and
Triton 3.6 on H200, including silent zero logits during evaluation. CUDA graphs
and the fused attention and n-gram kernels remain enabled.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/nanochat
uv --quiet run --frozen pytest priml/baselines/nanochat/train_step_test.py::test_exp022_two_updates_match_portable_golden -m compute_training
uv --quiet run --frozen pytest priml/baselines/nanochat -m gpu_torch_cuda
```

The first command runs the fast CPU tests. Full configuration goldens run with
`-m compute_large_fixture`. The second runs the reduced-size, two-update
model/optimizer golden using portable kernels; it is excluded from the fast
tier. Native GPU tests check fused outputs and gradients separately. The
portable golden does not certify CUDA numerical identity or a full training
score.

## Files

| File | Responsibility |
|---|---|
| `experiments.py` | Experiment factories and budgeted training loop |
| `model.py`, `attention.py`, `ngram.py` | Model, gates, memory, and fused kernels |
| `train_step.py` | Training, losses, and reference metric |
| `optimizers.py` | Sparse updates and schedules |
| `data.py` | Online packing, prepared rows, and reference replay |
| `scripts/prepare_data.py` | Pinned recipe, corpus, packing, verification, and launch entry |
| `scripts/prepare_tokenizer.py` | Fitting samples and byte-level tokenizers |
| `scripts/karpathy_*.py` | Parity checks against the reference implementation |
