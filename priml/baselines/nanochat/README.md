# NanoChat

Time-budgeted language-model training with n-gram memory, local attention,
layer pooling, and sparse embedding updates. Experiment factories expose the
model, optimizer, data, and evaluation settings in one configuration tree.

For our [(auto)²-research blog post](https://rekursiv.ai/blog/autoautoresearch/),
see [Reproducing the blog post](#reproducing-the-blog-post) for experiment
checkpoints, results, and hardware guidance.

## Table of contents

- [Run the portable baseline](#run-the-portable-baseline)
- [Run Unigram tokenizer experiments (exp020-exp022)](#run-unigram-tokenizer-experiments-exp020-exp022)
- [Reproducing the blog post](#reproducing-the-blog-post)
  - [H200, 300 seconds](#h200-300-seconds)
  - [H200, 525 seconds](#h200-525-seconds)
  - [B200, 300 seconds](#b200-300-seconds)
  - [References](#references)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Files](#files)

## Run the portable baseline

From a Priml checkout with dependencies installed:

```bash
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --num-train-shards 7
uv --quiet run --frozen python -m priml priml.baselines.nanochat.experiments.exp001
```

This prepares an 8K BPE vocabulary and parquet shards under
`/opt/scratch/datasets/nanochat`. `exp001` uses PyTorch attention. The test-only
`exp_smoke` uses a reduced vocabulary and synthetic fixtures; it does not accept
these prepared 8K BPE inputs unchanged.

## Run Unigram tokenizer experiments (exp020-exp022)

`exp022` uses FlashAttention-4, Triton, BF16 weights, and prepared 16K Unigram
rows. It targets a single high-memory NVIDIA GPU. Its default training budget is
525 seconds for H-series GPUs, excluding compilation warmup and evaluation.
For B200, uncomment the marked 300-second budget line in `exp022`; it updates
both the training schedule and the loop's stop time.

The 16K experiments select ATen matrix multiplication inside the compiled
model. This avoids large-matrix indexing errors observed with PyTorch 2.11
and Triton 3.6 on H200, including silent zero logits during evaluation.
CUDA graphs and the fused attention and n-gram kernels remain enabled.

Install FlashAttention-4 into the same environment as Priml:

```bash
uv add flash-attn-4==4.0.0b29
```

Inspect the preparation recipe, then build its inputs in a fresh directory:

```bash
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --print-config
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --directory /opt/scratch/datasets/nanochat-unigram --stage all
```

Preparation downloads training and held-out shards at the configured revision, moves the
selected training documents, fits the tokenizer, and writes token arrays.
Tokenizer fitting excludes validation data. Existing prepared output
directories are not overwritten; individual stages can be selected with
`--stage`.

Launch with the prepared paths bound to the config:

```bash
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data --directory /opt/scratch/datasets/nanochat-unigram --stage train --experiment exp022 --seed 42 --run-directory /opt/scratch/runs/nanochat-exp022-42
```

Use a new run directory for each seed. Metrics and the resolved configuration
are saved there. Add `--save-checkpoint` to retain the final model state.
Prepared training stops on exhaustion; it never wraps or falls back to online
tokenization.

## Reproducing the blog post

These experiments checkpoint the major improvements from the research
campaigns described in [(auto)²-research: SoTA on Karpathy's NanoChat
Benchmark](https://rekursiv.ai/blog/autoautoresearch/). The code collects those
improvements into a cumulative sequence of runnable recipes. Each checkpoint
records the changes from its named parent; some combine several campaign
discoveries. The tables below report the measured checkpoints at each hardware
and training budget.

As explained in the blog post, we ran the majority of our experiments on H200s.
This is why you will see 525s instead of 300s time limits, which closely matches
the steps that you would get on a B200, but on on an H-series GPU.

`exp004` starts from `exp000`; later factories inherit their predecessor.
`exp004`–`exp022` default to 525 seconds. To reproduce the final experiment, `exp022`,
on b200, override with `--override step.train_budget_sec=300.0 --override max_time=300.0`.

Δ BPB compares each recipe with its parent at the same budget; negative is
better. `—` means no measured comparison is available.

### H200, 300 seconds

| Experiment | Change | Mean BPB | Δ BPB | Seeds |
|---|---|---:|---:|---|
| `exp000` | Eight-layer reference with FlashAttention-3 | 1.005432 | — | 3 (42–44) |
| `exp001` | PyTorch attention | 1.056776 | +0.051344 | 3 (42–44) |
| `exp002` | Remove value embeddings | 1.074704 | +0.017928 | 3 (42–44) |
| `exp003` | Remove local windows | 1.030769 | -0.043935 | 3 (42–44) |
| `exp004` | 525-second training budget | 1.002721 | -0.002711 | 3 (42–44) |
| `exp005` | Five wider layers and larger microbatches | 0.988665 | -0.014056 | 3 (42–44) |
| `exp006` | Input bigram embeddings | 0.969540 | -0.019125 | 3 (42–44) |
| `exp007` | Input trigram embeddings and separate optimizer groups | 0.960311 | -0.009229 | 3 (42–44) |
| `exp008` | Larger memory and adjusted learning-rate schedule | 0.956233 | -0.004078 | 3 (42–44) |
| `exp009` | Eight layers and native BF16 weights | 0.962389 | +0.006156 | 3 (42–44) |
| `exp010` | Output normalization and thresholded squared ReLU | 0.961917 | -0.000472 | 3 (42–44) |
| `exp011` | Residual/head gates and two-layer pooling | 0.955083 | -0.006834 | 3 (42–44) |
| `exp012` | Fourteen training shards, excluding validation | 0.955053 | -0.000030 | 3 (42–44) |
| `exp013` | Per-layer hashed n-gram values and optimizer schedules | 0.946772 | -0.008281 | 3 (42–44) |
| `exp014` | Nonuniform feed-forward expansion | 0.943577 | -0.003195 | 3 (42–44) |
| `exp015` | Narrower model and row-wise RMSProp | 0.935564 | -0.008013 | 3 (42–44) |
| `exp016` | FlashAttention-4 and CUDA graphs | 0.935105 | -0.000459 | 3 (42–44) |
| `exp017` | Fused QK/RoPE and n-gram accumulation | 0.931969 | -0.003136 | 3 (42–44) |
| `exp018` | Wider model and larger microbatches | 0.936150 | +0.004181 | 3 (42–44) |
| `exp019` | Reuse attention inputs and reorder training documents | 0.932429 | -0.003721 | 3 (42–44) |
| `exp020` | Prepared 16K Unigram and reference-byte evaluation | 0.930738 | -0.001691 | 3 (42–44) |
| `exp021` | Sparse table updates and bounded-logit loss | 0.923441 | -0.007297 | 3 (42–44) |
| `exp022` | Zero-initialize memory tables | 0.922382 | -0.001059 | 10 (42–51) |

### H200, 525 seconds

| Experiment | Change | Mean BPB | Δ BPB | Seeds |
|---|---|---:|---:|---|
| `exp000` | Eight-layer reference with FlashAttention-3 | 0.973908 | — | 3 (42–44) |
| `exp001` | PyTorch attention | 1.011005 | +0.037097 | 3 (42–44) |
| `exp002` | Remove value embeddings | 1.028092 | +0.017087 | 3 (42–44) |
| `exp003` | Remove local windows | 0.995873 | -0.032219 | 3 (42–44) |
| `exp004` | 525-second training budget | 0.972782 | -0.001126 | 3 (42–44) |
| `exp005` | Five wider layers and larger microbatches | 0.961005 | -0.011777 | 3 (42–44) |
| `exp006` | Input bigram embeddings | 0.957160 | -0.003845 | 3 (42–44) |
| `exp007` | Input trigram embeddings and separate optimizer groups | 0.949300 | -0.007860 | 3 (42–44) |
| `exp008` | Larger memory and adjusted learning-rate schedule | 0.938595 | -0.010705 | 3 (42–44) |
| `exp009` | Eight layers and native BF16 weights | 0.934975 | -0.003620 | 3 (42–44) |
| `exp010` | Output normalization and thresholded squared ReLU | 0.934865 | -0.000110 | 3 (42–44) |
| `exp011` | Residual/head gates and two-layer pooling | 0.927435 | -0.007430 | 3 (42–44) |
| `exp012` | Fourteen training shards, excluding validation | 0.927194 | -0.000241 | 3 (42–44) |
| `exp013` | Per-layer hashed n-gram values and optimizer schedules | 0.910094 | -0.017100 | 3 (42–44) |
| `exp014` | Nonuniform feed-forward expansion | 0.907300 | -0.002794 | 3 (42–44) |
| `exp015` | Narrower model and row-wise RMSProp | 0.906465 | -0.000835 | 3 (42–44) |
| `exp016` | FlashAttention-4 and CUDA graphs | 0.906208 | -0.000257 | 3 (42–44) |
| `exp017` | Fused QK/RoPE and n-gram accumulation | 0.903610 | -0.002598 | 3 (42–44) |
| `exp018` | Wider model and larger microbatches | 0.900618 | -0.002992 | 3 (42–44) |
| `exp019` | Reuse attention inputs and reorder training documents | 0.897725 | -0.002893 | 3 (42–44) |
| `exp020` | Prepared 16K Unigram and reference-byte evaluation | 0.894827 | -0.002898 | 3 (42–44) |
| `exp021` | Sparse table updates and bounded-logit loss | 0.888283 | -0.006544 | 3 (42–44) |
| `exp022` | Zero-initialize memory tables | 0.887457 | -0.000826 | 10 (42–51) |

### B200, 300 seconds

| Experiment | Change | Mean BPB | Δ BPB | Seeds |
|---|---|---:|---:|---|
| `exp022` | Zero-initialize memory tables | 0.887791 | — | 10 (42–51) |

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

## Evaluation

BPE experiments use the base evaluator. Unigram experiments replay the rows
selected by the BPE reference tokenizer, preserving their source bytes and
document boundaries. Padding and boundary tokens are excluded from scoring.
A replay that exceeds the model context is rejected rather than truncated.

`bpb` uses the reference evaluator's decoded-token byte denominator.
`literal_bpb` uses the literal UTF-8 byte count. These denominators can differ
for tokens that contain partial UTF-8 characters; compare like with like.
Matching bytes does not make the token-level prediction tasks identical.

## Tests

```bash
uv --quiet run --frozen pytest priml/baselines/nanochat
uv --quiet run --frozen pytest priml/baselines/nanochat/train_step_test.py::test_exp022_two_updates_match_portable_golden -m compute_training
uv --quiet run --frozen pytest priml/baselines/nanochat -m gpu_torch_cuda
```

The first command runs the fast CPU tests. Full configuration goldens run with
`-m compute_large_fixture`. The second runs the reduced-size, two-update model/optimizer
golden using portable kernels; it is excluded from the fast tier. Native GPU tests
check fused outputs and gradients separately. The portable golden does not
certify CUDA numerical identity or a full training score.

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
