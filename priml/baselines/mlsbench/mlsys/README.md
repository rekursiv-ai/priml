# MLS-Bench `Sys` area: TODO

TODO: port the 10 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `dlm-dkv-policy` | maomaocun/dLLM-Cache | yes |
| `llm-kv-adaptive-quantization` | huggingface/transformers | yes |
| `llm-kv-selection-budgeting` | huggingface/transformers | yes |
| `llm-kv-structural-reduction` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-kernel` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-ptq-algorithm` | IST-DASLab/gptq | yes |
| `llm-qat-algorithm` | custom | yes |
| `mlsys-fused-attention` | Dao-AILab/flash-attention | yes |
| `mlsys-moe-load-balance` | deepseek-ai/eplb | yes |
| `mlsys-sparse-attention-inference` | custom | yes |
