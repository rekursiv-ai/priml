# MLS-Bench `LM` area: TODO

TODO: port the 16 in-scope tasks below as native priml
baselines, one subpackage each, the way [`../dl/`](../dl/) does: the
task's editable function becomes a config slot, `exp000` is the pristine
recipe, each published baseline is a fork of that slot, and
`scripts/capture_reference.py` proves parity against upstream commit
`b873944969c53b01e4ee6e843c77767b8c702485`. Tasks built on one upstream
package share one subpackage of common code, as
[`../dl/vision/`](../dl/vision/README.md) does for the DL area.

| Task | Upstream package | In scope |
|---|---|---|
| `agent-tool-reasoning` | zhichengg/StableToolBench | no: evaluation calls a hosted model API |
| `llm-dllm-demask-strategy` | ML-GSAI/LLaDA | yes |
| `llm-pretrain-attention` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-bitlinear` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-embedding` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-linear-attention` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-loss` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-lr-schedule` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-mlp` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-normalization` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-optimizer` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-pretrain-residual` | karpathy/nanoGPT, EleutherAI/lm-evaluation-harness | yes |
| `llm-rl-advantage` | volcengine/verl | yes |
| `llm-rl-importance-sampling` | volcengine/verl | yes |
| `llm-rl-kl-estimator` | volcengine/verl | yes |
| `llm-rl-reward-normalization` | volcengine/verl | yes |
| `llm-scaling-law-discovery` | trevorstephens/gplearn | yes |
| `mas-topology` | OpenBMB/ChatDev | no: evaluation calls a hosted model API |
