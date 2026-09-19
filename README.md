# priml✴️

[![PyPI version](https://img.shields.io/pypi/v/priml.svg)](https://pypi.org/project/priml/)
[![CI](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

ML building blocks for training experiments.

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uv add priml

# Alternatively: python -m pip install priml
```

## What's inside

- **model** -- composable model definitions and building blocks.
- **optimizers** -- optimizer implementations for training.
- **loss** -- loss functions.
- **metrics** -- evaluation metrics.
- **math** -- numerical and math utilities.
- **train** -- the training loop and experiment scaffolding.
- **data** -- the data pipeline and dataset utilities.
- **inference** -- inference helpers.

### Training Loops

```python
from priml.baselines.nanochat import experiments as nanochat

nanochat.exp000().pprint(indent=3)
```

```
NanoChatLoop.Config(
      study_name='nanochat',
      experiment_name='exp000',
      working_dir=PosixPath('/opt/scratch/runs/nanochat/exp000'),
      step=NanoChatTrainStep.Config(
      │  model=NanoChatLM.Config(
      │  │  block=[
      │  │  │  TransformerBlock.Config(
      │  │  │     channels_in=512,
      │  │  │     channels_out=512,
      │  │  │     attn=ValueGatedAttention.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        num_heads=4,
      │  │  │        norm_qk=RMSNorm.Config(
      │  │  │           channels_in=128,
      │  │  │           channels_out=128,
      │  │  │           eps=1.1920928955078125e-07
      │  │  │        ),
      │  │  │        kernel=Flash3Attention.Config(),
      │  │  │        window=1_024,
      │  │  │        max_seq_len=2_048,
      │  │  │        gated=False,
      │  │  │        depth_index=((0, 8),)
      │  │  │     ),
      │  │  │     ffn=SwiGLUReluSquared.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        channels_hidden=2_048,
      │  │  │        expansion=4.0,
      │  │  │        round_to=1,
      │  │  │        depth_index=((0, 8),),
      │  │  │        shard='colwise'
      │  │  │     ),
      │  │  │     norm1=RMSNorm.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        eps=1.1920928955078125e-07
      │  │  │     ),
      │  │  │     norm2=RMSNorm.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        eps=1.1920928955078125e-07
      │  │  │     ),
      │  │  │     depth_index=((0, 8),)
      │  │  │  ),
        ...
      │  │  │  TransformerBlock.Config(
      │  │  │     channels_in=512,
      │  │  │     channels_out=512,
      │  │  │     attn=ValueGatedAttention.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        num_heads=4,
      │  │  │        norm_qk=RMSNorm.Config(
      │  │  │           channels_in=128,
      │  │  │           channels_out=128,
      │  │  │           eps=1.1920928955078125e-07
      │  │  │        ),
      │  │  │        kernel=Flash3Attention.Config(),
      │  │  │        window=2_048,
      │  │  │        max_seq_len=2_048,
      │  │  │        depth_index=((7, 8),)
      │  │  │     ),
      │  │  │     ffn=SwiGLUReluSquared.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        channels_hidden=2_048,
      │  │  │        expansion=4.0,
      │  │  │        round_to=1,
      │  │  │        depth_index=((7, 8),),
      │  │  │        shard='colwise'
      │  │  │     ),
      │  │  │     norm1=RMSNorm.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        eps=1.1920928955078125e-07
      │  │  │     ),
      │  │  │     norm2=RMSNorm.Config(
      │  │  │        channels_in=512,
      │  │  │        channels_out=512,
      │  │  │        eps=1.1920928955078125e-07
      │  │  │     ),
      │  │  │     depth_index=((7, 8),)
      │  │  │  )
      │  │  ],
      │  │  value_embedding_stride=2,
      │  │  embedding=NarrowEmbedding.Config(
      │  │     channels_in=8_192,
      │  │     channels_out=512,
      │  │     dtype=torch.bfloat16,
      │  │     inner=Embedding.Config(
      │  │        channels_in=8_192,
      │  │        channels_out=512,
      │  │        init_weight=functools.partial(<function priml.model.init.normal at 0xdefacedeface>, std=1.0)
      │  │     )
      │  │  ),
      │  │  norm=RMSNorm.Config(
      │  │     channels_in=512,
      │  │     channels_out=512,
      │  │     eps=None
      │  │  ),
      │  │  lm_head=SoftCap.Config(
      │  │     channels_in=512,
      │  │     channels_out=8_192,
      │  │     inner=Linear.Config(
      │  │        channels_in=512,
      │  │        channels_out=8_192,
      │  │        init_weight=functools.partial(<function priml.model.init.normal at 0xdefacedeface>, std=0.001)
      │  │     )
      │  │  ),
      │  │  rope=RoPE.Config(
      │  │     channels_head=128,
      │  │     frequencies=HuggingFaceFrequencies.Config(base=10000.0),
      │  │     dtype=torch.bfloat16
      │  │  ),
      │  │  mix=ResidualMix.Config(
      │  │     num_layers=8,
      │  │     channels_in=512
      │  │  )
      │  ),
      │  optimizer=CompositeOptimizer.Config(
      │     optimizers=[
      │        PartialConfig(<class 'functools.partial'>, <class 'priml.optimizers.fused_adamw.FusedAdamW'>, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, compile=True, lr=0.004898979485566357),
      │        PartialConfig(<class 'functools.partial'>, <class 'priml.optimizers.fused_adamw.FusedAdamW'>, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, compile=True, lr=0.7348469228349535),
      │        PartialConfig(<class 'functools.partial'>, <class 'priml.optimizers.fused_adamw.FusedAdamW'>, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, compile=True, lr=0.7348469228349535),
      │        PartialConfig(<class 'functools.partial'>, <class 'priml.optimizers.fused_adamw.FusedAdamW'>, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, compile=True, lr=0.005),
      │        PartialConfig(<class 'functools.partial'>, <class 'priml.optimizers.fused_adamw.FusedAdamW'>, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0, compile=True, lr=0.5),
      │        NorMuon.Config()
      │     ],
      │     select=[
      │        matching('lm_head'),
      │        excluding(matching('embed'), 'value_embeds'),
      │        matching('value_embeds'),
      │        matching('mix.running'),
      │        matching('mix.original'),
      │        excluding(NorMuon.eligible_tensor, 'embed', 'lm_head')
      │     ],
      │     drop_empty=True
      │  ),
      │  loss=TokenCrossEntropy.Config(ignore_index=-1),
      │  lr_schedule=PartialConfig(<class 'functools.partial'>, <function priml.math.schedules.constant at 0xdefacedeface>),
      │  parallelism=NoParallel.Config(device='cuda'),
      │  model_quantization=NoModelQuantization.Config(),
      │  activation_memoization=DefaultActivationStorage.Config(),
      │  compile=PartialConfig(<class 'functools.partial'>, <function torch.compile at 0xdefacedeface>),
      │  ema=NoEMA.Config(),
      │  schedule=PartialConfig(<class 'functools.partial'>, <function priml.math.schedules.trapezoidal at 0xdefacedeface>, flat=0.5),
      │  adam_lr_tuned_at_channels=512
      ),
      dataset=NanoChatData.Config(
         base_dir='/opt/scratch',
         working_dir=PosixPath('/opt/scratch/datasets/nanochat'),
         tokenizer_dir=PosixPath('/opt/scratch/datasets/nanochat/tokenizer'),
         device='cuda',
         vocab_size=8_192
      ),
      metrics_train={},
      metrics_eval={'val': BitsPerByte.Config()},
      checkpointer=Checkpointer.Config(
         base_dir=PosixPath('/opt/scratch/runs/nanochat/exp000'),
         working_dir=PosixPath('/opt/scratch/runs/nanochat/exp000/checkpoints'),
         storer=SyncLocalStateDictStorer.Config(),
         resume=False
      ),
      phase_timer=PhaseTimer.Config(
         base_dir=PosixPath('/opt/scratch/runs/nanochat/exp000'),
         working_dir=PosixPath('/opt/scratch/runs/nanochat/exp000/profiling')
      ),
      tracker=TrackerList.Config(
         trackers={  'metrics': FileTracker.Config(
               base_dir=PosixPath('/opt/scratch/runs/nanochat/exp000'),
               working_dir=PosixPath('/opt/scratch/runs/nanochat/exp000/metrics.json')
            ),
            'wandb': AsyncTracker.Config(
               tracker=WandbTracker.Config(
                  project='nanochat',
                  base_dir=PosixPath('/opt/scratch/runs/nanochat/exp000'),
                  working_dir=PosixPath('/opt/scratch/runs/nanochat/exp000/wandb'),
                  ingestion=WandbIngestion(),
                  metric_step_metrics={},
                  run_config={}
               )
            )},
         base_dir=PosixPath('/opt/scratch/runs/nanochat/exp000'),
         working_dir=PosixPath('/opt/scratch/runs/nanochat/exp000')
      ),
      max_time=300.0,
      max_time_kind='train',
      num_steps_eval=-1,
      num_steps_log=1,
      early_train_log_steps=0,
      eval_every_epoch=False,
      seed=42,
      runtime=SingleProcess.Config(
         device='cuda',
         float32_matmul_precision='high'
      )
   )
```


### Computational Cost

Priml has rich support for understanding where compute is (theoretically) being
spent.

For example,

```python
from priml import cost as theoretical
from priml.baselines.sudoku import experiments as sudoku

sudoku.exp000().finalize().step.model.cost(batch_size=64, dtype=torch.bfloat16)
sudoku.exp001().finalize().step.model.cost(batch_size=64, dtype=torch.bfloat16)

theoretical.peak()["rtx5090", torch.bfloat16]
theoretical.peak()["h100", torch.bfloat16]
```

prints,

```
Cost(params=6841858, params_active=6824450, bytes_state=8192)
                     flops  bytes  bytes intensity
                      bf16   bf16  int64      bf16
primal  matmul      144.8G 532.4M      -       272
primal  elementwise 322.1M   907M      -    0.3551
primal  reduction   47.73M 96.96M      -    0.4923
primal  selection        - 11.11M 43.42K         -
adjoint matmul      289.7G 1.065G      -       272
adjoint elementwise 393.7M 1.454G      -    0.2708
adjoint reduction   37.08M 75.15M      -    0.4934
adjoint selection   2.779M 16.71M 43.42K    0.1663
total               435.3G 4.158G 86.83K     104.7


Cost(params=4869122, params_active=4851714, bytes_state=0)
                     flops  bytes  bytes intensity
                      bf16   bf16  int64      bf16
primal  matmul      114.2G 502.2M      -     227.4
primal  elementwise 601.2M 1.946G      -    0.3089
primal  reduction   86.33M 173.9M      -    0.4965
primal  selection        - 11.11M 43.42K         -
adjoint matmul      228.4G 1.004G      -     227.4
adjoint elementwise 837.5M 3.888G      -    0.2154
adjoint reduction   88.95M 179.3M      -    0.4962
adjoint selection   2.779M 16.71M 43.42K    0.1663
total               344.2G 7.722G 86.83K     44.58


             flops  bytes intensity
matmul      209.5T 1.792T     116.9
elementwise 104.8T 1.792T     58.48
reduction   104.8T 1.792T     58.48
selection   104.8T 1.792T     58.48
sort        104.8T 1.792T     58.48


            flops bytes intensity
matmul       989T 3.35T     295.2
elementwise   67T 3.35T        20
reduction     67T 3.35T        20
selection     67T 3.35T        20
sort          67T 3.35T        20
```

The H100's BF16 matmul roofline knee is `295.2` FLOPs/byte. The
transformer's matmul intensity is about 8% below this knee
(`272 / 295.2 - 1 = -0.07859`); the mixer's is about 23% below it
(`227.4 / 295.2 - 1 = -0.2297`). The knee is the crossover from memory-bound to
compute-bound operation. By this matmul-intensity criterion, the MLP mixer
configuration is less well tuned to the H100's compute-to-bandwidth ratio
than the transformer configuration.

Without a duration, `utilization` returns intensity divided by the device's
roofline knee for each cell. With `duration_sec`, it returns achieved FLOP/s
divided by `min(peak FLOP/s, intensity * peak bytes/s)`. For a compute-bound
matmul cell this equals MFU; for a memory-bound cell the denominator is lower
than peak compute. The `device` argument accepts a name such as `"h100"`
or a single-device report such as `theoretical.peak()["h100"]`.

```python
theoretical.utilization(
    sudoku.exp000().finalize().step.model.cost(batch_size=64, dtype=torch.bfloat16),
    device="h100",
)
```

prints,

```
                        bf16
primal  matmul        0.9214
primal  elementwise  0.01775
primal  reduction    0.02462
adjoint matmul        0.9214
adjoint elementwise  0.01354
adjoint reduction    0.02467
adjoint selection   0.008314
total                  1.932
```

The displayed utilization `total` sums per-cell ratios; it is not a
whole-model utilization and should not be interpreted as one.

Why is the smaller mixer less well tuned? It has about 29% fewer parameters,
but shrinking the model does not reduce computation and operand traffic in
the same proportion. Its primal matmul work falls about 21% (`114.2G` versus
`144.8G` FLOPs), while its primal matmul bytes fall only about 6% (`502.2M`
versus `532.4M`). The mixer therefore does less arithmetic for nearly the
same operand traffic, moving its matmul intensity farther below the H100's
knee. The transformer gets about 20% more arithmetic per byte
(`272 / 227.4 - 1 = 0.1961`).

The parameter-to-traffic comparison tells the same tuning story. For primal
matmuls, the transformer has `6841858 / 532.4M = 0.01285` parameters per
operand byte, versus `4869122 / 502.2M = 0.009696` for the mixer: about 33%
more parameters for the same traffic budget. Fewer parameters alone do not
make a configuration better matched to the hardware.

Across primal and adjoint, the mixer's non-matmul operand traffic is about
2.43 times the transformer's (`6.216G` versus `2.561G` bytes). The displayed
BF16 intensity totals are `44.58` and `104.7`, respectively. These divide
summed BF16 FLOPs by summed BF16 bytes; they exclude the separate integer
traffic. An all-dtype intensity divides total FLOPs by total bytes, including
integer/index traffic, rather than adding or averaging per-dtype intensities.

These bytes describe logical operand I/O for the analytical unfused algorithm.
They are not measured HBM traffic: fusion and cache reuse can change physical
memory transfers. The roofline comparisons above use that analytical traffic
model; they do not establish achieved hardware utilization.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for local validation and the public
contribution flow.

## See also

Sibling projects in the [rekursiv-ai](https://github.com/rekursiv-ai) family:

- [sagent](https://github.com/rekursiv-ai/sagent) — The self-mutating multi-provider coding-agent CLI and typed Python library.
- [trackinizer](https://github.com/rekursiv-ai/trackinizer) — Centralized agent database for tracking inquiries, work, and the evidence behind conclusions.
- [wesearch](https://github.com/rekursiv-ai/wesearch) — Web search, resilient page fetch, and scholarly-paper lookup without a browser stack.
- [madcatter](https://github.com/rekursiv-ai/madcatter) — Rich-based Markdown renderer for the terminal; ships the `mdcat` CLI.
- [configgle](https://github.com/rekursiv-ai/configgle) — Hierarchical experiment configuration in typed pure-Python dataclasses instead of YAML.
- [copybarista](https://github.com/rekursiv-ai/copybarista) — Bidirectional source sync for publishing OSS-ready trees from a monorepo.

## Citing

If you find our work useful, please consider citing:

```bibtex
@misc{rekursivai2026priml,
      title={Priml - Strongly typed configurable building blocks for A/B testing ML research.}
      author={Joshua V. Dillon and Dan Kondratyuk},
      year={2026},
      howpublished={Github},
      url={https://github.com/rekursiv-ai/priml},
}
```
