# priml✴️

[![PyPI version](https://img.shields.io/pypi/v/priml.svg)](https://pypi.org/project/priml/)
[![CI](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/priml/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

ML building blocks for training experiments.

## Quick Start

```bash
uv add priml
```

See [Install](#install) for additional details.

## What's inside

- **model** -- composable model definitions and building blocks.
- **optimizers** -- optimizer implementations for training.
- **loss** -- loss functions.
- **metrics** -- evaluation metrics.
- **math** -- numerical and math utilities.
- **train** -- the training loop and experiment scaffolding.
- **data** -- the data pipeline and dataset utilities.
- **inference** -- inference helpers.

Modules, costs, and metrics pass open keyword arguments down the tree; see
[the keyword bus](docs/kwargs_bus.md) for the rule and why it is sound.

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
      │  │  │        kernel=Flash4Attention.Config(),
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
      │  │  │        kernel=Flash4Attention.Config(),
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

theoretical.peak()["rtx5090", torch.bfloat16]
theoretical.peak()["h100", torch.bfloat16]
```

prints,

```
Cost(params=6833666, params_active=6823426, bytes_state=20480)
                     flops  flops  bytes  bytes  bytes intensity intensity
                      bf16    f32   bf16    f32  int64      bf16       f32
primal  matmul      1.307T      - 4.807G      -      -     271.8         -
primal  elementwise 3.506G 524.3K 12.24G 4.194M      -    0.2864     0.125
primal  reduction   762.3M      - 1.539G      -      -    0.4954         -
primal  selection        -      - 22.61M      - 84.48K         -         -
adjoint matmul        871G      - 3.205G      -      -     271.8         -
adjoint elementwise 1.585G 524.3K 7.525G 4.194M      -    0.2106     0.125
adjoint reduction   206.4M      - 415.6M      -      -    0.4967         -
adjoint selection   5.308M      - 31.87M      - 82.94K    0.1665         -
total               2.184T 1.049M 29.79G 8.389M 167.4K     73.31     0.125


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
(`271.8 / 295.2 - 1 = -0.07927`). The knee is the crossover from
memory-bound to compute-bound operation.

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
                        bf16     f32
primal  matmul        0.9206       -
primal  elementwise  0.01432 0.00625
primal  reduction    0.02477       -
adjoint matmul        0.9206       -
adjoint elementwise  0.01053 0.00625
adjoint reduction    0.02484       -
adjoint selection   0.008327       -
total                  1.924  0.0125
```

The displayed utilization `total` sums per-cell ratios; it is not a
whole-model utilization and should not be interpreted as one.

These bytes describe logical operand I/O for the analytical unfused algorithm.
They are not measured HBM traffic: fusion and cache reuse can change physical
memory transfers. The roofline comparisons above use that analytical traffic
model; they do not establish achieved hardware utilization.


## Install

```bash
uv add priml
```

If you're missing `uv`,
- Mac
  ```bash
  brew install uv
  ```
- Linux
  ```bash
  sudo apt install -y curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Or with pip,
```bash
python -m pip install priml
```

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
