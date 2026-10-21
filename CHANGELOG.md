# Changelog

All notable priml changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.7 - 2026-10-21

### Added

- `priml.model.ae`: pretrained autoencoders behind one contract
  (`custom_types.py`): `Encoder.encode(x) -> Distribution`,
  `Decoder.decode(z) -> Tensor`, `Autoencoder` both, over arbitrary leading
  dims. A `Distribution` has `sample(*, generator)` and `mode()`; INVAE
  returns a `DiagonalGaussian`, RAE and VTP a `PointMass`. Implementations:
  `INVAE`, `VTP` (`vtp_small`, `vtp_base`, `vtp_large`), and `RAE`
  (`rae_dinov2_base`). Each config has `encoder` and `decoder` slots, either
  `None`; each half carries its own `checkpoint` and reads only its own
  tensors. `Config.from_hf` parses a published `config.json`, `load()` builds
  from a Hub repository or local directory, and `remap_*_state_dict` renames
  published tensors. Checkpoints are config nodes (`HubFile` at a commit
  revision, `UrlFile`, `LocalFile`), and every published default pins its
  SHA-256; each config's `latent_norm` holds its published normalizer
  (`ScaleLatents`, `ElementwiseLatentStats`, `ChannelLatentStats`).
- `priml/model/ae/testdata/rae_reference.pt` and `vtp_reference.pt`: each
  reference's own outputs at its pinned commit, replayed bit for bit by
  `rae_test` and `vtp_test`.
- `priml.math.quantization`: `lloyd_max` (exact Lloyd-Max fitting on sorted
  samples, the cube-root high-resolution start, standard-normal levels),
  `fits` (uniform, quantile, and moment-matched levels), and `nearest`
  (batched nearest-level `quantize` / `dequantize`).
- SpeedrunDiT: storage codecs (`FloatCodec`, `ScalarTableCodec` with injected
  table fits and channel grouping), a corpus receipt checked at load time,
  `prepare_data.py --experiment`, `benchmark_codec.py`, and experiments
  exp002 (VTP), exp003 (RAE, float16 storage), and exp004 (RAE, uint8
  storage). The loader refuses a corpus whose preparation has not finished.
  `images/source.json` binds the crops every corpus shares to their ImageNet
  source, and a rerun resumes only with the same source, producers, seed, and
  (for a seeded corpus) batch size.
- `cost(..., context_len=...)` prices a step after `context_len` cached
  tokens: `Attention`, `GatedAttention`, and `MultiStreamAttention` attend
  their `seq_len` queries to `context_len + seq_len` keys. One decode step is
  `cost(model, seq_len=1, context_len=T, ...)`, and `utilization` places it
  against a device's ridge. `MultiHeadLatentAttention` raises
  `NotImplementedError` for a cached step.
- `priml.cost.nbytes` returns the bytes a count of values occupies at a dtype.
- Baseline `mlsbench.dl.vision`: MLS-Bench `cv-classification-loss` at
  `resnet56-cifar100` as native priml experiments, `exp000` (cross-entropy)
  to `exp003`, each loss a fork of one slot. `scripts/capture_reference.py`
  proves the port bit-for-bit against the upstream script.
- Baselines `mlsbench.dl.lr_schedule`, `pooling`, `normalization`,
  `initialization`, `activation`, `regularization`, `multitask`, `weighting`,
  `residual` and `augmentation`: ten more MLS-Bench tasks, each `exp000` the
  pristine recipe and each published baseline a fork of the task's slot, each
  proven bit-for-bit by its own `scripts/capture_reference.py`.
- `mlsbench.dl.vision` gains the slots those tasks fill:
  `ResNet.Config.norm`, `pool`, `init` and a per-position `block` list, and
  `Cifar100Data.Config.augment`.

### Changed

- `cv-classification-loss` moved from `mlsbench.dl.vision` to its own baseline,
  `mlsbench.dl.classification_loss`; `vision` keeps the shared scaffold and
  `exp000`, now under study `mlsbench_dl_vision`.
- `priml.model.invae` moved to `priml.model.ae.invae`. `AutoencoderKL`,
  `VAE_F16D32`, `VAE_F8D4`, `vae_models`, `encode_image`, `decode_latents`,
  and `load_invae` are removed; build `INVAE.Config().make()` and call
  `encode` / `decode`. INVAE now loads the published diffusers file
  `REPA-E/e2e-invae-hf`, whose tensors equal REG's `e2e-invae-400k.pt`.
- SpeedrunDiT: `SpeedrunTrainStep.Config.latent_scale` is replaced by the
  `latent_norm` slot, filled at loop finalize from the dataset's autoencoder
  (INVAE's is the same `* 0.3099`). `PairedImageLatentDataset.Config` gains
  `latent_subdir`, `autoencoder`, `codec`, and `seed`, and requires a
  `corpus.json` receipt; record one for an existing REG corpus, and bind its
  crops, with `prepare_data.py --receipt-only --source <imagenet>`.
  `prepare_data.py` takes `--experiment` in place of `--output`,
  `--checkpoint`, and `--resolution`. `PairedImageLatentDataset.Config.sample`
  chooses a posterior draw or its mode, and is part of the receipt's
  identity; the receipt format is 3, so re-record an existing corpus with
  `--receipt-only`.
- SpeedrunDiT: `SpeedrunTrainStep.Config.ema` keeps parameter shadows
  (`shadow_kind="param_dict"`); the module-copy default could not deepcopy the
  composable `replicate` model, so every distributed run failed at its first
  EMA update. EMA state saved under the old default does not load into it.
- `Cost` and `Report` carry their `axes`, so a slice answers by the axes it
  kept whether or not it holds a cell. An empty slice used to accept any axis
  and read a sub-table where a full one read zero.
- Naming every axis of a sub-table, such as `cost["flops"]["primal",
  "matmul", torch.bfloat16]`, is a `KeyError`; read it with `.sum()`, or name
  all four axes on the full table. A three-value `Report` lookup is typed
  `float | Report`: a three-axis report reads a value, `peak()` a sub-table.
- `CompositeOptimizer.Config` with `require_total=False` freezes the trainable
  parameters no filter claims, as documented; their gradients used to pile up
  because no member zeroed them. An empty `select` serves a single member
  only, and a member that is itself a `CompositeOptimizer.Config` is rejected
  by `make()`.
- The filter combinators `matching`, `excluding`, `complement` and
  `with_ndim` equal only instances of the same class, and their fields cannot
  be reassigned, so a filter used as a dict key keeps its hash.
- priml declares `triton` (Linux), `pillow` and `tqdm`, which it imports and
  used to receive only through torch, torchvision and lpips. The `cuda` extra
  installs `flash-attn-4` and `flash-linear-attention`.
- `Flash4Attention` and `Flash4Varlen` raise `Flash4UnavailableError` naming
  `priml[cuda]` when FlashAttention 4 is not installed, instead of
  `ModuleNotFoundError`.
- The wheel no longer ships test support no shipped module imports:
  `priml.lib.testing`, `priml.distributed.testing`,
  `priml.testing.{experiments,fixtures,policy_gradient,qwen3_5}`, and the
  `testing` modules of the convextok, craftax, sudoku and tmax baselines.

### Removed

- `priml.data.processors.logarithm`. The casmi2026 and wellbore baselines
  evaluate `log`, `log1p`, `expm1` and `pow` in float64 and round once to
  float32, which does not depend on the host's libm.

### Fixed

- SpeedrunDiT: gradient accumulation weights micro-batches by their sample
  counts, the step reports `skipped_steps`, refuses a closure-based optimizer
  its stochastic objective cannot serve, and names a teacher that returns no
  feature tuple.
- SpeedrunDiT: the model refuses `encoder_blocks < 1`, whose value-residual
  blends never trained, and freezes the fusion mask token when no token or path
  is ever dropped; replicated ranks let such untrained parameters drift apart.
  It refuses a `drop_ratio` outside `[0, 1)` and a `path_drop_prob` outside
  `[0, 1]`. A model built on the meta device initializes every tensor, and its
  cost counts the projector when no depth projects.
- SpeedrunDiT: the corpus loader refuses two files naming one pair, and an
  image that is not `[3, H, W]` uint8.
- Costs at `torch.float4_e2m1fn_x2` count two values per byte. Every byte
  count multiplied by `itemsize`, which is one byte per two packed fp4 values,
  so fp4 traffic and cache state were doubled.
- `CompositeOptimizer.state` reads such as `get`, `items` and `in` see state a
  member created after construction, and `param_groups` includes groups added
  through a member's `add_param_group`, so schedules reach them. `step()`
  raises `ValueError` when a member has taken a parameter another member owns,
  instead of updating it twice.
- `CompositeOptimizer` keeps torch's `Optimizer` contract. It sets
  `requires_closure` when a member does, so the training loop hands a Newton
  member its closure, and that closure runs once per step instead of twice.
  A closure no member takes runs with grad enabled. Checkpoint hooks
  registered on the composite run, a failed `load_state_dict` leaves every
  member unchanged, `deepcopy` and pickle keep the members, and `defaults`
  lists the hyperparameters every member shares, so `CyclicLR` and
  `OneCycleLR` accept a momentum composite.
- Indexing `CompositeOptimizer.state` with a parameter that has no entry yet
  creates the entry on its member, as torch's defaultdict does, instead of
  raising `KeyError`. `filter_name` no longer returns a memory address.
- `setup_logging`'s startup replay buffer keeps at most 10,000 records. It
  used to keep every record for the life of the process unless a W&B tracker
  replayed it, which is off by default, so a run held each step line and any
  exception traceback (frames included) until exit. A replay past the cap
  says how many records it skipped.

## 0.1.6 - 2026-10-07

### Added

- `priml.data.passes.Passes` wraps a function that draws one pass of
  batches, so a training loader iterates afresh at every epoch boundary
  instead of handing back an exhausted generator.
- `priml.runtime.best_device()` returns the best accelerator this process
  can use, else CPU.
- `priml.testing.golden.assert_pprint_golden(test_file=..., name=...,
  config=...)` compares a config's full finalized pprint with a
  test-local golden, with an optional `normalize`; it replaces
  `configgle.testing.assert_pprint_golden`. `priml.testing.regenerate`
  adds the `--regenerate-golden` pytest option.
- ARC evaluation can write ranked per-task pass@k JSON.
- An ETTh1 DLinear forecasting baseline (`priml.baselines.etth1`).

### Changed

- **Breaking:** runtime `device` defaults to `None`, which selects the best
  usable accelerator, else CPU. The `"auto"` spelling is gone:
  `get_device("auto")` now raises. `get_device(None)` returns the device
  being built under (`torch.get_default_device()`).
- A CUDA build on a machine with no usable GPU resolves the default device
  to CPU instead of failing on its first allocation. Processors and model
  loaders default to their build-time device; explicitly configured
  devices are kept.
- **Breaking:** `RoPE`'s `dtype` defaults to `torch.float32` (was `None`).
  Fixed frequencies stay float32 through module casts; learned frequencies
  follow the module dtype. Numerics are unchanged.
- **Breaking:** the vendored `priml.lib.custom_json` is replaced by
  `priml.lib.codec`. Dataset metadata, manifests and training checkpoints
  are read with its typed parser.
- Reductions normalize their axes and reject invalid dimensions; dataset
  recipes, resume state, model configs and optimizer inputs are validated
  up front.
- Development: the bundled typeshed patch and pytest options are updated, and
  a worker-count helper is vendored as `priml.lib.worker_count`. Runtime
  dependencies and the supported Python versions (3.12+) are unchanged.

### Fixed

- Transformer mixers are no longer passed an absent cache.
- A missing false-accept count is treated as zero during evaluation.
- Fixes to attention, sampling, metrics, image processing, checkpoint
  handling, and parallel data and distributed workflows.

### Removed

- **Breaking:** `priml.testing.fixtures.get_device`; use
  `priml.runtime.best_device()`.
- `Checkpointer.has_pending_write`.

## 0.1.5 - 2026-10-03

### Changed

- Requires configgle 1.4.1 or newer.
- `priml.model` is split into the `priml.model.attention` and
  `priml.model.transformer` packages. The modules `priml.model.rope`,
  `mla`, `kvcache`, `gated_delta_net`, `value_gated_attention`, `mmdit`,
  `qwen3`, `kimi_k2`, and `causal_lm` are gone: import from the new
  packages, or from `priml.model`, which still re-exports the main
  classes. `SdpaFused` and `SdpaNaive` are in
  `priml.model.attention.kernel`, the window helpers in
  `priml.model.attention.window`, and `TransformerBlock` in
  `priml.model.transformer.block`.
- `SelfAttention` is renamed `Attention`, which also attends to a memory
  (cross-attention). Attention configs call the head count `num_heads`
  (was `heads`); `GatedDeltaNet` takes `num_heads_k` and `num_heads_v`.
- `CausalLM` is replaced by `Transformer`: a block stack with optional
  `proj_in` and `proj_out`. The head owns its final norm, the output
  width is `channels_out` (was `vocab_size`), and a tied head is
  `TiedLinear.Config(tied="proj_in")`. `final_norm`, `lm_head`, and
  `tie_embeddings` are gone, and the state-dict keys change with them.
- `Qwen3` and `KimiK2` subclass `Transformer`. Their flat Hugging
  Face-style config fields are replaced by `num_layers`, `proj_in`,
  `proj_out`, and a `block` template; `Config.from_hf` and `load` work as
  before.
- Layer configs name widths by channels. `Embedding` and
  `NarrowEmbedding` take `channels_in` (was `num_embeddings`). Norms,
  attention blocks, `MLPMixerBlock`, and `Patchify` gain `channels_out`,
  and a config given either width infers the other. Baseline configs
  rename `hidden_size` and `vocab_size` the same way.
- Depth-scaled initialization takes `depth_index`, a tuple of
  `(index, count)` pairs, instead of the integer `depth`, on every layer
  config and every `priml.model.init` initializer. Called without one, an
  initializer no longer scales; the old default `depth=1` divided by
  sqrt(2).
- `RoPE` takes a `frequencies` table (`HuggingFaceFrequencies`,
  `GeometricFrequencies`, or a `YarnScaling` wrapper) in place of `base`,
  `yarn`, and `hf_inv_freq`.
- `Router` is an abstract base. `SoftmaxRouter` (auxiliary-loss
  balancing, the `MoE` default) and `SigmoidRouter` (correction bias and
  grouped top-k, as in DeepSeek-V3 and Kimi K2) each own their fields.
- `SwiGLU` defaults change, so freshly built models differ: `up_proj`
  initializes with `unit_fan_in_uniform`, `down_proj` starts at zero, and
  an ungated block expands by 4 rather than 8/3. `MoE` experts inherit
  this. Set `init_weight`, `init_weight_out`, and `expansion` to keep the
  old behavior.
- `rgb2float` and `float2rgb` take keyword-only `float_dtype`, `inplace`,
  and `unit_interval` (for the [0, 1] range). `rgb2float` raises
  `TypeError` on integer input unless `float_dtype` is given; it used to
  cast to float32 silently.
- Training configs are renamed. On `TrainLoop.Config`, `metrics` is
  `metrics_eval` (beside a new `metrics_train`), `checkpointing` is
  `checkpointer`, `profiling` is `profiler`, and `phase_heartbeat_sec`
  moves to `PhaseTimer.Config.heartbeat_interval_sec`. On
  `TrainStep.Config`, `learning_rate_scheduler` is `lr_schedule`.
- `priml.train.checkpointing` is `priml.train.checkpointer` and
  `priml.train.profiling` is `priml.train.profiler`. `TorchProfiling` is
  `TorchProfiler`, `CheckpointingProtocol` is `CheckpointerProtocol`, and
  `ProfileProtocol` is `ProfilerProtocol`.
- Parameter selectors move to `priml.optimizers.parameter_filter`
  (`matching`, `excluding`, `complement`, `everything`, and the new
  `trainable`) and are no longer re-exported from `priml.optimizers`.
  `ParameterFilter` replaces `Selector`, and `EMA.Config.select` takes the
  same filters.
- `JobProtocol` and `LaunchableExperiment` move to `priml.custom_types`,
  `priml.lib.custom_types` is renamed `priml.lib.absent`, and
  `convert_to_tensor` moves from `priml.math.custom_types` to
  `priml.memory`.
- `priml.lib.custom_json` is rebuilt around typed codecs (`IntCodec`,
  `ListCodec`, `DataclassCodec`, and others); `json_freeze` and
  `json_unfreeze` take `allow_nan`.
- Hugging Face loaders leave the cache location to `HF_HOME`, and other
  model downloads use `TORCH_HOME` when it is set, so models fetched by
  0.1.4's Hugging Face loaders download again. `load_transformers_model`
  passes `dtype` instead of the deprecated `torch_dtype`.
- An evaluation due at a checkpoint step now runs before the save, so the
  checkpoint holds the post-evaluation state. A single-process run still
  saves when that evaluation raises.
- `priml.testing.bfb` goldens live in `testdata/`, record every tensor the
  run changed, and replay bit for bit across CPU architectures. Regenerate
  goldens made with 0.1.4 (`BFB_REGENERATE=1`).
- `imageio-ffmpeg`, `opencv-contrib-python-headless`, `numba`, and
  `tokenizers` are core dependencies, and `wandb` needs 0.29 or newer.
  `pyturbojpeg` 2.x is allowed; it needs libjpeg-turbo 3.0 or newer, which
  Ubuntu 24.04 does not ship, and the repository's
  `install-libjpeg-turbo.sh` installs it.
- The package ships `py.typed`, so type checkers read priml's annotations.
- `setup_logging` requires its `level`; it defaulted to `"INFO"`.
- `lazy_torch_compile` forwards only keyword arguments to `torch.compile`.
  A positional argument other than the decorated function raises
  `TypeError`.
- A mesh dimension below -1 raises `ValueError` instead of acting as -1
  (auto).
- `q_lambda_targets` no longer rounds the discount to bf16 when its inputs
  are bf16, and bf16 Q-values beside fp32 rewards stay in bf16 instead of
  reaching fp32 through the done mask. bf16 targets can move by a rounding
  step; fp32 targets are unchanged.

### Added

- `priml.cost`: the analytical FLOP and byte cost of one model call, read
  from its config and split by kernel kind and by forward and backward
  pass, with datasheet peaks and roofline utilization for A100, H100,
  H200, B200, RTX 50-series, and RTX PRO 6000 GPUs. The `Utilization`
  training metric reports MFU from it, and `priml.testing.cost` checks a
  config's cost against what torch dispatches.
- Models: `Qwen35` (the Qwen3.5 hybrid gated-delta text model, with
  Hugging Face loading and cached generation), `MMDiTGraft` and
  `Qwen3MMDiTGraft` (add modality streams to a language backbone),
  `MinGRU`, `DinoV2Teacher`, the INVAE autoencoder, diffusion
  `TimestepEmbedder` and `LabelEmbedder`, `MultiHotEmbedding`,
  `TiedLinear`, SPRINT token routing, and pooling costs.
- Attention: `GatedAttention`, `ValueResidualAttention`, `SdpaVarlen` and
  `segment_mask` for packed sequences, `Flash3Attention` (built once by
  `prepare_flash3`), and `Flash4Attention` and `Flash4Varlen`, which run
  under `torch.compile(fullgraph=True)`.
- Optimizers: `FusedMuon` (fp32 master weights, global-norm clipping, and
  fused CUDA kernels) and `learning_rate`.
- Losses and math: `cross_entropy_logz`, `contrastive_flow_loss`, the
  `PPO` rule with `TorchPPO` and a fused Triton `TritonPPO`,
  `observation_aligned_advantage`, the chunked and recurrent gated delta
  rule, sin-cos position tables, Euler-Maruyama sampling and
  resolution-aware time shifting for flow matching, `holm`,
  `total_variation`, Gumbel-max sampling, and the `cyclic` schedule.
- Training: best-checkpoint saving (`Checkpointer.Config.best_metric` and
  `best_mode`), `TrainStep.Config.skip_step_on_nonfinite_grad`, and thread
  stack dumps when a phase stalls
  (`PhaseTimer.Config.fault_dump_interval_sec`).
- Data: `priml.data.pipeline`, `priml.data.processors`, and
  `priml.data.sources` for batching, shuffling, parallel prefetch, GPU
  transfer, JPEG decoding and augmentation, and ImageNet and parquet/tar
  datasets; `priml.data.ensure` for resumable, verified dataset downloads.
- Baselines: `arcagi2`, `convextok` (a PyTorch port of the ConvexTok
  linear-programming tokenizer), `imagenet` (the ffcv ResNet-50 recipe),
  and `speedrundit` (REG/SPRINT latent diffusion). The `arcagi1`,
  `nanochat`, and `sudoku` experiment ladders are extended.
- `priml.kernel` (lazy Triton kernel builds), `priml.memory`,
  `priml.testing.golden` (tensor golden files), and the Hub loaders
  `load_hf_checkpoint` and `load_torch_hub_distributed`.

### Removed

- The `render` extra; video encoding is a core dependency.
- `priml.math.activations`; `relu_squared` and the other activations are
  in `priml.model.swiglu`.
- `NormProtocol`, `NormConfigProtocol`, `LossProtocol`, `AttentionBlock`,
  `HasDepth`, `ShardableConfig`, and `ModuleLike`.
- `JsonCodec`, `dataclass_to_json`, `dataclass_from_json`, and the
  `*_val` helpers in `priml.lib.custom_json`.
- `priml.baselines.nanochat.flash3`; use `priml.model.attention.flash3`,
  and `priml.model.attention.prepare_flash3` to build FA3. `is_prepared` is
  gone; an empty `artifact_validation_error(artifact_path())` means the same.

### Fixed

- Cropped JPEG decodes find libturbojpeg where `TurboJPEG()` does. The
  region decoder only asked `find_library`, which misses a Homebrew install
  on Apple Silicon, so every crop there decoded to `None`.
- Cached decoding honors a configured attention window. Single-token
  decode passed an all-zero mask that switched the window off.
- Initializers draw bf16, fp16, and fp8 tensors in fp32 and round once. On
  CPU a bf16 uniform draw held only 256 distinct values, and a bf16 normal
  draw stopped near 3.3 sigma.
- `SoftCap` passes keyword arguments to its inner module instead of
  dropping them.
- The Craftax baseline matches upstream Craftax: combat, crafting, chest
  rewards, mob spawning, observations, and world generation are corrected,
  and player arrows and spells now hurt mobs. Its rollout and learner run
  as CUDA graphs, and its GPU evaluation works.
- The nanochat baseline's Torch 2.9 runtime project installs configgle, which
  every priml module imports, so the FA3 build command runs in it.
- The ARC-AGI-1 baseline's `PassK` merges every rank's votes before ranking
  answers, so a multi-GPU evaluation no longer counts a puzzle whose samples
  span ranks once per rank.
- The mean of `cross_entropy_with_batched_smoothing` divides by the exact
  count of non-ignored targets. In bf16 that count was summed in bf16, which
  rounds above 256.

## 0.1.4 - 2026-08-19

### Changed

- Requires configgle 1.3.7 or newer.
- Path helpers moved to `priml.paths`. `runtime_output_path` is renamed
  `validated_output_path` and gained an optional `protected` argument that
  refuses a destination aliasing one of the run's own inputs.
- `resolve_working_dir` now lives in `priml.paths` (previously vendored).
- `validated_output_path` expands `~` and is canonicalized; the
  Git-checkout guard is removed.
- Learning-rate schedules moved from `priml.train.schedules` to
  `priml.math.schedules`, and take a `progress` float in `[0, 1]` rather
  than `step` and `total_steps`.
- `log_stablemax`, `stablemax_cross_entropy`, and
  `cross_entropy_with_batched_smoothing` moved to `priml.math.loss`; the
  stablemax pair is no longer re-exported from `priml.loss`.
- `Learnable` and `LearnableProtocol` are removed from `priml.train`;
  `TrainStep` absorbs their role, and its `compile` field now wraps the
  model independently of the optimizer.
- `priml.lib.userdirs` helpers (`data_dir`, `config_dir`, `cache_dir`,
  `state_dir`) no longer take an `app` argument and `platform` is
  keyword-only. The model cache moved from `~/.cache/loop/models` to
  `cache_dir() / "rekursiv-ai" / "models"`.
- `priml.math.stats.pca` takes an injected `decompose` callable
  (`pca_eigh`, `pca_svd`, `pca_power`) in place of the `algorithm` string
  with `power_iters` and `power_tol`.
- `num_steps_eval` now names an eval regime: `> 0` is a cadence plus the
  final eval, `-1` is the final eval only, and `0` or `inf` disables eval
  entirely.
- Distributed topology is validated before resources are acquired.
- Normalization layer widths are inferred rather than declared.
- Budget warmup counting and eval cadence are correct under gradient
  accumulation.
- The data, tokenizer, and media stacks (`pyarrow`, `rustbpe`, `tiktoken`,
  `pygame-ce`, `einops`) are core dependencies; policy rendering is the
  new optional `render` extra.

### Added

- `priml.baselines`: reference end-to-end training pipelines built from
  priml components, one subpackage per dataset (`arcagi1`, `cifar10`,
  `craftax`, `nanochat`, `sudoku`), each exposing a frozen `exp000`
  control to fork from.
- Optimizers `NorMuon` and `FusedAdamW`, plus `CompositeOptimizer` and
  its parameter selectors (`matching`, `excluding`, `complement`,
  `everything`).
- Metric `BitsPerByte`.
- Model layers `ValueGatedAttention`, `SwiGLUReluSquared`, `SoftCap`,
  `ResidualMix`, and `NarrowEmbedding`; the `unit_fan_in_uniform`
  initializer; and the `TensorModule` protocol.
- Reinforcement-learning building blocks: `priml.math.advantage`
  (`generalized_advantage`, `q_lambda_targets`, `explained_variance`),
  `priml.loss.policy_gradient`, and the `priml.data.environment`
  protocols.
- `priml.timer.CheckpointableStepTimer` for training-time accounting, and
  explicit PhaseTimer intervals for progress reports and fault dumps.
- GPU data augmentation (`priml.data.augmentation_gpu`) and
  `priml.math.activations.relu_squared`.
- Gradient clipping (`priml.train.grad_clip`).
- Optional numpy recovery and variance correction in seeding.

## 0.1.2 - 2026-08-01

### Changed

- Requires configgle 1.3.5 or newer.
- README leads with a Quick Start and carries a one-line description.

## 0.1.1 - 2026-08-01

### Changed

- README leads with a Quick Start; the duplicate Install section is folded
  into it, with `uv add` first and pip named as the alternative.

- Initial public release of priml: malleable ML building blocks for training
  experiments (models, optimizers, losses, metrics, math, training loop, and
  data pipeline).
