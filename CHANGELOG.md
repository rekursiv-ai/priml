# Changelog

All notable priml changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

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
  a `phase_heartbeat_sec` setting for long-phase progress logging.
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
