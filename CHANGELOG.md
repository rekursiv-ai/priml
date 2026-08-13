# Changelog

All notable priml changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Path helpers moved to `priml.paths`. `runtime_output_path` is renamed
  `validated_output_path` and gained an optional `protected` argument that
  refuses a destination aliasing one of the run's own inputs.
- `resolve_working_dir` now lives in `priml.paths` (previously vendored).

## 0.1.4 - 2026-08-05

### Changed

- Requires configgle 1.3.7 or newer.
- `runtime_output_path` expands `~` and is canonicalized; the Git-checkout
  guard is removed.
- Distributed topology is validated before resources are acquired.
- Normalization layer widths are inferred rather than declared.

### Added

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
