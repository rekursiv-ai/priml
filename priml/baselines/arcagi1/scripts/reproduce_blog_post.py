#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Reproduce the ARC-AGI-1 blog-post experiment with public Priml.

The blog post's model is exp008: QK-norm, prediction feedback, and
corrupted-feedback repair over the URM. From the Priml repository
root, set up the environment and exp008 data:
  uv sync --all-groups
  priml/baselines/arcagi1/scripts/prepare_data.py --experiment exp008

Train from scratch on eight visible H200 GPUs and score at step 280,000:
  uv run torchrun --standalone --nproc_per_node=8 \
    priml/baselines/arcagi1/scripts/reproduce_blog_post.py
On successful completion, Priml scores first, then writes a distributed
checkpoint directory at
  /opt/scratch/runs/arcagi1/exp008_blog_8gpu/checkpoints/step_00280000.pt/

To evaluate the recovered internal 8xH100 checkpoint instead, run with
  --checkpoint /path/to/historical-hps-step280000.pt
This option accepts only the SHA-pinned historical archive, not a new Priml
checkpoint produced by the training command above.

To train on one GX10/GB10 (128 GB shared memory):
  priml/baselines/arcagi1/scripts/reproduce_blog_post.py
Its final checkpoint is the native file
  /opt/scratch/runs/arcagi1/exp008_blog_gx10/checkpoints/step_00280000.pt
It uses one GPU and batch 8 instead of 8x96; at 280,000 steps it sees 96x
fewer examples, so its score is not like-for-like. Neither a full GX10 run
nor this launcher's eight-H200 run has been completed.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import hashlib
import logging
import math
import os

from priml.baselines.arcagi1.experiments import TrmTrainLoop, exp008
from priml.baselines.arcagi1.model import from_reference_name
from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import FileTracker


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


def main() -> int:
    """Train and score a new model, or score the historical archive.

    Returns:
      status: Process exit code.

    """
    if __doc__ is None:
        raise ValueError("Expected __doc__ is not None.")
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (world_size, torch.cuda.device_count()) not in {(1, 1), (8, 8)}:
        raise SystemExit("Use one CUDA GPU or torchrun with eight CUDA GPUs")
    cfg = recipe(single_gpu=world_size == 1, checkpoint=flags.checkpoint is not None)
    logging.basicConfig(level=logging.INFO)
    sha256 = "64bd795f8dc984d7e647be155afc098bb41c7a8c22daeac36f57c4ad76fc1cff"
    if flags.checkpoint is not None:
        with flags.checkpoint.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != sha256:
                raise SystemExit(f"Historical checkpoint SHA256 must be {sha256}")
    loop = cfg.make()
    if flags.checkpoint is not None:
        archive = cast(
            "dict[str, object]",
            torch.load(flags.checkpoint, "cpu", weights_only=True),
        )
        loop.load_state_dict(overlay(archive, dict(loop.state_dict())))
    loop.run()
    return 0


def recipe(
    *,
    single_gpu: bool,
    checkpoint: bool,
    steps: int = 280_000,
) -> TrmTrainLoop:
    """Set the same model recipe for distributed or single-GPU execution.

    Args:
      single_gpu: One GB10 at batch 8 rather than eight GPUs at 96 each.
      checkpoint: Score a loaded archive instead of training.
      steps: Training horizon; the blog post scored at 280,000.

    Returns:
      cfg: exp008 with this launcher's runtime, tracker, and stopping point.

    """
    cfg = exp008()
    cfg.experiment_name = "exp008_blog_gx10" if single_gpu else "exp008_blog_8gpu"
    cfg.max_steps = steps
    cfg.max_time = math.inf
    cfg.num_steps_eval = -1  # Final scoring uses the trained EMA in memory.
    cfg.eval_warmup_batches = 0
    cfg.tracker = FileTracker.Config()
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.resume = False
    # Keep exp008's model and optimizer; only adapt runtime and batch to GB10.
    if single_gpu:
        cfg.runtime = SingleProcess.Config(device="cuda")
        cfg.step.parallelism = NoParallel.Config(device="cuda")
        assert isinstance(cfg.step.pool, AtomicPool.Config)
        assert isinstance(cfg.step.model.prefix, SparsePuzzleEmbedding.Config)
        cfg.step.pool.batch_size = cfg.dataset.batch_size = 8
        cfg.step.model.prefix.batch_size = 8
        cfg.dataset.eval_batch_size = 32
    if checkpoint:
        cfg.experiment_name += "_historical_eval"
        cfg.eval_only = True
        cfg.checkpointer = None
    return cfg


def overlay(
    archive: dict[str, object],
    into: dict[str, object],
    *,
    steps: int = 280_000,
) -> dict[str, object]:
    """Map the historical model and EMA onto a fresh eval-only Priml state.

    Args:
      archive: The historical checkpoint.
      into: A fresh loop's ``state_dict``, updated in place.
      steps: Step count the archive was saved at.

    Returns:
      into: The updated state.

    """
    old = cast("dict[str, object]", archive["step"])
    fresh = cast("dict[str, object]", into["step"])
    source = cast("dict[str, torch.Tensor]", old["model"])
    old_ema = cast("dict[str, torch.Tensor]", old["ema"])
    fresh["model"] = {from_reference_name(k): v for k, v in source.items()}
    ema = {from_reference_name(k): v for k, v in old_ema.items()}
    fresh["ema"] = {"shadow_params": ema, "global_step": steps}
    fresh["timer_step"] = {"global_count": steps, "global_sec": 0.0}
    return into


class _Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="SHA-pinned historical archive",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
