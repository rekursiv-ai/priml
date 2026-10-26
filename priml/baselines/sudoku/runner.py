"""Plumbing shared by the evaluation runners.

Checkpoint loading (:func:`load_eval_weights`, shared by every eval-side
loader) accepts BOTH schemas: full training checkpoints
(``state["step"]["model"]`` plus the EMA state ``state["step"]["ema"]``, whose
``shadow_params`` hold the name-keyed shadow) and HF model-only files
(top-level ``state["model"]``). Eval weights are ALWAYS the EMA weights on the
recipe (EMA warmup 0), so a full checkpoint's shadow overlays the live
parameters exactly like the train step's eval-time EMA swap; model-only HF files
are exported EMA-applied already and load as-is.

Outputs follow the run-dir convention: metrics JSON and npz dumps land under
``<scratch>/runs/{experiment_name}/dumps/``, resolved by
:func:`fill_template` and written atomically.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import hashlib
import json
import os

from torch import Tensor
from treekle import from_plain

import numpy as np
import torch

from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.lib.npyio import NpzFile

    from priml.baselines.sudoku.latent_recursion import LatentRecursion
    from priml.baselines.sudoku.puzzle_data import PuzzleBatch


def load_eval_weights(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Load eval-ready model weights from either checkpoint schema.

    The shared eval-side loader. Accepts BOTH schemas:

    * full training checkpoints -- ``state["step"]["model"]`` plus, once the
      run's EMA has a shadow, ``state["step"]["ema"]["shadow_params"]``;
    * HF model-only files -- top-level ``state["model"]``.

    Eval weights are ALWAYS the EMA weights on the recipe (EMA warmup 0):
    when a full checkpoint carries a shadow, the shadow OVERLAYS the live
    parameters -- exactly the train step's eval-time ``EMA.apply_to`` swap
    (buffers are never shadowed and come from the model state). Model-only
    HF files are exported EMA-applied already, so they load as-is.

    Args:
      path: Checkpoint file in either schema.
      device: ``map_location`` for the loaded tensors.

    Returns:
      weights: Name-keyed state dict ready for ``LatentRecursion.load_state_dict``.

    """
    step = checkpoint_payload(Path(path), device)
    weights = dict(step["model"])
    weights.update(step.get("ema", {}))  # Eval weights are the EMA shadow.
    return weights


def checkpoint_payload(path: Path, device: torch.device | str) -> _Weights:
    """Load a checkpoint; return its nested full-state or top-level payload.

    Args:
      path: Checkpoint file in either schema.
      device: ``map_location`` for the loaded tensors.

    Returns:
      weights: The ``model`` state and, when present, the EMA shadow as ``ema``.

    """
    state = cast(
        dict[str, object],
        torch.load(path, map_location=device, weights_only=True),
    )
    nested = state.get("step")
    payload = from_plain(nested, dict[str, object]) if nested is not None else state
    weights: _Weights = {"model": from_plain(payload["model"], dict[str, Tensor])}
    # An EMA that has not yet updated, or no EMA at all, saves only its counters.
    ema = from_plain(payload.get("ema", {}), dict[str, object])
    if "shadow_params" in ema:
        weights["ema"] = from_plain(ema["shadow_params"], dict[str, Tensor])
    return weights


def eval_model(
    config: LatentRecursion.Config,
    checkpoint_path: Path,
    device: torch.device,
) -> LatentRecursion:
    """Build a frozen TRM carrying a checkpoint's eval (EMA) weights."""
    model = config.make()
    model.to(device)
    model.load_state_dict(load_eval_weights(checkpoint_path, device=device))
    model.eval()
    return model


# Every full-set engine shares this: the dataset silently clips a too-large instance
# cap to the split, so an unchecked pass would publish metrics over fewer puzzles than
# configured.
def require_population(evaluated: int, expected: int, *, label: str) -> None:
    """Raise unless a pass evaluated exactly the configured population.

    Args:
      evaluated: Puzzles the pass scored.
      expected: The configured population.
      label: Pass name in the error.

    Raises:
      RuntimeError: The counts differ.

    """
    if evaluated != expected:
        raise RuntimeError(
            f"{label} evaluated {evaluated} puzzles; the configured population "
            f"is {expected} (eval_instance_indices if set, else "
            "evaluation_count).",
        )


# A ``Path`` template is a literal override kept verbatim; a ``str`` template has its
# ``{experiment_name}`` (and any ``extra``) fields filled, then the resulting logical
# path is resolved beneath ``base_dir`` (the scratch root).
def fill_template(
    template: str | Path,
    *,
    base_dir: Path | str | None,
    experiment_name: str,
    **extra: object,
) -> Path:
    """Resolve a run-output path beneath ``base_dir``."""
    if isinstance(template, Path):
        return template
    filled = template.format(experiment_name=experiment_name, **extra)
    return Path(resolve_working_dir(base_dir, filled))


def prepared(path: Path) -> Path:
    """Create the parent directory of an output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# A crash mid-write must never tear the pipeline's aggregate metrics: the destination
# either keeps its previous content or holds the full new one.
def write_json(path: Path, metrics: Mapping[str, object]) -> None:
    """Write the aggregate metrics JSON via tmp + atomic rename."""
    destination = prepared(path)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


# No ``gc.collect()``. One was here, on the theory that a reference cycle might hold
# tensors ``empty_cache`` could then reclaim. Instrumenting the suite measured what it
# actually bought: 77 collections freed 8,887 cycle objects and reclaimed ZERO bytes of
# device memory, because ``memory_allocated()`` was 0 at every one -- the eval path
# builds its models on CPU, so the allocator it was clearing had never allocated.
#
# The cost was not zero. A collection is linear in the live heap, and the heap here is
# every module the tier imported: removing it took this file's slow tier from 21.3s to
# 10.9s on a 128-core Linux host, and the whole file's 25 tests from 22.4s to 10.9s.
#
# Guarding it on ``is_initialized()`` rather than deleting it was measured too, and does
# NOT work: something in the eval path touches the device, so the flag reads True at 71
# of 77 call sites while the allocator still holds nothing. The guard looks right and is
# inert.
#
# ``empty_cache`` alone stays. It is cheap when there is no allocator and correct when a
# caller does run on GPU; only the collection was speculative.
def release() -> None:
    """Return the CUDA caching allocator's free blocks between passes/rounds."""
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def to_device(batch: PuzzleBatch, device: torch.device) -> PuzzleBatch:
    """Move a batch's tensors to ``device``.

    Args:
      batch: One dataset batch.
      device: Destination device.

    Returns:
      batch: The same keys, tensors on ``device``.

    """
    return {
        "media": batch["media"].to(device),
        "label": batch["label"].to(device),
        "valid_count": batch["valid_count"],
        "puzzle_identifiers": batch["puzzle_identifiers"].to(device),
    }


def open_npz(path: Path) -> NpzFile:
    """Open an npz archive; ``np.load`` itself returns ``Any``."""
    return cast("NpzFile", np.load(path))


def file_sha256(path: Path) -> str:
    """Hash one completed immutable shard."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(partial(file.read, 1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Weights(TypedDict):
    """One checkpoint's model payload, in either schema."""

    model: dict[str, Tensor]
    ema: NotRequired[dict[str, Tensor]]
