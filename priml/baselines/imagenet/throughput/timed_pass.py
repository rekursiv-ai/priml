"""One timed pass of a pipeline, run by :mod:`throughput` in a fresh process.

This module imports no pipeline code. A spawned pass imports it before its
clock starts and subtracts that CPU, so a pipeline module imported here would
load for free in ``exp000``'s pass while a fork bringing its own modules paid
for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import os
import pickle
import resource
import time
import traceback

from torch import Tensor

import numpy as np
import torch

from priml.math.seed import set_seed_local


if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from configgle import Makeable
    from numpy.typing import NDArray

    from priml.data.pipeline.dataset import DataPipeline


__all__ = ["Diff", "PassResult", "compare", "drain"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Diff:
    """How one batch field differs from the reference's."""

    num_differing: int
    numel: int
    max_abs_diff: float
    first_index: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PassResult:
    """What one pass process sends back."""

    diffs: dict[tuple[int, str], Diff | str]
    """Each (batch, field) that differs from the reference: how, or why the
    two cannot be compared (dtype or shape)."""

    num_batches: int
    num_reference_batches: int
    num_images: int
    elapsed_sec: float
    first_batch_sec: float
    cpu_outside_sec: float
    """The pass process's CPU before the clock started, plus its main
    thread's CPU spent comparing; not scored."""


def drain(
    pipeline: bytes,
    *,
    seed: int,
    fields: list[str],
    save_dir: str | None,
    compare_dir: str | None,
    connection: Connection,
) -> None:
    """Drain one pass on the clock, then compare it off it.

    Sends a :class:`PassResult` on ``connection``, or the traceback if the
    pass raised. What runs after the send, freeing the batches and
    interpreter teardown, is charged to the pipeline.

    Args:
      pipeline: The pickled pipeline config, unpickled on the clock.
      seed: Seed set before the pipeline is built.
      fields: Batch keys kept and compared; the first counts the images.
      save_dir: Where to save every batch, for the reference pass.
      compare_dir: Where the reference's batches are, for a timed pass.
      connection: Where the result goes.

    """
    os.setpgid(0, 0)
    try:
        cpu_before = _self_cpu_sec()
        started = time.perf_counter()
        first_batch_sec = 0.0
        batches: list[dict[str, Tensor]] = []
        set_seed_local(seed)
        made = cast("Makeable[DataPipeline]", pickle.loads(pipeline))  # noqa: S301 -- Pickled by the parent.
        for batch in made.make():
            batches.append(_kept(batch, fields))
            if len(batches) == 1:
                first_batch_sec = time.perf_counter() - started
        elapsed = time.perf_counter() - started

        # Only this thread's CPU from here on is subtracted: a pipeline thread
        # still running is the pipeline's cost.
        comparing_started = time.thread_time()
        diffs: dict[tuple[int, str], Diff | str] = {}
        num_reference_batches = 0
        for index, batch in enumerate(batches):
            if save_dir is not None:
                torch.save(batch, Path(save_dir) / f"{index}.pt")
        if compare_dir is not None:
            num_reference_batches = len(list(Path(compare_dir).glob("*.pt")))
            for index, batch in enumerate(batches[:num_reference_batches]):
                want = cast(
                    "dict[str, Tensor]",
                    torch.load(Path(compare_dir) / f"{index}.pt", weights_only=True),
                )
                for name in fields:
                    diff = compare(want[name], batch[name])
                    if diff is not None:
                        diffs[index, name] = diff
        result = PassResult(
            diffs=diffs,
            num_batches=len(batches),
            num_reference_batches=num_reference_batches,
            num_images=sum(len(b[fields[0]]) for b in batches),
            elapsed_sec=elapsed,
            first_batch_sec=first_batch_sec,
            cpu_outside_sec=cpu_before + time.thread_time() - comparing_started,
        )
    except BaseException:
        connection.send(traceback.format_exc())
        raise
    connection.send(result)


def compare(want: Tensor, got: Tensor) -> Diff | str | None:
    """Return how ``got`` differs from ``want``, or None when the bits match.

    Args:
      want: The reference's field.
      got: The timed pipeline's field.

    Returns:
      diff: How the values differ; why they cannot be compared (dtype or
        shape); or None when every bit matches.

    """
    if (want.dtype, want.shape) != (got.dtype, got.shape):
        return f"{got.dtype}{list(got.shape)} vs {want.dtype}{list(want.shape)}"
    # `numpy`, not torch: its elementwise ops stay on this thread, whose CPU is
    # subtracted. Bit patterns, so a NaN equals itself and -0.0 differs from 0.0.
    differing = cast("NDArray[np.bool_]", _bits(want) != _bits(got))
    if not differing.any():
        return None
    first = cast("list[int]", np.argwhere(differing)[0].tolist())  # pyright: ignore[reportAny] -- numpy types `ndarray.tolist` as Any; the cast is the boundary.
    return Diff(
        num_differing=int(np.count_nonzero(differing)),
        numel=want.numel(),
        max_abs_diff=_max_abs_diff(want, got, differing),
        first_index=tuple(first),
    )


def _bits(tensor: Tensor) -> NDArray[np.unsignedinteger]:
    """Return ``tensor``'s elements as same-width unsigned integers."""
    unsigned = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32, 8: torch.uint64}
    return tensor.contiguous().view(unsigned[tensor.element_size()]).numpy()


# Floats subtract in float64, which holds every float32 and bfloat16 value. Integers
# subtract exactly, larger minus smaller in uint64: the true gap fits, so arithmetic
# modulo 2**64 gives it even for int64 extremes.
def _max_abs_diff(want: Tensor, got: Tensor, differing: NDArray[np.bool_]) -> float:
    """Return the largest ``|want - got|`` where they differ; inf where one is NaN."""
    if want.dtype.is_floating_point:
        moved = np.abs(_floats(want)[differing] - _floats(got)[differing])
        return float(
            cast("np.float64", np.nan_to_num(moved, nan=np.inf, posinf=np.inf).max()),
        )
    wanted = want.contiguous().numpy()[differing]
    gotten = got.contiguous().numpy()[differing]
    high = np.maximum(wanted, gotten).astype(np.uint64)
    low = np.minimum(wanted, gotten).astype(np.uint64)
    return float(cast("np.uint64", (high - low).max()))


def _floats(tensor: Tensor) -> NDArray[np.float64]:
    """Return a floating ``tensor``'s elements as float64, exactly."""
    if tensor.dtype == torch.bfloat16:
        # `numpy` has no bfloat16; its bits are a float32's top half.
        return (
            (_bits(tensor).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        )
    return tensor.contiguous().numpy().astype(np.float64)


def _kept(batch: dict[str, object], fields: list[str]) -> dict[str, Tensor]:
    """Return the compared fields of ``batch``."""
    kept: dict[str, Tensor] = {}
    for name in fields:
        value = batch[name]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Batch field {name!r} is {type(value).__name__}, not a Tensor.",
            )
        kept[name] = value
    return kept


def _self_cpu_sec() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime
