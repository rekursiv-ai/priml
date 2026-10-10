"""Input pipeline throughput: images per CPU-second, with the batches pinned.

:class:`LoaderThroughput` drains a training input pipeline, times it, and
compares every batch it produced, tensor by tensor, against the batches the
reference pipeline makes from the same image set under the same seed. A
pipeline that is faster but changes one pixel or one label is a different
recipe, not a faster one, so a mismatch fails the run instead of reporting a
number, and says which batch and field moved, how many values, by how much,
and where the first one is.

The reference drains once per run, in its own untimed process, and its
batches go to a scratch directory (about 40 MB per 512-image batch, so
roughly 650 MB for exp000's set). What it computes is pinned in CI by the
tensor golden in ``throughput_test.py``, which a change to shared decode or
crop code fails; a fork that changes the reference to match itself is caught
there, not here.

Each pass runs in a fresh spawned process, so nothing a pipeline caches in
memory survives into the next pass, and the reference is never computed in a
timed process. The window opens before the pipeline's config is unpickled
(so its modules' import-time work counts) and closes when the last batch is
in hand; construction and the first batch are inside it.

CPU time is read by this process with ``getrusage(RUSAGE_CHILDREN)`` around
starting and joining the pass process. It covers every thread of the pass
process and every descendant process that was waited for before its parent
exited, which is how ``multiprocessing``, process pools, and
``subprocess.run`` end. The pass process reports, and this process subtracts,
its own CPU before the window opened (interpreter start, harness imports) and
the main thread's CPU spent comparing after it closed. Not counted: a
descendant nobody waits for. The pass process leads its own process group,
and a run whose group still has a live process after the pass refuses to
report; a descendant that leaves the group, or exits without being waited
for, is beyond what ``getrusage`` can see, and the rules forbid both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import logging
import multiprocessing
import os
import pickle
import resource
import statistics
import tempfile
import time
import traceback

from configgle import Fig, Makeable
from torch import Tensor

import numpy as np
import torch

from priml.baselines.imagenet.data import ffcv_train_data_pipeline
from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.data.pipeline.dataset import DataPipeline
from priml.math.seed import set_seed_local
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from multiprocessing.connection import Connection


logger = logging.getLogger(__name__)


__all__ = ["LoaderThroughput", "PixelTolerance", "ThroughputReport"]


@dataclass(frozen=True)
class ThroughputReport:
    """What one :meth:`LoaderThroughput.measure` call observed."""

    images_per_cpu_sec: list[float]
    """The score: images over the CPU time of the pass's whole process tree,
    one per pass. Adding threads or processes alone cannot raise it."""

    images_per_sec: list[float]
    """Wall-clock rate per pass; reported, not scored."""

    first_batch_sec: list[float]
    """Seconds from the window opening to the first batch, per pass."""

    num_images: int
    """Images in each pass, every batch counted."""

    mismatches: list[str]
    """Every batch field outside the correctness tier; empty when it holds."""

    max_abs_diff: float
    """Largest per-value difference from the reference across every pass; 0
    under the exact tier, where differing batches are already mismatches."""

    max_fraction_differing: float
    """Largest fraction of one batch field's values that differ."""


class PixelTolerance:
    """How far a pipeline's images may drift from the reference's.

    Opting in admits recipes that change pixels (``scale_to_target``, the
    accurate IDCT); a run under a tolerance says so in its log line.
    """

    class Config(Fig["PixelTolerance"]):
        max_abs_diff: float = 0.0
        """Largest difference allowed in any one value, in pixel levels."""

        max_fraction_differing: float = 0.0
        """Largest fraction of one batch's values allowed to differ at all."""

        fields: list[str] = field(default_factory=lambda: ["image"])
        """Keys the tolerance applies to; every other compared key stays exact."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def admits(self, max_abs_diff: float, fraction_differing: float) -> bool:
        """Return whether a batch field this far from the reference passes."""
        return (
            max_abs_diff <= self.config.max_abs_diff
            and fraction_differing <= self.config.max_fraction_differing
        )


class LoaderThroughput:
    """Time ``pipeline`` and require its batches to match ``reference``'s."""

    class Config(Fig["LoaderThroughput"]):
        pipeline: Makeable[DataPipeline] = field(
            default_factory=ffcv_train_data_pipeline,
        )
        """The pipeline being timed; a fork changes this and nothing else."""

        reference: Makeable[DataPipeline] = field(
            default_factory=ffcv_train_data_pipeline,
        )
        """The pipeline whose batches ``pipeline`` must reproduce, drained once
        per run in an untimed process of its own."""

        tolerance: Makeable[PixelTolerance] | None = None
        """None compares every field bit for bit."""

        fields: list[str] = field(default_factory=lambda: ["image", "label"])
        """Batch keys compared against the reference: what the train step reads.
        The first one's leading dimension counts the images."""

        seed: int = 0
        """Reseeds every pass, so crop and flip draws repeat bit for bit."""

        num_repeats: int = 5
        """Timed passes over the data; the report keeps every one."""

        survivor_grace_sec: float = 5.0
        """How long a pass's process group may take to empty after the pass exits."""

        base_dir: Path | str | None = "/opt/scratch"
        """Resource root the pipelines resolve beneath."""

        working_dir: Path | str = "/"
        """Transparent scope: each pipeline's source owns its dataset root."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            for pipeline in (self.pipeline, self.reference):
                if (
                    isinstance(pipeline, HasNormalizedWorkingDirPattern)
                    and pipeline.base_dir is None
                ):
                    pipeline.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.num_repeats <= 0:
            raise ValueError(f"num_repeats must be positive; got {config.num_repeats}.")
        self.config = config
        self.tolerance = None if config.tolerance is None else config.tolerance.make()

    def run(self, *args: str) -> None:
        """Measure, log the rates, and fail if any batch left its tier.

        Args:
          *args: Ignored; accepted so launcher passthrough arguments never raise.

        Raises:
          AssertionError: A batch is outside the correctness tier.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        report = self.measure()
        if report.mismatches:
            raise AssertionError(
                f"{len(report.mismatches)} batch mismatches against the reference:\n"
                + "\n".join(report.mismatches),
            )
        if self.tolerance is None:
            tier = "batches bit-identical to the reference"
        else:
            tol = self.tolerance.config
            tier = (
                f"UNDER TOLERANCE max_abs_diff={tol.max_abs_diff:g} "
                f"max_fraction_differing={tol.max_fraction_differing:g} on "
                f"{tol.fields} (observed {report.max_abs_diff:g} and "
                f"{report.max_fraction_differing:.4%}), not exact"
            )
        for unit, rates in (
            ("images/cpu-sec (scored)", report.images_per_cpu_sec),
            ("images/sec (wall, not scored)", report.images_per_sec),
            ("first-batch sec", report.first_batch_sec),
        ):
            logger.info(
                "%s over %d passes of %d images: median %.1f, mean %.1f, "
                "min %.1f, max %.1f; %s.",
                unit,
                len(rates),
                report.num_images,
                statistics.median(rates),
                statistics.fmean(rates),
                min(rates),
                max(rates),
                tier,
            )

    def measure(self) -> ThroughputReport:
        """Drain ``reference`` once, then time ``num_repeats`` fresh-process passes.

        Returns:
          report: Per-pass rates and every batch outside the correctness tier.

        Raises:
          ValueError: A pass yielded no batch.

        """
        cfg = self.config
        with tempfile.TemporaryDirectory() as scratch:
            _ = _run_pass(cfg.reference, cfg, save_dir=scratch)
            passes = [
                _run_pass(cfg.pipeline, cfg, compare_dir=scratch)
                for _ in range(cfg.num_repeats)
            ]
        mismatches: list[str] = []
        worst_diff, worst_fraction = 0.0, 0.0
        for repeat, (result, _) in enumerate(passes):
            if result.num_images == 0:
                raise ValueError(f"Pass {repeat} yielded no batch.")
            lines, diff, fraction = self._check(result, repeat)
            mismatches += lines
            worst_diff, worst_fraction = (
                max(worst_diff, diff),
                max(worst_fraction, fraction),
            )
        return ThroughputReport(
            images_per_cpu_sec=[r.num_images / cpu for r, cpu in passes],
            images_per_sec=[r.num_images / r.elapsed_sec for r, _ in passes],
            first_batch_sec=[r.first_batch_sec for r, _ in passes],
            num_images=passes[-1][0].num_images,
            mismatches=mismatches,
            max_abs_diff=worst_diff,
            max_fraction_differing=worst_fraction,
        )

    def _check(self, result: _Pass, repeat: int) -> tuple[list[str], float, float]:
        """Return one line per field outside its tier, and the worst drift."""
        lines: list[str] = []
        if result.num_batches != result.num_reference_batches:
            lines.append(
                f"pass {repeat}: {result.num_batches} batches vs "
                f"{result.num_reference_batches}",
            )
        worst_diff, worst_fraction = 0.0, 0.0
        for (index, name), diff in result.diffs.items():
            where = f"pass {repeat} batch {index} {name}"
            if isinstance(diff, str):
                lines.append(f"{where}: {diff}")
                continue
            fraction = diff.num_differing / diff.numel
            if self.tolerance is not None and name in self.tolerance.config.fields:
                worst_diff = max(worst_diff, diff.max_abs_diff)
                worst_fraction = max(worst_fraction, fraction)
                if self.tolerance.admits(diff.max_abs_diff, fraction):
                    continue
            lines.append(
                f"{where}: {diff.num_differing}/{diff.numel} differ, max diff "
                f"{diff.max_abs_diff:g}, first at {list(diff.first_index)}",
            )
        return lines, worst_diff, worst_fraction


@dataclass(frozen=True)
class _Diff:
    """How one batch field differs from the reference's."""

    num_differing: int
    numel: int
    max_abs_diff: float
    first_index: tuple[int, ...]


@dataclass(frozen=True)
class _Pass:
    """What one pass process sends back."""

    diffs: dict[tuple[int, str], _Diff | str]
    """Each (batch, field) that differs from the reference: how, or why the
    two cannot be compared (dtype or shape)."""
    num_batches: int
    num_reference_batches: int
    num_images: int
    elapsed_sec: float
    first_batch_sec: float
    cpu_outside_sec: float
    """The pass process's own CPU outside the window, which is not scored."""


def _run_pass(
    pipeline: Makeable[DataPipeline],
    cfg: LoaderThroughput.Config,
    *,
    save_dir: str | None = None,
    compare_dir: str | None = None,
) -> tuple[_Pass, float]:
    """Drain ``pipeline`` once in a fresh process; return it and its CPU sec."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_pass,
        # Pickled here and unpickled inside the window, so whatever the
        # pipeline's own modules do at import is timed.
        args=(pickle.dumps(pipeline),),
        kwargs={
            "seed": cfg.seed,
            "fields": cfg.fields,
            "save_dir": save_dir,
            "compare_dir": compare_dir,
            "connection": sender,
        },
    )
    started_cpu = _children_cpu_sec()
    process.start()
    sender.close()
    try:
        # A _Pass, or the pass's traceback if it raised.
        result = cast("_Pass | str", receiver.recv())
    except EOFError:
        result = "the pass process exited without a result"
    process.join()
    cpu = _children_cpu_sec() - started_cpu
    if isinstance(result, str):
        raise RuntimeError(  # noqa: TRY004 -- The string is the pass's traceback, not a caller passing the wrong type.
            f"Pass failed (exit code {process.exitcode}):\n{result}",
        )
    assert process.pid is not None
    _require_empty_group(process.pid, cfg.survivor_grace_sec)
    return result, cpu - result.cpu_outside_sec


def _pass(
    pipeline: bytes,
    *,
    seed: int,
    fields: list[str],
    save_dir: str | None,
    compare_dir: str | None,
    connection: Connection,
) -> None:
    """Run in a fresh process: drain one pass, then compare it outside the window."""
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
        diffs: dict[tuple[int, str], _Diff | str] = {}
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
                    diff = _diff(want[name], batch[name])
                    if diff is not None:
                        diffs[index, name] = diff
        result = _Pass(
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


def _diff(want: Tensor, got: Tensor) -> _Diff | str | None:
    """Return how ``got`` differs from ``want``, or None when the bits match."""
    if (want.dtype, want.shape) != (got.dtype, got.shape):
        return f"{got.dtype}{list(got.shape)} vs {want.dtype}{list(want.shape)}"
    # numpy, not torch: its elementwise ops stay on this thread, whose CPU is
    # subtracted. Compared as bytes so a NaN equals itself; float64 holds every
    # difference of two values of the integer dtypes a batch carries exactly.
    a, b = want.contiguous().numpy(), got.contiguous().numpy()
    if a.tobytes() == b.tobytes():
        return None
    differing = a != b  # pyright: ignore[reportAny] -- numpy comparisons are dtype-erased.
    where = np.argwhere(differing)  # pyright: ignore[reportAny] -- numpy indexing is dtype-erased.
    return _Diff(
        num_differing=int(np.count_nonzero(differing)),  # pyright: ignore[reportAny] -- numpy comparisons are dtype-erased.
        numel=a.size,
        max_abs_diff=float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),  # pyright: ignore[reportAny] -- numpy reductions are dtype-erased.
        # Empty when only the bytes differ (a signed zero): the count says 0.
        first_index=tuple(int(i) for i in where[0]) if len(where) else (),  # pyright: ignore[reportAny] -- numpy indexing is dtype-erased.
    )


def _require_empty_group(group: int, grace_sec: float) -> None:
    """Raise if any process of the pass's group outlives ``grace_sec``."""
    deadline = time.monotonic() + grace_sec
    while True:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                "A process the pipeline started outlived its pass; its CPU "
                "time cannot be counted, so no rate is reported.",
            )
        time.sleep(0.05)


def _children_cpu_sec() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _self_cpu_sec() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


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
