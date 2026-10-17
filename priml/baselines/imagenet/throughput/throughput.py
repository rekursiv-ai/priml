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

Each pass runs :func:`timed_pass.drain` in a fresh spawned process, so
nothing a pipeline caches in memory survives into the next pass, and the
reference is never computed in a timed process. The wall clock
(``images_per_sec``, ``first_batch_sec``) starts before the pipeline's config
is unpickled and stops when the last batch is in hand; construction and the
first batch are inside it. ``timed_pass`` imports no pipeline code, so every
pipeline, ``exp000``'s included, pays for its own modules' import.

CPU time is read by this process with ``getrusage(RUSAGE_CHILDREN)`` around
starting the pass process and reaping its process tree, so it spans the pass
process's whole life. It covers every thread of the pass process and every
descendant: one waited for by its parent, as ``multiprocessing``, process
pools, and ``subprocess.run`` do, and one orphaned, which this process (a
child subreaper) adopts and reaps. The pass process reports, and this
process subtracts, two spans of its own: its CPU before the clock started
(interpreter start, harness imports) and its main thread's CPU spent
comparing. Everything else is charged, sending the result, freeing the
batches, and interpreter teardown included, so a pipeline whose shutdown is
costly pays for it.

The pass process leads its own process group. After it reports, it has
``survivor_grace_sec`` to exit, and its group as long again to empty; past
either, the group is killed and the run refuses to report. A descendant that
leaves the group is beyond this check, and the rules forbid it. A pass whose
process exits nonzero is a failure even if it sent a result. If this process
is interrupted mid-pass, it kills the pass's group before unwinding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast, override

import contextlib
import ctypes
import logging
import math
import multiprocessing
import os
import pickle
import resource
import signal
import statistics
import tempfile
import time

from configgle import Fig, Makeable

from priml.baselines.imagenet.data import ffcv_train_data_pipeline
from priml.baselines.imagenet.throughput import timed_pass
from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.data.pipeline.dataset import DataPipeline
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from priml.baselines.imagenet.throughput.timed_pass import PassResult


logger = logging.getLogger(__name__)


__all__ = ["LoaderThroughput", "PixelTolerance", "ThroughputReport"]


POLL_SEC: Final = 0.05
"""How often a wait on the pass process rechecks that it is alive."""

NO_RESULT: Final = "the pass process exited without a result"

PR_SET_CHILD_SUBREAPER: Final = 36
"""From ``<linux/prctl.h>``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ThroughputReport:
    """What one :meth:`LoaderThroughput.measure` call observed."""

    images_per_cpu_sec: list[float]
    """The score: images over the CPU time of the pass's whole process tree,
    one per pass. Adding threads or processes alone cannot raise it."""

    images_per_sec: list[float]
    """Wall-clock rate per pass; reported, not scored."""

    first_batch_sec: list[float]
    """Seconds from the clock starting to the first batch, per pass."""

    num_images: int
    """Images in each pass, every batch counted."""

    mismatches: list[str]
    """Every batch field outside the correctness tier; empty when it holds."""

    max_abs_diff: float
    """Largest per-value difference from the reference in a tolerance field,
    across every pass; inf for a NaN on one side. 0 under the exact tier,
    where differing batches are already mismatches."""

    max_fraction_differing: float
    """Largest fraction of one tolerance field's values that differ in one
    batch; 0 under the exact tier."""


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
        if math.isnan(config.max_abs_diff) or config.max_abs_diff < 0:
            raise ValueError(
                f"max_abs_diff must be a nonnegative number; got {config.max_abs_diff}.",
            )
        fraction = config.max_fraction_differing
        if math.isnan(fraction) or fraction < 0 or fraction > 1:
            raise ValueError(
                f"max_fraction_differing must lie in [0, 1]; got {fraction}.",
            )
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
        """How long the pass process may take to exit after it reports, and
        its process group to empty after that."""

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
        if not config.fields:
            raise ValueError("fields must name at least one batch key; got [].")
        self.config = config
        self.tolerance = None if config.tolerance is None else config.tolerance.make()
        if self.tolerance is not None:
            uncompared = sorted(set(self.tolerance.config.fields) - set(config.fields))
            if uncompared:
                raise ValueError(
                    f"tolerance names fields that are not compared: {uncompared}; "
                    f"compared fields are {config.fields}.",
                )

    def run(self, *args: str) -> None:
        """Measure, log the rates, and fail if any batch left its tier.

        Args:
          *args: Ignored; accepted so launcher passthrough arguments never raise.

        Raises:
          RuntimeError: A batch is outside the correctness tier, or a pass
            raised, exited nonzero, or left a process running.
          ValueError: A pass yielded no batch.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        report = self.measure()
        if report.mismatches:
            raise RuntimeError(
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
          ValueError: The reference or a pass yielded no batch.
          RuntimeError: A pass raised, exited nonzero, or left a process running.

        """
        cfg = self.config
        _become_subreaper()
        passes: list[tuple[PassResult, float]] = []
        with tempfile.TemporaryDirectory(prefix="throughput-") as scratch:
            reference, _ = _run_pass(cfg.reference, cfg, save_dir=scratch)
            if reference.num_images == 0:
                raise ValueError("The reference yielded no batch.")
            for repeat in range(cfg.num_repeats):
                passes.append(_run_pass(cfg.pipeline, cfg, compare_dir=scratch))
                if passes[-1][0].num_images == 0:
                    raise ValueError(f"Pass {repeat} yielded no batch.")
        mismatches: list[str] = []
        worst_diff, worst_fraction = 0.0, 0.0
        for repeat, (result, _) in enumerate(passes):
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

    def _check(
        self,
        result: PassResult,
        repeat: int,
    ) -> tuple[list[str], float, float]:
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


# Any exception here, ``KeyboardInterrupt`` included, kills the pass's process group and
# reaps the pass process before it propagates.
def _run_pass(
    pipeline: Makeable[DataPipeline],
    cfg: LoaderThroughput.Config,
    *,
    save_dir: str | None = None,
    compare_dir: str | None = None,
) -> tuple[PassResult, float]:
    """Drain ``pipeline`` once in a fresh process; return it and its CPU sec."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=timed_pass.drain,
        # Pickled here and unpickled on the clock, so whatever the
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
    group = process.pid
    if group is None:
        raise ValueError("Expected group is not None.")
    try:
        result = _receive(receiver, process)
        # A pass process exits only once its non-daemon children are joined.
        process.join(cfg.survivor_grace_sec)
        hung = process.exitcode is None
        if hung:
            _kill(process, group)
        survived = _group_survives(group, cfg.survivor_grace_sec) or hung
    except BaseException:
        _kill(process, group)
        raise
    finally:
        receiver.close()
    cpu = _children_cpu_sec() - started_cpu
    passed = _verdict(result, exitcode=process.exitcode, survived=survived)
    return passed, cpu - passed.cpu_outside_sec


# Liveness is rechecked between polls: a descendant still holding the sending end would
# keep ``recv`` from ever seeing the pass process die.
def _receive(receiver: Connection, process: BaseProcess) -> PassResult | str:
    """Return the pass's result or traceback, or why it sent neither."""
    alive = True
    while alive:
        alive = process.is_alive()
        if receiver.poll(POLL_SEC if alive else 0):
            try:
                return cast("PassResult | str", receiver.recv())
            except EOFError:
                break
    return NO_RESULT


def _kill(process: BaseProcess, group: int) -> None:
    """Kill the pass process and its group, then reap the pass process."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, signal.SIGKILL)
    # The pass process may not have made its group yet.
    process.kill()
    process.join()


def _verdict(
    result: PassResult | str,
    *,
    exitcode: int | None,
    survived: bool,
) -> PassResult:
    """Return ``result`` if the pass may be scored; raise every reason it may not."""
    reasons: list[str] = []
    if isinstance(result, str) or exitcode != 0:
        reasons.append(
            f"Pass failed (exit code {exitcode}):\n"
            f"{result if isinstance(result, str) else 'it sent a result first'}",
        )
    if survived:
        reasons.append(
            "A process the pipeline started outlived its pass; its CPU "
            "time cannot be counted, so no rate is reported.",
        )
    if reasons:
        raise RuntimeError("\n".join(reasons))
    return cast("PassResult", result)


def _become_subreaper() -> None:
    """Adopt every orphaned descendant, so its CPU is charged when it is reaped."""
    prctl = cast("Callable[..., int]", ctypes.CDLL(None, use_errno=True).prctl)
    if prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_CHILD_SUBREAPER): {os.strerror(errno)}")


# Must run only once the pass process is reaped: it reaps every exited member of
# ``group`` this process adopted, which would steal the pass process's exit status from
# ``multiprocessing``.
def _group_survives(group: int, grace_sec: float) -> bool:
    """Return whether the pass's group outlived ``grace_sec``, killing it if so."""
    deadline = time.monotonic() + grace_sec
    while True:
        _reap(group)
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        if time.monotonic() > deadline:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                return False
            _reap(group)
            return True
        time.sleep(POLL_SEC)


def _reap(group: int) -> None:
    """Reap every exited child of this process in ``group``, charging its CPU."""
    while True:
        try:
            pid, _ = os.waitpid(-group, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _children_cpu_sec() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime
