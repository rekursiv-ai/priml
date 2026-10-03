"""Input pipeline throughput: images per CPU-second, with the batches pinned.

:class:`LoaderThroughput` drains a training input pipeline, times it, and
checks every batch it produced against frozen SHA-256 digests of the
reference pipeline's batches on a fixed image set. A pipeline that is faster
but changes one pixel or one label is a different recipe, not a faster one,
so a mismatch fails the run instead of reporting a number.

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
the main thread's CPU spent hashing after it closed. Not counted: a
descendant nobody waits for. The pass process leads its own process group,
and a run whose group still has a live process after the pass refuses to
report; a descendant that leaves the group, or exits without being waited
for, is beyond what ``getrusage`` can see, and the rules forbid both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast, override

import hashlib
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


__all__ = [
    "LoaderThroughput",
    "PixelTolerance",
    "ReferenceDigests",
    "TensorDigest",
    "ThroughputReport",
    "digest_image_set",
]


_CWD: Final = Path(__file__).resolve().parent
SURVIVOR_GRACE_SEC: Final = 5.0
"""How long a pass's process group may take to empty after the pass exits."""


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


@dataclass(frozen=True)
class TensorDigest:
    """One tensor, pinned by dtype, shape, and the SHA-256 of its bytes."""

    dtype: str
    shape: tuple[int, ...]
    sha256: str

    @classmethod
    def of(cls, tensor: Tensor) -> Self:
        """Digest ``tensor``'s row-major bytes.

        Returns:
          digest: The tensor's dtype, shape, and content hash.

        """
        array = tensor.detach().cpu().contiguous().numpy()
        return cls(
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        )


@dataclass(frozen=True)
class ReferenceDigests:
    """The frozen reference: the image set's hash and every batch's digests.

    A text file, one line per item, so a re-mint reads as a diff::

        image_set <sha256>
        batch 0 image uint8 512x3x160x160 <sha256>
    """

    image_set: str
    batches: list[dict[str, TensorDigest]]

    @classmethod
    def read(cls, path: Path) -> Self:
        """Parse a digest file, skipping ``#`` comment lines.

        Returns:
          digests: The file's contents.

        """
        image_set = ""
        batches: list[dict[str, TensorDigest]] = []
        for line in path.read_text().splitlines():
            words = line.split()
            if not words or words[0].startswith("#"):
                continue
            if words[0] == "image_set":
                image_set = words[1]
                continue
            _, index, name, dtype, shape, sha256 = words
            if int(index) == len(batches):
                batches.append({})
            batches[int(index)][name] = TensorDigest(
                dtype=dtype,
                shape=tuple(int(n) for n in shape.split("x")),
                sha256=sha256,
            )
        return cls(image_set=image_set, batches=batches)

    def write(self, path: Path, *, header: str = "") -> None:
        """Write the digests, prefixing each line of ``header`` with ``#``."""
        lines = [f"# {line}" for line in header.splitlines()]
        lines.append(f"image_set {self.image_set}")
        for index, batch in enumerate(self.batches):
            for name, digest in batch.items():
                shape = "x".join(str(n) for n in digest.shape)
                lines.append(
                    f"batch {index} {name} {digest.dtype} {shape} {digest.sha256}",
                )
        _ = path.write_text("\n".join(lines) + "\n")


def digest_image_set(directory: Path) -> str:
    """Hash every file under ``directory``: its relative path and its bytes.

    Symlinks are followed, so a linked ImageNet subset hashes its images.

    Returns:
      sha256: Hex digest of the sorted ``<path> <sha256>`` manifest.

    """
    manifest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        with path.open("rb") as file:
            content = hashlib.file_digest(file, "sha256").hexdigest()
        manifest.update(f"{path.relative_to(directory)} {content}\n".encode())
    return manifest.hexdigest()


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
    """Time ``pipeline`` and require its batches to match the frozen reference."""

    class Config(Fig["LoaderThroughput"]):
        pipeline: Makeable[DataPipeline] = field(
            default_factory=ffcv_train_data_pipeline,
        )
        """The pipeline being timed; a fork changes this and nothing else."""

        reference: Makeable[DataPipeline] = field(
            default_factory=ffcv_train_data_pipeline,
        )
        """The pipeline the digests were minted from. Its source names the
        image set; it runs only under a tolerance, to recover the pixels a
        digest cannot hold, and then must still reproduce its digests."""

        reference_digests: Path | str = "throughput_exp000_synthetic.sha256"
        """Frozen digests of ``reference``'s batches, beside this module."""

        tolerance: Makeable[PixelTolerance] | None = None
        """None compares every field bit for bit."""

        fields: list[str] = field(default_factory=lambda: ["image", "label"])
        """Batch keys compared against the reference: what the train step reads.
        The first one's leading dimension counts the images."""

        seed: int = 0
        """Reseeds every pass, so crop and flip draws repeat bit for bit."""

        num_repeats: int = 5
        """Timed passes over the data; the report keeps every one."""

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
        reference = config.reference
        if not (
            isinstance(reference, DataPipeline.Config)
            and isinstance(reference.source, HasNormalizedWorkingDirPattern)
        ):
            raise TypeError("reference must be a DataPipeline reading a directory.")
        self.config = config
        self.image_set_dir = Path(reference.source.working_dir)
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
            tier = "batches bit-identical to the frozen reference"
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
        """Check the image set, then time ``num_repeats`` fresh-process passes.

        Returns:
          report: Per-pass rates and every batch outside the correctness tier.

        Raises:
          ValueError: The staged image set is not the one the digests were
            minted on, or the reference no longer reproduces its digests.

        """
        cfg = self.config
        expected = ReferenceDigests.read(_CWD / cfg.reference_digests)
        image_set = digest_image_set(self.image_set_dir)
        if image_set != expected.image_set:
            raise ValueError(
                f"The image set under {self.image_set_dir} hashes to {image_set}; "
                f"the reference digests were minted on {expected.image_set}. "
                "Restage it, or re-mint the digests for this set.",
            )
        with tempfile.TemporaryDirectory() as scratch:
            compare_dir = None
            if self.tolerance is not None:
                reference, _ = _run_pass(cfg.reference, cfg, save_dir=scratch)
                if reference.digests != expected.batches:
                    raise ValueError(
                        "The reference no longer reproduces its frozen digests.",
                    )
                compare_dir = scratch
            passes = [
                _run_pass(
                    cfg.pipeline,
                    cfg,
                    compare_dir=compare_dir,
                    drift_fields=None
                    if self.tolerance is None
                    else self.tolerance.config.fields,
                )
                for _ in range(cfg.num_repeats)
            ]
        mismatches: list[str] = []
        worst_diff, worst_fraction = 0.0, 0.0
        for repeat, (result, _) in enumerate(passes):
            if result.num_images == 0:
                raise ValueError(f"Pass {repeat} yielded no batch.")
            lines, diff, fraction = self._check(expected.batches, result, repeat)
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

    def mint_reference(self) -> ReferenceDigests:
        """Drain ``reference`` in a fresh process and digest what it made.

        Returns:
          digests: The image set's hash and every reference batch's digests.

        """
        result, _ = _run_pass(self.config.reference, self.config)
        return ReferenceDigests(
            image_set=digest_image_set(self.image_set_dir),
            batches=result.digests,
        )

    def _check(
        self,
        expected: list[dict[str, TensorDigest]],
        result: _Pass,
        repeat: int,
    ) -> tuple[list[str], float, float]:
        """Return one line per field outside its tier, and the worst drift."""
        if len(expected) != len(result.digests):
            return (
                [f"pass {repeat}: {len(result.digests)} batches vs {len(expected)}"],
                0.0,
                0.0,
            )
        lines: list[str] = []
        worst_diff, worst_fraction = 0.0, 0.0
        for index, (want, got) in enumerate(zip(expected, result.digests, strict=True)):
            for name, digest in want.items():
                other = got[name]
                where = f"pass {repeat} batch {index} {name}"
                if (other.dtype, other.shape) != (digest.dtype, digest.shape):
                    lines.append(
                        f"{where}: {other.dtype}{list(other.shape)} vs "
                        f"{digest.dtype}{list(digest.shape)}",
                    )
                elif other.sha256 == digest.sha256:
                    continue
                elif self.tolerance is None or name not in self.tolerance.config.fields:
                    lines.append(f"{where}: bytes differ from the frozen reference")
                else:
                    diff, fraction = result.drift[index][name]
                    worst_diff = max(worst_diff, diff)
                    worst_fraction = max(worst_fraction, fraction)
                    if not self.tolerance.admits(diff, fraction):
                        lines.append(
                            f"{where}: max diff {diff:g}, {fraction:.4%} differ",
                        )
        return lines, worst_diff, worst_fraction


@dataclass(frozen=True)
class _Pass:
    """What one pass process sends back."""

    digests: list[dict[str, TensorDigest]]
    drift: list[dict[str, tuple[float, float]]]
    """Per batch, each tolerance field's max diff and fraction differing."""
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
    drift_fields: list[str] | None = None,
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
            "drift_fields": drift_fields or [],
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
    _require_empty_group(process.pid)
    return result, cpu - result.cpu_outside_sec


def _pass(
    pipeline: bytes,
    *,
    seed: int,
    fields: list[str],
    drift_fields: list[str],
    save_dir: str | None,
    compare_dir: str | None,
    connection: Connection,
) -> None:
    """Run in a fresh process: drain one pass, then digest it outside the window."""
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
        hashing_started = time.thread_time()
        drift: list[dict[str, tuple[float, float]]] = []
        for index, batch in enumerate(batches):
            if save_dir is not None:
                torch.save(batch, Path(save_dir) / f"{index}.pt")
            if compare_dir is not None:
                want = cast(
                    "dict[str, Tensor]",
                    torch.load(Path(compare_dir) / f"{index}.pt", weights_only=True),
                )
                drift.append(
                    {name: _drift(want[name], batch[name]) for name in drift_fields},
                )
        result = _Pass(
            digests=[{n: TensorDigest.of(t) for n, t in b.items()} for b in batches],
            drift=drift,
            num_images=sum(len(b[fields[0]]) for b in batches),
            elapsed_sec=elapsed,
            first_batch_sec=first_batch_sec,
            cpu_outside_sec=cpu_before + time.thread_time() - hashing_started,
        )
    except BaseException:
        connection.send(traceback.format_exc())
        raise
    connection.send(result)


def _drift(want: Tensor, got: Tensor) -> tuple[float, float]:
    """Return the max per-value difference and the fraction that differ."""
    if want.shape != got.shape:
        return float("inf"), 1.0
    # numpy, not torch: its elementwise ops stay on this thread, whose CPU is
    # subtracted. float32 holds every difference of two uint8 values exactly.
    a = want.contiguous().numpy().astype(np.float32)
    b = got.contiguous().numpy().astype(np.float32)
    max_diff = float(np.abs(a - b).max())  # pyright: ignore[reportAny] -- numpy reductions are dtype-erased.
    num_differ = int(np.count_nonzero(a != b))  # pyright: ignore[reportAny] -- numpy comparisons are dtype-erased.
    return max_diff, num_differ / a.size


def _require_empty_group(group: int) -> None:
    """Raise if any process of the pass's group outlives the grace period."""
    deadline = time.monotonic() + SURVIVOR_GRACE_SEC
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
