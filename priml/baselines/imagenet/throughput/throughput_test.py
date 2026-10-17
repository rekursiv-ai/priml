"""Tests for the input pipeline throughput job.

The job's one promise is that a number is reported only for batches equal to
the reference's, and only for work it actually timed. No test starts a
process: passes run inline through a stand-in ``spawn`` context, and the
process-group and CPU-usage calls are faked, so the accounting each exploit
dodged (worker processes, caches across passes, work before the first batch,
leftover processes) is pinned piece by piece.
The golden freezes what ``exp000``'s pipeline makes of fixed JPEG bytes,
element by element: the run compares a fork against ``reference`` live, and
this is what pins ``reference`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import ctypes
import errno
import io
import logging
import multiprocessing
import os
import resource
import signal
import time

from configgle import Fig, Makeable
from PIL import Image

import numpy as np
import pytest
import torch

from priml.baselines.imagenet.throughput import throughput, timed_pass
from priml.baselines.imagenet.throughput.experiments import exp000
from priml.baselines.imagenet.throughput.throughput import (
    LoaderThroughput,
    PixelTolerance,
)
from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import DataPipeline
from priml.data.processors.augmentation import (
    GetRandomResizedCropBoxFromDimensions,
)
from priml.data.processors.decode_batch import DecodeCropResizeBatch
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.math.seed import set_seed_local
from priml.testing.golden import assert_tensor_golden, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from torch import Tensor


_CWD: Final = Path(__file__).resolve().parent
GOLDEN: Final = _CWD / "testdata" / "throughput.pt"
SYNSETS: Final = ("n01440764", "n01443537")
NUM_IMAGES: Final = 5
"""Two batches of two, and a remainder the batcher drops."""
IMAGE: Final = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
LABEL: Final = torch.tensor([3, 5])


def _encoded(index: int) -> bytes:
    """Encode a seeded noise image as a 4:2:0 JPEG, as most of ImageNet is."""
    height, width = 9 + index, 13 + 2 * index
    rgb = np.random.default_rng(index).integers(0, 256, (height, width, 3), np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90, subsampling=2)
    return buffer.getvalue()


def _jpegs() -> list[bytes]:
    """Return the golden's stored JPEGs: re-encoding would pin Pillow, not the pipeline."""
    if GOLDEN.is_file():
        stored = read_tensors(GOLDEN)
        return [stored[f"jpeg{i}"].numpy().tobytes() for i in range(NUM_IMAGES)]
    return [_encoded(i) for i in range(NUM_IMAGES)]


def _write(root: Path, jpegs: list[bytes]) -> None:
    for index, data in enumerate(jpegs):
        synset = SYNSETS[index % len(SYNSETS)]
        (root / "train" / synset).mkdir(parents=True, exist_ok=True)
        _ = (root / "train" / synset / f"{synset}_{index}.JPEG").write_bytes(data)


def _shrink(pipeline: object, root: Path) -> DataPipeline.Config:
    """Shrink ``pipeline`` by size only: crop, batch, and the image set."""
    assert isinstance(pipeline, DataPipeline.Config)
    assert isinstance(pipeline.source, ExtractedImageNetSource.Config)
    pipeline.source.working_dir = root
    for processor in pipeline.processors:
        if isinstance(processor, GetRandomResizedCropBoxFromDimensions.Config):
            processor.size = (5, 4)
        elif isinstance(processor, Batcher.Config):
            processor.size = 2
    return pipeline


def _tiny(root: Path) -> LoaderThroughput.Config:
    cfg = exp000()
    cfg.base_dir = None
    cfg.num_repeats = 2
    _ = _shrink(cfg.pipeline, root)
    _ = _shrink(cfg.reference, root)
    return cfg


def _decode(pipeline: object) -> DecodeCropResizeBatch.Config:
    assert isinstance(pipeline, DataPipeline.Config)
    (decode,) = (
        p for p in pipeline.processors if isinstance(p, DecodeCropResizeBatch.Config)
    )
    return decode


@pytest.fixture(scope="module")
def staged(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage the five JPEGs once for every test that runs the real pipeline."""
    root = tmp_path_factory.mktemp("throughput")
    _write(root, _jpegs())
    return root


def _result(
    diffs: dict[tuple[int, str], timed_pass.Diff | str],
    *,
    num_batches: int = 1,
) -> timed_pass.PassResult:
    return timed_pass.PassResult(
        diffs=diffs,
        num_batches=num_batches,
        num_reference_batches=1,
        num_images=2,
        elapsed_sec=1.0,
        first_batch_sec=0.5,
        cpu_outside_sec=0.0,
    )


def test_exact_tier_reports_any_changed_value() -> None:
    job = LoaderThroughput(LoaderThroughput.Config())
    moved = IMAGE.clone()
    moved[0, 1, 2] += 3
    diff = timed_pass.compare(IMAGE, moved)
    assert diff is not None

    lines, _, _ = job._check(_result({(0, "image"): diff}), repeat=1)

    assert lines == [
        "pass 1 batch 0 image: 1/24 differ, max diff 3, first at [0, 1, 2]",
    ]


def test_shape_and_batch_count_changes_are_reported() -> None:
    job = LoaderThroughput(LoaderThroughput.Config())

    reshaped, _, _ = job._check(
        _result({(0, "image"): "torch.uint8[3, 2, 4] vs torch.uint8[2, 3, 4]"}),
        repeat=0,
    )
    extra, _, _ = job._check(_result({}, num_batches=2), repeat=0)

    assert reshaped == [
        "pass 0 batch 0 image: torch.uint8[3, 2, 4] vs torch.uint8[2, 3, 4]",
    ]
    assert extra == ["pass 0: 2 batches vs 1"]


def _image_drift(level: float) -> timed_pass.Diff:
    """Return a quarter of a 24-value image moved by ``level``."""
    return timed_pass.Diff(
        num_differing=6,
        numel=24,
        max_abs_diff=level,
        first_index=(0, 0, 0),
    )


def _tolerance(
    max_abs_diff: float,
    max_fraction_differing: float,
) -> PixelTolerance.Config:
    tolerance = PixelTolerance.Config()
    tolerance.max_abs_diff = max_abs_diff
    tolerance.max_fraction_differing = max_fraction_differing
    return tolerance


def test_a_tolerance_admits_drift_within_it_and_keeps_labels_exact() -> None:
    cfg = LoaderThroughput.Config()
    cfg.tolerance = _tolerance(2, 0.25)
    job = LoaderThroughput(cfg)
    label = timed_pass.compare(LABEL, LABEL + 1)
    assert label is not None

    within, diff, fraction = job._check(
        _result({(0, "image"): _image_drift(2), (0, "label"): label}),
        repeat=0,
    )
    beyond, _, _ = job._check(_result({(0, "image"): _image_drift(3)}), repeat=0)
    _, unbounded, _ = job._check(
        _result({(0, "image"): _image_drift(0), (1, "image"): _image_drift(np.inf)}),
        repeat=0,
    )

    assert within == ["pass 0 batch 0 label: 2/2 differ, max diff 1, first at [0]"]
    assert (diff, fraction) == (2.0, 0.25)
    assert unbounded == np.inf
    assert beyond == [
        "pass 0 batch 0 image: 6/24 differ, max diff 3, first at [0, 0, 0]",
    ]


def test_rejects_a_nonsensical_count() -> None:
    cfg = LoaderThroughput.Config()
    cfg.num_repeats = 0

    with pytest.raises(ValueError, match=r"^num_repeats must be positive; got 0\.$"):
        _ = LoaderThroughput(cfg)


def test_accepts_a_single_repeat() -> None:
    cfg = LoaderThroughput.Config()
    cfg.num_repeats = 1

    assert LoaderThroughput(cfg).config.num_repeats == 1


def test_rejects_comparing_no_fields() -> None:
    cfg = LoaderThroughput.Config()
    cfg.fields = []

    with pytest.raises(
        ValueError,
        match=r"^fields must name at least one batch key; got \[\]\.$",
    ):
        _ = LoaderThroughput(cfg)


def test_rejects_a_tolerance_on_a_field_that_is_not_compared() -> None:
    tolerance = PixelTolerance.Config()
    tolerance.fields = ["image", "media"]
    cfg = LoaderThroughput.Config()
    cfg.tolerance = tolerance

    with pytest.raises(
        ValueError,
        match=(
            r"^tolerance names fields that are not compared: \['media'\]; "
            r"compared fields are \['image', 'label'\]\.$"
        ),
    ):
        _ = LoaderThroughput(cfg)


def test_accepts_a_tolerance_on_compared_fields() -> None:
    tolerance = _tolerance(1, 1)
    tolerance.fields = ["image", "label"]
    cfg = LoaderThroughput.Config()
    cfg.tolerance = tolerance

    job = LoaderThroughput(cfg)

    assert job.tolerance is not None
    assert job.tolerance.config.fields == ["image", "label"]


@pytest.mark.parametrize(
    ("max_abs_diff", "max_fraction_differing", "match"),
    [
        (-1.0, 0.0, "max_abs_diff"),
        (float("nan"), 0.0, "max_abs_diff"),
        (0.0, -0.5, "max_fraction_differing"),
        (0.0, 1.5, "max_fraction_differing"),
        (0.0, float("nan"), "max_fraction_differing"),
    ],
)
def test_rejects_a_tolerance_outside_its_range(
    max_abs_diff: float,
    max_fraction_differing: float,
    match: str,
) -> None:
    config = _tolerance(max_abs_diff, max_fraction_differing)

    with pytest.raises(ValueError, match=match):
        _ = config.make()


def test_accepts_a_tolerance_at_its_bounds() -> None:
    assert _tolerance(0, 1).make().admits(0, 1)


def test_verdict_scores_a_clean_pass() -> None:
    result = _result({})

    assert throughput._verdict(result, exitcode=0, survived=False) is result


@pytest.mark.parametrize(
    ("result", "exitcode", "survived", "match"),
    [
        (
            "Traceback: boom",
            1,
            False,
            r"^Pass failed \(exit code 1\):\nTraceback: boom$",
        ),
        (_result({}), 3, False, r"exit code 3\):\nit sent a result first$"),
        (
            throughput.NO_RESULT,
            -9,
            True,
            (
                r"exit code -9\):\nthe pass process exited without a result\n"
                r"A process the pipeline started outlived its pass"
            ),
        ),
        (_result({}), 0, True, "^A process the pipeline started outlived its pass"),
    ],
)
def test_verdict_refuses_a_failed_or_leaky_pass(
    result: timed_pass.PassResult | str,
    exitcode: int,
    survived: bool,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _ = throughput._verdict(result, exitcode=exitcode, survived=survived)


class _ScriptedKillpg:
    """Stand in for ``os.killpg``: raise or return per call, in order."""

    def __init__(self, *outcomes: bool) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, group: int, sig: int) -> None:
        self.calls.append((group, sig))
        if not self.outcomes.pop(0):
            raise ProcessLookupError


def _no_sleep(seconds: float) -> None:
    del seconds


class _ScriptedWaitpid:
    """Stand in for ``os.waitpid``: reap each pid in order, then have no child."""

    def __init__(self, *pids: int) -> None:
        self.pids = list(pids)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, pid: int, options: int) -> tuple[int, int]:
        self.calls.append((pid, options))
        if not self.pids:
            raise ChildProcessError
        return self.pids.pop(0), 0


def test_group_survives_waits_out_a_group_that_empties_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killpg = _ScriptedKillpg(True, False)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "waitpid", _ScriptedWaitpid())
    monkeypatch.setattr(time, "sleep", _no_sleep)

    assert not throughput._group_survives(7, grace_sec=60)
    assert killpg.calls == [(7, 0), (7, 0)]


def test_group_survives_kills_a_group_that_outlives_its_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killpg = _ScriptedKillpg(True, True)
    waitpid = _ScriptedWaitpid()
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "waitpid", waitpid)

    assert throughput._group_survives(7, grace_sec=-1)
    assert killpg.calls == [(7, 0), (7, signal.SIGKILL)]
    assert waitpid.calls == [(-7, os.WNOHANG)] * 2


def test_group_survives_reports_a_group_that_empties_while_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killpg = _ScriptedKillpg(True, False)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "waitpid", _ScriptedWaitpid())

    assert not throughput._group_survives(7, grace_sec=-1)
    assert killpg.calls == [(7, 0), (7, signal.SIGKILL)]


def test_reap_charges_every_exited_orphan_of_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waitpid = _ScriptedWaitpid(11, 12, 0)
    monkeypatch.setattr(os, "waitpid", waitpid)

    throughput._reap(7)

    assert waitpid.calls == [(-7, os.WNOHANG)] * 3


class _Libc:
    """Stand in for libc: ``prctl`` records its arguments and returns ``status``."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.calls: list[tuple[int, ...]] = []

    def prctl(self, *args: int) -> int:
        self.calls.append(args)
        return self.status


def _load(libc: _Libc, name: str | None, *, use_errno: bool) -> _Libc:
    assert (name, use_errno) == (None, True)
    return libc


def _eperm() -> int:
    return errno.EPERM


def test_become_subreaper_adopts_orphaned_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libc = _Libc(status=0)
    monkeypatch.setattr(ctypes, "CDLL", partial(_load, libc))

    throughput._become_subreaper()

    assert libc.calls == [(36, 1, 0, 0, 0)]


def test_become_subreaper_raises_when_the_kernel_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ctypes, "CDLL", partial(_load, _Libc(status=-1)))
    monkeypatch.setattr(ctypes, "get_errno", _eperm)

    with pytest.raises(OSError, match=r"^\[Errno 1\] prctl\(PR_SET_CHILD_SUBREAPER\)"):
        throughput._become_subreaper()


class _ScriptedAlive:
    """Stand in for the pass process: ``is_alive`` answers in order."""

    def __init__(self, *answers: bool) -> None:
        self.answers = list(answers)

    def is_alive(self) -> bool:
        return self.answers.pop(0)


def test_receive_returns_what_the_pass_sent() -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    sender.send("Traceback: boom")
    process = cast("BaseProcess", _ScriptedAlive(True))

    assert throughput._receive(receiver, process) == "Traceback: boom"


def test_receive_gives_up_once_the_pass_is_dead_though_its_pipe_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A descendant holding the sending end must not hang the run."""
    monkeypatch.setattr(throughput, "POLL_SEC", 0.001)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = cast("BaseProcess", _ScriptedAlive(True, True, False))

    assert throughput._receive(receiver, process) == throughput.NO_RESULT
    assert not sender.closed


def test_receive_says_when_the_pipe_closed_without_a_result() -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    sender.close()
    process = cast("BaseProcess", _ScriptedAlive(True))

    assert throughput._receive(receiver, process) == throughput.NO_RESULT


class _InlineProcess:
    """Run a pass's target in this process when started, as ``spawn`` would."""

    def __init__(
        self,
        target: Callable[..., None],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: _InlineContext,
    ) -> None:
        self.target, self.args, self.kwargs = target, args, kwargs
        self.status, self.pid, self.hangs = context.exitcode, context.pid, context.hangs
        self.runs, self.sends = context.runs, context.sends
        self.exitcode: int | None = None
        self.started = self.killed = False
        self.joins: list[float | None] = []

    def start(self) -> None:
        self.started = True
        if self.sends is not None:
            # `isinstance` cannot bind Connection's send type; ``_run_pass`` built it.
            sender = cast("Connection[object, object]", self.kwargs["connection"])
            sender.send(self.sends)
        elif self.runs:
            self.target(*self.args, **self.kwargs)
        if not self.hangs:
            self.exitcode = self.status

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def kill(self) -> None:
        self.killed = True
        self.exitcode = -signal.SIGKILL


class _InlineContext:
    """A ``spawn`` context whose processes run inline, for ``_run_pass``.

    Records every pipe and process it makes, every survivor check made of
    one, every group kill, and every time the harness became a subreaper.
    """

    def __init__(
        self,
        *,
        runs: bool = True,
        sends: timed_pass.PassResult | None = None,
        exitcode: int = 0,
        pid: int | None = 7,
        hangs: bool = False,
    ) -> None:
        self.runs, self.sends, self.exitcode, self.pid = runs, sends, exitcode, pid
        self.hangs = hangs
        self.pipes: list[tuple[Connection, Connection]] = []
        self.processes: list[_InlineProcess] = []
        self.survivor_checks: list[tuple[int, float]] = []
        self.subreapers: list[None] = []
        self.killpg = _ScriptedKillpg(*[True] * 8)

    def Pipe(self, *, duplex: bool) -> tuple[Connection, Connection]:  # noqa: N802 -- Mirrors multiprocessing's context API.
        pipe = multiprocessing.Pipe(duplex=duplex)
        self.pipes.append(pipe)
        return pipe

    def Process(  # noqa: N802 -- Mirrors multiprocessing's context API.
        self,
        *,
        target: Callable[..., None],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> _InlineProcess:
        process = _InlineProcess(target, args, kwargs, self)
        self.processes.append(process)
        return process


class Batches:
    """Yield ``num_batches`` copies of ``IMAGE`` and ``LABEL``."""

    class Config(Fig["Batches"]):
        num_batches: int = 2

    def __init__(self, config: Config) -> None:
        self.config = config

    def __iter__(self) -> Iterator[dict[str, object]]:
        for _ in range(self.config.num_batches):
            yield {"image": IMAGE, "label": LABEL}


def _go_inline(monkeypatch: pytest.MonkeyPatch, context: _InlineContext) -> None:
    """Run passes inline: no spawn, no process group, nothing left to wait for."""
    monkeypatch.setattr(multiprocessing, "get_context", partial(_context, context))
    monkeypatch.setattr(os, "setpgid", _no_setpgid)
    monkeypatch.setattr(os, "killpg", context.killpg)
    monkeypatch.setattr(
        throughput,
        "_group_survives",
        partial(_nothing_survives, context.survivor_checks),
    )
    monkeypatch.setattr(
        throughput,
        "_become_subreaper",
        partial(context.subreapers.append, None),
    )


def _batches(num_batches: int) -> Makeable[DataPipeline]:
    batches = Batches.Config()
    batches.num_batches = num_batches
    return cast("Makeable[DataPipeline]", batches)


def _inline(
    monkeypatch: pytest.MonkeyPatch,
    context: _InlineContext,
) -> LoaderThroughput.Config:
    """Return a config of two-batch stand-in pipelines whose passes run inline."""
    _go_inline(monkeypatch, context)
    cfg = LoaderThroughput.Config()
    cfg.pipeline = cast("Makeable[DataPipeline]", Batches.Config())
    cfg.reference = cast("Makeable[DataPipeline]", Batches.Config())
    cfg.num_repeats = 2
    return cfg


def _context(context: _InlineContext, method: str) -> _InlineContext:
    assert method == "spawn"
    return context


def _no_setpgid(pid: int, pgrp: int) -> None:
    del pid, pgrp


def _nothing_survives(
    checks: list[tuple[int, float]],
    group: int,
    grace_sec: float,
) -> bool:
    checks.append((group, grace_sec))
    return False


def test_every_pass_runs_in_a_fresh_spawned_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _InlineContext()
    cfg = _inline(monkeypatch, context)
    cfg.survivor_grace_sec = 0.5

    _ = cfg.make().measure()

    assert len(context.processes) == cfg.num_repeats + 1
    assert all(
        p.started and p.joins == [0.5] and isinstance(p.args[0], bytes)
        for p in context.processes
    )
    assert context.survivor_checks == [(7, 0.5)] * (cfg.num_repeats + 1)
    assert all(receiver.closed for receiver, _ in context.pipes)
    assert context.subreapers == [None]


def test_run_pass_kills_a_pass_that_will_not_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-daemon child nobody joins keeps the pass process from exiting."""
    context = _InlineContext(hangs=True)
    cfg = _inline(monkeypatch, context)

    with pytest.raises(
        RuntimeError,
        match=r"^Pass failed \(exit code -9\):\nit sent a result first\n"
        r"A process the pipeline started outlived its pass",
    ):
        _ = throughput._run_pass(cfg.pipeline, cfg)

    (process,) = context.processes
    assert (process.killed, process.joins) == (True, [5.0, None])
    assert context.killpg.calls == [(7, signal.SIGKILL)]
    assert context.survivor_checks == [(7, 5.0)]


def _interrupted(receiver: Connection, process: BaseProcess) -> str:
    """Stand in for ``_receive``: the run is interrupted mid-pass."""
    del receiver, process
    raise KeyboardInterrupt


def test_run_pass_kills_the_pass_group_when_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _InlineContext()
    cfg = _inline(monkeypatch, context)
    monkeypatch.setattr(throughput, "_receive", _interrupted)

    with pytest.raises(KeyboardInterrupt):
        _ = throughput._run_pass(cfg.pipeline, cfg)

    (process,) = context.processes
    ((receiver, _),) = context.pipes
    assert (process.killed, process.joins, receiver.closed) == (True, [None], True)
    assert context.killpg.calls == [(7, signal.SIGKILL)]
    assert context.survivor_checks == []


def test_kill_reaps_a_pass_that_never_made_its_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _InlineContext()
    killpg = _ScriptedKillpg(False)
    monkeypatch.setattr(os, "killpg", killpg)
    process = _InlineProcess(_no_target, (), {}, context)

    throughput._kill(cast("BaseProcess", process), 7)

    assert (process.killed, process.joins) == (True, [None])
    assert killpg.calls == [(7, signal.SIGKILL)]


def _no_target() -> None:
    pass


def test_run_pass_charges_the_process_tree_less_its_work_off_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = timed_pass.PassResult(
        diffs={},
        num_batches=1,
        num_reference_batches=1,
        num_images=2,
        elapsed_sec=1.0,
        first_batch_sec=0.5,
        cpu_outside_sec=0.5,
    )
    cfg = _inline(monkeypatch, _InlineContext(sends=sent))
    monkeypatch.setattr(
        throughput,
        "_children_cpu_sec",
        partial(next, iter([10, 13.5])),
    )

    assert throughput._run_pass(cfg.pipeline, cfg) == (sent, 3.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Usage:
    """The two ``struct_rusage`` fields ``_children_cpu_sec`` reads."""

    ru_utime: float
    ru_stime: float


def _children_usage(who: int) -> _Usage:
    assert who == resource.RUSAGE_CHILDREN
    return _Usage(ru_utime=1.25, ru_stime=0.5)


def test_children_cpu_is_user_plus_system_time_of_waited_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource, "getrusage", _children_usage)

    assert throughput._children_cpu_sec() == 1.75


def test_measure_reports_every_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _inline(monkeypatch, _InlineContext()).make().measure()

    assert report.mismatches == []
    assert report.num_images == 4
    assert len(report.images_per_cpu_sec) == 2
    assert len(report.images_per_sec) == 2
    assert all(sec > 0 for sec in report.first_batch_sec)
    assert (report.max_abs_diff, report.max_fraction_differing) == (0.0, 0.0)


def test_measure_reports_a_pass_with_fewer_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _inline(monkeypatch, _InlineContext())
    cfg.pipeline = _batches(1)
    cfg.num_repeats = 1

    assert cfg.make().measure().mismatches == ["pass 0: 1 batches vs 2"]


def test_measure_stops_at_the_first_pass_that_yielded_no_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _InlineContext()
    cfg = _inline(monkeypatch, context)
    cfg.pipeline = _batches(0)

    with pytest.raises(ValueError, match=r"^Pass 0 yielded no batch\.$"):
        _ = cfg.make().measure()
    assert len(context.processes) == 2


def test_measure_names_a_reference_that_yielded_no_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _InlineContext()
    cfg = _inline(monkeypatch, context)
    cfg.reference = _batches(0)

    with pytest.raises(ValueError, match=r"^The reference yielded no batch\.$"):
        _ = cfg.make().measure()
    assert len(context.processes) == 1


def test_run_pass_says_when_the_pass_sent_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _inline(monkeypatch, _InlineContext(runs=False, exitcode=-9))

    with pytest.raises(
        RuntimeError,
        match=r"exit code -9\):\nthe pass process exited without a result",
    ):
        _ = throughput._run_pass(cfg.pipeline, cfg)


def test_run_pass_refuses_a_process_without_a_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _InlineContext(pid=None)
    cfg = _inline(monkeypatch, context)

    with pytest.raises(ValueError, match=r"^Expected group is not None\.$"):
        _ = throughput._run_pass(cfg.pipeline, cfg)


def _report(
    *,
    mismatches: list[str],
    max_abs_diff: float = 0.0,
) -> throughput.ThroughputReport:
    return throughput.ThroughputReport(
        images_per_cpu_sec=[4.0, 2.0],
        images_per_sec=[8.0, 6.0],
        first_batch_sec=[0.5, 0.25],
        num_images=4,
        mismatches=mismatches,
        max_abs_diff=max_abs_diff,
        max_fraction_differing=0.125,
    )


def _measured(report: throughput.ThroughputReport) -> throughput.ThroughputReport:
    return report


def _run_lines(
    job: LoaderThroughput,
    report: throughput.ThroughputReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    *args: str,
) -> list[str]:
    monkeypatch.setattr(job, "measure", partial(_measured, report))
    with caplog.at_level(logging.INFO, logger=throughput.__name__):
        job.run(*args)
    return [r.message for r in caplog.records]


def test_run_logs_every_rate_under_the_exact_tier(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lines = _run_lines(
        LoaderThroughput(LoaderThroughput.Config()),
        _report(mismatches=[]),
        monkeypatch,
        caplog,
        "--passthrough",
    )

    assert lines == [
        "Ignoring launcher passthrough args: ('--passthrough',).",
        (
            "images/cpu-sec (scored) over 2 passes of 4 images: median 3.0, mean 3.0, "
            "min 2.0, max 4.0; batches bit-identical to the reference."
        ),
        (
            "images/sec (wall, not scored) over 2 passes of 4 images: median 7.0, "
            "mean 7.0, min 6.0, max 8.0; batches bit-identical to the reference."
        ),
        (
            "first-batch sec over 2 passes of 4 images: median 0.4, mean 0.4, "
            "min 0.2, max 0.5; batches bit-identical to the reference."
        ),
    ]


def test_run_logs_the_tolerance_a_run_was_held_to(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = LoaderThroughput.Config()
    cfg.tolerance = _tolerance(2, 0.5)

    lines = _run_lines(
        LoaderThroughput(cfg),
        _report(mismatches=[], max_abs_diff=1),
        monkeypatch,
        caplog,
    )

    assert len(lines) == 3
    assert lines[0].endswith(
        "; UNDER TOLERANCE max_abs_diff=2 max_fraction_differing=0.5 on "
        "['image'] (observed 1 and 12.5000%), not exact.",
    )


def test_run_refuses_to_log_a_rate_for_changed_batches(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = LoaderThroughput(LoaderThroughput.Config())

    with pytest.raises(
        RuntimeError,
        match=r"^2 batch mismatches against the reference:\na\nb$",
    ):
        _ = _run_lines(job, _report(mismatches=["a", "b"]), monkeypatch, caplog)


def test_exp000_batches_match_the_golden(tmp_path: Path) -> None:
    jpegs = _jpegs()
    _write(tmp_path, jpegs)
    cfg = _tiny(tmp_path)
    pipeline = cfg.pipeline
    assert isinstance(pipeline, DataPipeline.Config)

    record: dict[str, Tensor] = {
        f"jpeg{i}": torch.frombuffer(bytearray(data), dtype=torch.uint8)
        for i, data in enumerate(jpegs)
    }
    _ = set_seed_local(cfg.seed)
    for index, batch in enumerate(pipeline.make()):
        image, label = batch["image"], batch["label"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        record[f"image{index}"] = image.contiguous()
        record[f"label{index}"] = label

    assert_tensor_golden(GOLDEN, record)


def test_decode_threads_leave_exp000_batches_alone(
    staged: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _go_inline(monkeypatch, _InlineContext())
    cfg = _tiny(staged)
    cfg.num_repeats = 1
    _decode(cfg.pipeline).num_threads = 3

    report = cfg.make().measure()

    assert (report.mismatches, report.num_images) == ([], 4)


def test_the_accurate_idct_moves_pixels_a_tolerance_can_admit(
    staged: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _go_inline(monkeypatch, _InlineContext())
    cfg = _tiny(staged)
    cfg.num_repeats = 1
    _decode(cfg.pipeline).fast_dct = False
    cfg.tolerance = _tolerance(255, 1)

    report = cfg.make().measure()

    assert report.mismatches == []
    assert report.max_abs_diff > 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
