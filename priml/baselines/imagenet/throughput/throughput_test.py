"""Tests for the input pipeline throughput job.

The job's one promise is that a number is reported only for batches equal to
the reference's, and only for work it actually timed. Most tests here are
about the comparison biting; the subprocess tests run three pipelines that
gamed the earlier in-process harness (decode in a worker process, a cache
shared across passes, everything decoded before the first batch) and check
each is now charged for its work. The golden freezes what ``exp000``'s
pipeline makes of fixed JPEG bytes, element by element: the run compares a
fork against ``reference`` live, and this is what pins ``reference`` itself.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import io
import logging
import multiprocessing
import os
import subprocess
import time

from configgle import Fig, Makeable
from PIL import Image

import numpy as np
import pytest
import torch

from priml.baselines.imagenet.throughput import throughput
from priml.baselines.imagenet.throughput.experiments import exp000
from priml.baselines.imagenet.throughput.throughput import (
    LoaderThroughput,
    PixelTolerance,
)
from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import DataPipeline
from priml.data.processors.augmentation import GetRandomResizedCropBoxFromDimensions
from priml.data.processors.decode_batch import DecodeCropResizeBatch
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.math.seed import set_seed_local
from priml.testing.golden import assert_tensor_golden, read_tensors


if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventType

    from torch import Tensor


GOLDEN: Final = Path(__file__).parent.resolve() / "testdata" / "throughput.pt"
SYNSETS: Final = ("n01440764", "n01443537")
NUM_IMAGES: Final = 5
"""Two batches of two, and a remainder the batcher drops."""
BURN_SEC: Final = 0.3
"""CPU a gaming pipeline spends where the old harness could not see it."""


def _encoded(index: int) -> bytes:
    """Encode a seeded noise image as a 4:2:0 JPEG, as most of ImageNet is."""
    height, width = 9 + index, 13 + 2 * index
    rgb = np.random.default_rng(index).integers(0, 256, (height, width, 3), np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90, subsampling=2)
    return buffer.getvalue()


def _jpegs() -> list[bytes]:
    """Return the golden's stored JPEGs: re-encoding would pin Pillow, not the pipeline."""
    if GOLDEN.is_file() and os.environ.get("BFB_REGENERATE", "0") != "1":
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
    """Stage the five JPEGs once for every subprocess test."""
    root = tmp_path_factory.mktemp("throughput")
    _write(root, _jpegs())
    return root


def _result(
    diffs: dict[tuple[int, str], throughput._Diff | str],
    *,
    num_batches: int = 1,
) -> throughput._Pass:
    return throughput._Pass(
        diffs=diffs,
        num_batches=num_batches,
        num_reference_batches=1,
        num_images=2,
        elapsed_sec=1.0,
        first_batch_sec=0.5,
        cpu_outside_sec=0.0,
    )


IMAGE: Final = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
LABEL: Final = torch.tensor([3, 5])


def test_diff_names_the_count_the_largest_move_and_where_it_starts() -> None:
    moved = IMAGE.clone()
    moved[1, 2, :2] += torch.tensor([4, 1], dtype=torch.uint8)

    assert throughput._diff(IMAGE, IMAGE.clone()) is None
    assert throughput._diff(IMAGE, moved) == throughput._Diff(
        num_differing=2,
        numel=IMAGE.numel(),
        max_abs_diff=4.0,
        first_index=(1, 2, 0),
    )
    assert throughput._diff(IMAGE, IMAGE.reshape(3, 2, 4)) == (
        "torch.uint8[3, 2, 4] vs torch.uint8[2, 3, 4]"
    )


def test_exact_tier_reports_any_changed_value() -> None:
    job = LoaderThroughput(LoaderThroughput.Config())
    moved = IMAGE.clone()
    moved[0, 1, 2] += 3
    diff = throughput._diff(IMAGE, moved)
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


def test_a_tolerance_admits_drift_within_it_and_keeps_labels_exact() -> None:
    cfg = LoaderThroughput.Config()
    cfg.tolerance = PixelTolerance.Config(max_abs_diff=2, max_fraction_differing=0.25)
    job = LoaderThroughput(cfg)
    label = throughput._diff(LABEL, LABEL + 1)
    assert label is not None

    def image(level: int) -> throughput._Diff:
        return throughput._Diff(
            num_differing=6,
            numel=24,
            max_abs_diff=level,
            first_index=(0, 0, 0),
        )

    within, diff, fraction = job._check(
        _result({(0, "image"): image(2), (0, "label"): label}),
        repeat=0,
    )
    beyond, _, _ = job._check(_result({(0, "image"): image(3)}), repeat=0)

    assert within == ["pass 0 batch 0 label: 2/2 differ, max diff 1, first at [0]"]
    assert (diff, fraction) == (2.0, 0.25)
    assert beyond == [
        "pass 0 batch 0 image: 6/24 differ, max diff 3, first at [0, 0, 0]",
    ]


def test_rejects_a_nonsensical_count() -> None:
    cfg = LoaderThroughput.Config()
    cfg.num_repeats = 0

    with pytest.raises(ValueError, match="num_repeats"):
        _ = LoaderThroughput(cfg)


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


# Everything below runs passes in spawned interpreters, a few seconds each.


@pytest.mark.cli_python_subprocess
def test_an_unchanged_pipeline_matches_and_counts_every_batch(staged: Path) -> None:
    report = _tiny(staged).make().measure()

    assert report.mismatches == []
    # Two batches of two, the first one included.
    assert report.num_images == 4
    assert len(report.images_per_cpu_sec) == 2
    assert all(rate > 0 for rate in report.images_per_cpu_sec)
    assert all(rate > 0 for rate in report.images_per_sec)
    assert all(sec > 0 for sec in report.first_batch_sec)


@pytest.mark.cli_python_subprocess
def test_decode_threads_leave_the_batches_alone(staged: Path) -> None:
    cfg = _tiny(staged)
    cfg.num_repeats = 1
    _decode(cfg.pipeline).num_threads = 3

    assert cfg.make().measure().mismatches == []


@pytest.mark.cli_python_subprocess
def test_run_logs_the_score_and_the_exact_tier(
    staged: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _tiny(staged)
    cfg.num_repeats = 1

    with caplog.at_level(
        logging.INFO,
        logger="priml.baselines.imagenet.throughput.throughput",
    ):
        cfg.make().run("--passthrough")

    lines = [r.message for r in caplog.records if "passes of" in r.message]
    assert [m.split(" over ")[0] for m in lines] == [
        "images/cpu-sec (scored)",
        "images/sec (wall, not scored)",
        "first-batch sec",
    ]
    assert all("bit-identical to the reference" in m for m in lines)


@pytest.mark.cli_python_subprocess
def test_run_refuses_to_report_a_rate_for_changed_batches(staged: Path) -> None:
    cfg = _tiny(staged)
    cfg.num_repeats = 1
    _decode(cfg.pipeline).flip_p = 0.0

    with pytest.raises(AssertionError, match="batch mismatches"):
        cfg.make().run()


@pytest.mark.cli_python_subprocess
def test_the_accurate_idct_fails_exact_and_passes_a_tier_that_says_so(
    staged: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _tiny(staged)
    cfg.num_repeats = 1
    # The accurate IDCT lands a level or two off the fast one exp000 uses.
    _decode(cfg.pipeline).fast_dct = False
    exact = cfg.make().measure()
    cfg.tolerance = PixelTolerance.Config(max_abs_diff=255, max_fraction_differing=1)

    with caplog.at_level(
        logging.INFO,
        logger="priml.baselines.imagenet.throughput.throughput",
    ):
        cfg.make().run()

    assert exact.mismatches
    assert all(" image: " in line for line in exact.mismatches)
    lines = [r.message for r in caplog.records if "passes of" in r.message]
    assert lines
    assert all("UNDER TOLERANCE" in m and "not exact" in m for m in lines)


def _burn(seconds: float) -> None:
    """Spend ``seconds`` of this thread's CPU."""
    started = time.thread_time()
    while time.thread_time() - started < seconds:
        pass


def _drain_into(
    inner: Makeable[DataPipeline],
    seed: int,
    queue: Queue[object],
    drained: EventType,
) -> None:
    set_seed_local(seed)
    _burn(BURN_SEC)
    for batch in inner.make():
        queue.put({"image": batch["image"], "label": batch["label"]})
    queue.put(None)
    # On Linux torch shares a tensor as a file descriptor the reader collects
    # from this process when it unpickles the batch, so stay up until it has.
    _ = drained.wait()


class WorkerDecode:
    """Run the whole pipeline in a worker process; this one only relays."""

    class Config(Fig["WorkerDecode"]):
        inner: Makeable[DataPipeline] = field(default_factory=DataPipeline.Config)
        seed: int = 0

    def __init__(self, config: Config) -> None:
        self.config = config

    def __iter__(self) -> Iterator[dict[str, object]]:
        context = multiprocessing.get_context("spawn")
        queue = cast("Queue[object]", context.Queue(maxsize=2))
        drained = context.Event()
        worker = context.Process(
            target=_drain_into,
            args=(self.config.inner, self.config.seed, queue, drained),
        )
        worker.start()
        while isinstance(batch := queue.get(), dict):
            # ``_drain_into`` puts only the ``image`` / ``label`` dicts, then None.
            yield cast("dict[str, object]", batch)
        drained.set()
        worker.join()


_CACHE: Final[list[dict[str, object]]] = []


class Cached:
    """Replay batches from a module global once warm, noting hits in a file."""

    class Config(Fig["Cached"]):
        inner: Makeable[DataPipeline] = field(default_factory=DataPipeline.Config)
        log: Path = Path("/dev/null")

    def __init__(self, config: Config) -> None:
        self.config = config

    def __iter__(self) -> Iterator[dict[str, object]]:
        with self.config.log.open("a") as log:
            _ = log.write(f"{'hit' if _CACHE else 'miss'} {os.getpid()}\n")
        if not _CACHE:
            _CACHE.extend(self.config.inner.make())
        yield from _CACHE


class FrontLoaded:
    """Spend the CPU and decode every batch before yielding the first."""

    class Config(Fig["FrontLoaded"]):
        inner: Makeable[DataPipeline] = field(default_factory=DataPipeline.Config)

    def __init__(self, config: Config) -> None:
        self.config = config

    def __iter__(self) -> Iterator[dict[str, object]]:
        _burn(BURN_SEC)
        yield from list(self.config.inner.make())


class LeaksAProcess:
    """Start a process that outlives the pass and is never waited for."""

    class Config(Fig["LeaksAProcess"]):
        inner: Makeable[DataPipeline] = field(default_factory=DataPipeline.Config)

    def __init__(self, config: Config) -> None:
        self.config = config

    def __iter__(self) -> Iterator[dict[str, object]]:
        _ = subprocess.Popen(["sleep", "2"])  # noqa: S607 -- Any sleep will do.
        yield from self.config.inner.make()


def _wrapped(
    root: Path,
    wrapper: WorkerDecode.Config
    | Cached.Config
    | FrontLoaded.Config
    | LeaksAProcess.Config,
) -> LoaderThroughput.Config:
    cfg = _tiny(root)
    wrapper.inner = cfg.pipeline
    # The harness only iterates what ``pipeline`` makes; a wrapper yields the
    # same batches, so it stands in for a DataPipeline here.
    cfg.pipeline = cast("Makeable[DataPipeline]", wrapper)
    return cfg


@pytest.mark.cli_python_subprocess
def test_decode_in_a_worker_process_is_charged_to_the_pass(staged: Path) -> None:
    cfg = _wrapped(staged, WorkerDecode.Config())
    cfg.num_repeats = 1
    started = time.process_time()
    _ = list(cfg.pipeline.make())
    relaying_cpu = time.process_time() - started

    report = cfg.make().measure()

    assert report.mismatches == []
    # The old harness read only its own process's CPU, so the relay alone.
    assert relaying_cpu < BURN_SEC / 2
    assert report.num_images / report.images_per_cpu_sec[0] > BURN_SEC


@pytest.mark.cli_python_subprocess
def test_no_cache_survives_into_another_pass(staged: Path, tmp_path: Path) -> None:
    cfg = _wrapped(staged, Cached.Config(log=tmp_path / "cache.log"))

    report = cfg.make().measure()

    assert report.mismatches == []
    passes = [
        line.split() for line in (tmp_path / "cache.log").read_text().splitlines()
    ]
    assert [state for state, _ in passes] == ["miss"] * cfg.num_repeats
    assert len({pid for _, pid in passes} | {str(os.getpid())}) == cfg.num_repeats + 1


@pytest.mark.cli_python_subprocess
def test_work_before_the_first_batch_is_timed(staged: Path) -> None:
    cfg = _wrapped(staged, FrontLoaded.Config())
    cfg.num_repeats = 1

    report = cfg.make().measure()

    assert report.mismatches == []
    assert report.num_images == 4
    assert report.first_batch_sec[0] > BURN_SEC
    assert report.num_images / report.images_per_cpu_sec[0] > BURN_SEC
    assert report.num_images / report.images_per_sec[0] > BURN_SEC


@pytest.mark.cli_python_subprocess
def test_a_process_left_running_refuses_the_run(staged: Path) -> None:
    cfg = _wrapped(staged, LeaksAProcess.Config())
    cfg.num_repeats = 1
    cfg.survivor_grace_sec = 0.2

    with pytest.raises(RuntimeError, match="outlived its pass"):
        _ = cfg.make().measure()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
