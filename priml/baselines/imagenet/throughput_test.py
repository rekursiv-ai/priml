"""Tests for the input pipeline throughput job.

The job's one promise is that a number is reported only for batches equal to
the reference's, and only for work it actually timed. Most tests here are
about the comparison biting; the subprocess tests run three pipelines that
gamed the earlier in-process harness (decode in a worker process, a cache
shared across passes, everything decoded before the first batch) and check
each is now charged for its work. The golden freezes what ``exp000``'s
pipeline makes of fixed JPEG bytes, element by element, so a regression the
digests only flag can be located here.
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

from priml.baselines.imagenet import throughput
from priml.baselines.imagenet.throughput import (
    LoaderThroughput,
    PixelTolerance,
    ReferenceDigests,
    TensorDigest,
    digest_image_set,
)
from priml.baselines.imagenet.throughput_experiments import exp000
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
def minted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage the five JPEGs and mint the tiny reference's digests next to the set.

    The digest file sits beside the image set, never in it: the set's hash
    covers every file under its directory.
    """
    root = tmp_path_factory.mktemp("throughput")
    _write(root, _jpegs())
    _tiny(root).make().mint_reference().write(_digest_file(root))
    return root


def _digest_file(root: Path) -> Path:
    return root.parent / f"{root.name}.sha256"


def _minted(root: Path) -> LoaderThroughput.Config:
    cfg = _tiny(root)
    cfg.reference_digests = _digest_file(root)
    return cfg


def _digests(*batches: dict[str, Tensor]) -> list[dict[str, TensorDigest]]:
    return [{n: TensorDigest.of(t) for n, t in b.items()} for b in batches]


def _pass(
    digests: list[dict[str, TensorDigest]],
    drift: list[dict[str, tuple[float, float]]] | None = None,
) -> throughput._Pass:
    return throughput._Pass(
        digests=digests,
        drift=drift or [],
        num_images=2,
        elapsed_sec=1.0,
        first_batch_sec=0.5,
        cpu_outside_sec=0.0,
    )


IMAGE: Final = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
LABEL: Final = torch.tensor([3, 5])


def test_digests_survive_a_round_trip(tmp_path: Path) -> None:
    digests = ReferenceDigests(
        image_set="ab" * 32,
        batches=_digests(
            {"image": IMAGE, "label": LABEL},
            {"image": IMAGE + 1, "label": LABEL},
        ),
    )
    digests.write(tmp_path / "d.sha256", header="two lines\nof header")

    assert ReferenceDigests.read(tmp_path / "d.sha256") == digests
    assert digests.batches[0]["image"].shape == (2, 3, 4)
    assert digests.batches[0]["image"].dtype == "uint8"


def test_the_image_set_hash_sees_content_and_names(tmp_path: Path) -> None:
    _write(tmp_path, _jpegs())
    before = digest_image_set(tmp_path)
    first = next((tmp_path / "train" / SYNSETS[0]).iterdir())

    _ = first.write_bytes(first.read_bytes()[:-1] + b"\0")
    edited = digest_image_set(tmp_path)
    _ = first.rename(first.with_name("renamed.JPEG"))

    assert len({before, edited, digest_image_set(tmp_path)}) == 3


def test_the_committed_digests_match_exp000s_batch_shape() -> None:
    cfg = exp000()
    digests = ReferenceDigests.read(
        Path(throughput.__file__).with_name(str(cfg.reference_digests)),
    )
    pipeline = cfg.pipeline
    assert isinstance(pipeline, DataPipeline.Config)
    (batcher,) = (p for p in pipeline.processors if isinstance(p, Batcher.Config))
    (crop,) = (
        p
        for p in pipeline.processors
        if isinstance(p, GetRandomResizedCropBoxFromDimensions.Config)
    )

    assert len(digests.image_set) == 64
    assert digests.batches
    for batch in digests.batches:
        assert batch["image"].shape == (batcher.size, 3, *crop.size)
        assert batch["image"].dtype == "uint8"
        assert batch["label"].shape == (batcher.size,)
        assert batch["label"].dtype == "int64"


def test_exact_tier_reports_any_changed_bytes() -> None:
    job = LoaderThroughput(LoaderThroughput.Config())
    moved = IMAGE.clone()
    moved[0, 0, 0] += 1

    lines, _, _ = job._check(
        _digests({"image": IMAGE, "label": LABEL}),
        _pass(_digests({"image": moved, "label": LABEL})),
        repeat=1,
    )

    assert lines == ["pass 1 batch 0 image: bytes differ from the frozen reference"]


def test_shape_and_batch_count_changes_are_reported() -> None:
    job = LoaderThroughput(LoaderThroughput.Config())
    want = _digests({"image": IMAGE, "label": LABEL})

    reshaped, _, _ = job._check(
        want,
        _pass(_digests({"image": IMAGE.reshape(3, 2, 4), "label": LABEL})),
        repeat=0,
    )
    extra, _, _ = job._check(want, _pass(want + want), repeat=0)

    assert reshaped == ["pass 0 batch 0 image: uint8[3, 2, 4] vs uint8[2, 3, 4]"]
    assert extra == ["pass 0: 2 batches vs 1"]


def test_a_tolerance_admits_drift_within_it_and_keeps_labels_exact() -> None:
    cfg = LoaderThroughput.Config()
    cfg.tolerance = PixelTolerance.Config(max_abs_diff=2, max_fraction_differing=0.25)
    job = LoaderThroughput(cfg)
    want = _digests({"image": IMAGE, "label": LABEL})
    moved = _digests({"image": IMAGE + 1, "label": LABEL + 1})

    within, diff, fraction = job._check(
        want,
        _pass(moved, [{"image": (2.0, 0.25)}]),
        repeat=0,
    )
    beyond, _, _ = job._check(want, _pass(moved, [{"image": (3.0, 0.25)}]), repeat=0)

    assert within == ["pass 0 batch 0 label: bytes differ from the frozen reference"]
    assert (diff, fraction) == (2.0, 0.25)
    assert beyond[0] == "pass 0 batch 0 image: max diff 3, 25.0000% differ"


def test_drift_is_the_largest_difference_and_the_share_that_moved() -> None:
    moved = IMAGE.clone()
    moved[1, 2, :2] += torch.tensor([4, 1], dtype=torch.uint8)

    assert throughput._drift(IMAGE, moved) == (4.0, 2 / IMAGE.numel())


def test_a_staged_set_unlike_the_digests_refuses_to_run(tmp_path: Path) -> None:
    _write(tmp_path, _jpegs())
    cfg = _tiny(tmp_path)
    ReferenceDigests(image_set="0" * 64, batches=[]).write(tmp_path / "d.sha256")
    cfg.reference_digests = tmp_path / "d.sha256"

    with pytest.raises(ValueError, match="re-mint the digests"):
        _ = cfg.make().measure()


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
def test_an_unchanged_pipeline_matches_and_counts_every_batch(minted: Path) -> None:
    report = _minted(minted).make().measure()

    assert report.mismatches == []
    # Two batches of two, the first one included.
    assert report.num_images == 4
    assert len(report.images_per_cpu_sec) == 2
    assert all(rate > 0 for rate in report.images_per_cpu_sec)
    assert all(rate > 0 for rate in report.images_per_sec)
    assert all(sec > 0 for sec in report.first_batch_sec)


@pytest.mark.cli_python_subprocess
def test_decode_threads_leave_the_batches_alone(minted: Path) -> None:
    cfg = _minted(minted)
    cfg.num_repeats = 1
    _decode(cfg.pipeline).num_threads = 3

    assert cfg.make().measure().mismatches == []


@pytest.mark.cli_python_subprocess
def test_run_logs_the_score_and_the_exact_tier(
    minted: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _minted(minted)
    cfg.num_repeats = 1

    with caplog.at_level(logging.INFO, logger="priml.baselines.imagenet.throughput"):
        cfg.make().run("--passthrough")

    lines = [r.message for r in caplog.records if "passes of" in r.message]
    assert [m.split(" over ")[0] for m in lines] == [
        "images/cpu-sec (scored)",
        "images/sec (wall, not scored)",
        "first-batch sec",
    ]
    assert all("bit-identical to the frozen reference" in m for m in lines)


@pytest.mark.cli_python_subprocess
def test_run_refuses_to_report_a_rate_for_changed_batches(minted: Path) -> None:
    cfg = _minted(minted)
    cfg.num_repeats = 1
    _decode(cfg.pipeline).flip_p = 0.0

    with pytest.raises(AssertionError, match="batch mismatches"):
        cfg.make().run()


@pytest.mark.cli_python_subprocess
def test_the_accurate_idct_fails_exact_and_passes_a_tier_that_says_so(
    minted: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _minted(minted)
    cfg.num_repeats = 1
    # The accurate IDCT lands a level or two off the fast one exp000 uses.
    _decode(cfg.pipeline).fast_dct = False
    exact = cfg.make().measure()
    cfg.tolerance = PixelTolerance.Config(max_abs_diff=255, max_fraction_differing=1)

    with caplog.at_level(logging.INFO, logger="priml.baselines.imagenet.throughput"):
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
    cfg = _minted(root)
    wrapper.inner = cfg.pipeline
    # The harness only iterates what ``pipeline`` makes; a wrapper yields the
    # same batches, so it stands in for a DataPipeline here.
    cfg.pipeline = cast("Makeable[DataPipeline]", wrapper)
    return cfg


@pytest.mark.cli_python_subprocess
def test_decode_in_a_worker_process_is_charged_to_the_pass(minted: Path) -> None:
    cfg = _wrapped(minted, WorkerDecode.Config())
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
def test_no_cache_survives_into_another_pass(minted: Path, tmp_path: Path) -> None:
    # The log lives outside the image set, whose hash covers every file in it.
    cfg = _wrapped(minted, Cached.Config(log=tmp_path / "cache.log"))
    # A tolerance makes the reference run too, in a process of its own.
    cfg.tolerance = PixelTolerance.Config()

    report = cfg.make().measure()

    assert report.mismatches == []
    passes = [
        line.split() for line in (tmp_path / "cache.log").read_text().splitlines()
    ]
    assert [state for state, _ in passes] == ["miss"] * cfg.num_repeats
    assert len({pid for _, pid in passes} | {str(os.getpid())}) == cfg.num_repeats + 1


@pytest.mark.cli_python_subprocess
def test_work_before_the_first_batch_is_timed(minted: Path) -> None:
    cfg = _wrapped(minted, FrontLoaded.Config())
    cfg.num_repeats = 1

    report = cfg.make().measure()

    assert report.mismatches == []
    assert report.num_images == 4
    assert report.first_batch_sec[0] > BURN_SEC
    assert report.num_images / report.images_per_cpu_sec[0] > BURN_SEC
    assert report.num_images / report.images_per_sec[0] > BURN_SEC


@pytest.mark.cli_python_subprocess
def test_a_process_left_running_refuses_the_run(
    minted: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _wrapped(minted, LeaksAProcess.Config())
    cfg.num_repeats = 1
    monkeypatch.setattr(throughput, "SURVIVOR_GRACE_SEC", 0.2)

    with pytest.raises(RuntimeError, match="outlived its pass"):
        _ = cfg.make().measure()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
