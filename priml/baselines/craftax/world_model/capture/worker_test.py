"""Tests for the capture worker: splits, the budget, resumption, halts and markers.

The worker plays ``RandomSource``: random legal play, whose episodes are a few
hundred decisions, so a test captures a handful of real episodes in moments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil

import pytest
import torch

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    read_manifest,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.control import (
    CaptureHaltedError,
    completed,
    halt,
    started,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.source import (
    CaptureError,
    RandomSource,
)
from priml.baselines.craftax.world_model.capture.worker import (
    CaptureWorker,
    shard_directory,
)
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from pathlib import Path


# Every twentieth started episode is validation, so both splits need 20 real
# random-play episodes, about 4,000 decisions captured, encoded and replayed:
# 0.19 s warm on x86.
@pytest.mark.compute_large_fixture
def test_a_worker_publishes_replayable_shards_of_both_splits(tmp_path: Path) -> None:
    report = _worker(tmp_path, decisions=4_000, shard_decisions=1_500).capture()
    ordinals: list[int] = []
    for split in (TRAIN, VALIDATION):
        directory = shard_directory(tmp_path, split=split, arm=1, worker=2)
        for line in read_manifest(directory):
            assert line.snapshot_stride == 256
            assert line.provenance["policy"] == "uniform-legal"
            summaries = read_summaries(directory, line)
            assert {s.receipt.split for s in summaries} == {split}
            ordinals += [from_plain(s.summary["episode"], int) for s in summaries]
            for episode in replay_episodes(directory, line, summaries=summaries):
                assert replay.verify(episode) == replay.MATCHED
    assert sorted(ordinals) == list(range(len(ordinals)))
    assert len(ordinals) >= 20
    assert report.episodes == len(ordinals)
    assert report.decisions >= 4_000


def test_shards_close_at_the_first_episode_boundary_after_the_threshold(
    tmp_path: Path,
) -> None:
    threshold = 500
    _worker(tmp_path, decisions=1_500, shard_decisions=threshold).capture()
    lines = read_manifest(shard_directory(tmp_path, split=TRAIN, arm=1, worker=2))
    assert len(lines) >= 2
    for line in lines[:-1]:
        summaries = read_summaries(
            shard_directory(tmp_path, split=TRAIN, arm=1, worker=2),
            line,
        )
        assert line.decisions - summaries[-1].decisions < threshold <= line.decisions


def test_resume_continues_shards_and_episode_ordinals(tmp_path: Path) -> None:
    first = _worker(tmp_path, decisions=1, shard_decisions=500).capture()
    budget = first.decisions + 1
    second = _worker(tmp_path, decisions=budget, shard_decisions=500).capture()
    assert second.decisions > 0
    directory = shard_directory(tmp_path, split=TRAIN, arm=1, worker=2)
    names = [line.shard for line in read_manifest(directory)]
    assert names == [f"shard-{i:06d}" for i in range(len(names))]
    ordinals = [
        from_plain(s.summary["episode"], int)
        for split in (TRAIN, VALIDATION)
        for directory in [shard_directory(tmp_path, split=split, arm=1, worker=2)]
        for line in read_manifest(directory)
        for s in read_summaries(directory, line)
    ]
    assert len(set(ordinals)) == len(ordinals) == first.episodes + second.episodes


def test_a_met_budget_starts_nothing_but_marks_the_launch_complete(
    tmp_path: Path,
) -> None:
    _worker(tmp_path, decisions=1_000).capture()
    again = _worker(tmp_path, decisions=1_000, launch="x").capture()
    assert again.episodes == 0
    assert started(tmp_path, launch="x") == completed(tmp_path, launch="x") == 1


def test_a_copy_of_a_workers_later_shard_is_not_resumed(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    _worker(owner, decisions=1_500, shard_decisions=500).capture()
    copy = tmp_path / "copy"
    source = shard_directory(owner, split=TRAIN, arm=1, worker=2)
    target = shard_directory(copy, split=TRAIN, arm=1, worker=2)
    shutil.copytree(source, target)
    for path in target.glob("shard-000000.*"):
        path.unlink()
    with pytest.raises(ValueError, match="not the owning copy"):
        _worker(copy, decisions=10_000).capture()


def test_a_halted_archive_stops_capture_before_it_starts(tmp_path: Path) -> None:
    halt(tmp_path, shard="train/arm3/w0/shard-000000", reason="test")
    with pytest.raises(CaptureHaltedError, match="test"):
        _worker(tmp_path, decisions=1_000, launch="x").capture()
    assert not started(tmp_path, launch="x")


def test_a_failure_publishes_every_episode_taken_and_is_raised(tmp_path: Path) -> None:
    worker = _worker(tmp_path, decisions=100_000, max_decisions=200)
    with pytest.raises(CaptureError, match="exceeds 200"):
        worker.capture()
    published = [
        summary.decisions
        for split in (TRAIN, VALIDATION)
        for directory in [shard_directory(tmp_path, split=split, arm=1, worker=2)]
        for line in read_manifest(directory)
        for summary in read_summaries(directory, line)
    ]
    # Two random-play episodes end within 200 decisions before one exceeds it.
    assert len(published) == 2
    assert max(published) <= 200


class _BuiltError(Exception):
    """Raised by the patched ``RandomSource`` with the device it was built under."""


def test_a_source_left_at_no_device_is_built_on_the_best_accelerator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``meta`` is never the host's own best device, so only the worker's choice
    # can put the build there.
    monkeypatch.setattr(
        "priml.baselines.craftax.world_model.capture.source.best_device",
        lambda: torch.device("meta"),
    )
    monkeypatch.setattr(RandomSource, "__init__", _report_build_device)
    with pytest.raises(_BuiltError) as built:
        _worker(tmp_path, decisions=1_000).capture()
    assert built.value.args == (torch.device("meta"),)


def _report_build_device(self: RandomSource, config: RandomSource.Config) -> None:
    """Stand in for ``RandomSource.__init__``: raise the device it is built under."""
    del self, config
    raise _BuiltError(torch.get_default_device())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("arm", 4, "arm"),
        ("worker", -1, "worker"),
        ("decisions", 0, "decisions"),
        ("snapshot_stride", 100, "multiple of 256"),
    ],
)
def test_invalid_settings_are_rejected(
    tmp_path: Path,
    name: str,
    value: int,
    message: str,
) -> None:
    config = _worker(tmp_path, decisions=1).config
    setattr(config, name, value)
    with pytest.raises(ValueError, match=message):
        CaptureWorker(config)


def _worker(
    root: Path,
    *,
    decisions: int,
    shard_decisions: int = 10_000_000,
    launch: str = "",
    max_decisions: int = 100_000,
) -> CaptureWorker:
    """Return arm 1's worker 2 capturing random play into ``root``."""
    config = CaptureWorker.Config()
    config.root = root
    config.run_root = root / "runs"
    config.arm = 1
    config.worker = 2
    config.decisions = decisions
    config.shard_decisions = shard_decisions
    config.snapshot_stride = 256
    config.launch = launch
    config.compressors = 2
    # Random play steps its environments inside every poll, so the wait between
    # polls that end no episode only idles: 0.05 s of it was most of each test.
    config.poll_seconds = 0.0
    source = config.source = RandomSource.Config()
    source.env.num_envs = 4
    source.env.max_decisions = max_decisions
    return CaptureWorker(config)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
