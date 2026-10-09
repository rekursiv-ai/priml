"""Tests for the capture worker: splits, the budget, resumption, halts and markers.

The worker plays ``RandomSource``: random legal play, its kernels run as
Python (``eager``) on tiny worlds whose clock runs out at the first decision
after the reset, or the second on a world of odd seed, so a test captures real
episodes in moments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import shutil

import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import DEFAULT_MAX_TIMESTEPS
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    Receipt,
    ReplayEpisode,
    read_manifest,
    read_summaries,
    write_replay_shard,
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
    from collections.abc import Generator
    from pathlib import Path

    import numpy as np

    from priml.baselines.craftax.game.state import Array1, EnvState


_DECISIONS: Final = 1
"""Decisions an episode of an even world seed plays; an odd one plays twice as many."""


def _world(state: EnvState, rng: Array1[np.uint32]) -> None:
    """Fill the tiny world, its clock running out after 1 decision, 2 on an odd seed."""
    tiny_world(
        state,
        rng,
        timestep=DEFAULT_MAX_TIMESTEPS - _DECISIONS * (1 + int(rng[0]) % 2),
    )


@pytest.fixture(autouse=True)
def cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture as a host without CUDA does: unpinned buffers, no CUDA context to start.

    On a GPU host the first pinned buffer of a process starts the CUDA context,
    120 ms on the x86 host, which these tests of play and recording never use.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def tiny() -> Generator[None]:
    with eager(world=_world):
        yield


def test_a_worker_publishes_replayable_shards_of_both_splits(tmp_path: Path) -> None:
    # Ordinal 18 is published: the worker resumes at 19, every twentieth
    # episode's, a validation one, and 20, a training one.
    train = shard_directory(tmp_path, split=TRAIN, arm=1, worker=2)
    _publish(train, index=0, ordinals=(18,))
    report = _worker(tmp_path, decisions=_DECISIONS + 1).capture()
    ordinals: list[int] = []
    decisions = 0
    for split in (TRAIN, VALIDATION):
        directory = shard_directory(tmp_path, split=split, arm=1, worker=2)
        for line in read_manifest(directory)[split == TRAIN :]:
            assert line.snapshot_stride == 256
            assert line.provenance["policy"] == "uniform-legal"
            summaries = read_summaries(directory, line)
            assert {s.receipt.split for s in summaries} == {split}
            ordinals += [from_plain(s.summary["episode"], int) for s in summaries]
            decisions += line.decisions
            for episode in replay_episodes(directory, line, summaries=summaries):
                assert replay.verify(episode) == replay.MATCHED
    assert ordinals == [20, 19]
    assert (report.episodes, report.decisions) == (2, decisions)


def test_shards_close_at_the_first_episode_boundary_after_the_threshold(
    tmp_path: Path,
) -> None:
    threshold = _DECISIONS + 1
    _worker(tmp_path, decisions=3 * _DECISIONS, shard_decisions=threshold).capture()
    directory = shard_directory(tmp_path, split=TRAIN, arm=1, worker=2)
    lines = read_manifest(directory)
    assert len(lines) >= 2
    for line in lines[:-1]:
        summaries = read_summaries(directory, line)
        assert line.decisions - summaries[-1].decisions < threshold <= line.decisions


def test_resume_continues_shards_and_episode_ordinals(tmp_path: Path) -> None:
    first = _worker(tmp_path, decisions=1, shard_decisions=_DECISIONS).capture()
    budget = first.decisions + 1
    second = _worker(tmp_path, decisions=budget, shard_decisions=_DECISIONS).capture()
    assert second.decisions > 0
    directory = shard_directory(tmp_path, split=TRAIN, arm=1, worker=2)
    names = [line.shard for line in read_manifest(directory)]
    assert names == [f"shard-{i:06d}" for i in range(len(names))]
    assert len(names) > len(first.shards)
    ordinals = [
        from_plain(s.summary["episode"], int)
        for split in (TRAIN, VALIDATION)
        for directory in [shard_directory(tmp_path, split=split, arm=1, worker=2)]
        for line in read_manifest(directory)
        for s in read_summaries(directory, line)
    ]
    assert sorted(ordinals) == list(range(first.episodes + second.episodes))


def test_a_met_budget_starts_nothing_but_marks_the_launch_complete(
    tmp_path: Path,
) -> None:
    _worker(tmp_path, decisions=1).capture()
    again = _worker(tmp_path, decisions=1, launch="x").capture()
    assert again.episodes == 0
    assert started(tmp_path, launch="x") == completed(tmp_path, launch="x") == 1


def test_a_copy_of_a_workers_later_shard_is_not_resumed(tmp_path: Path) -> None:
    owner = shard_directory(tmp_path / "owner", split=TRAIN, arm=1, worker=2)
    _publish(owner, index=0, ordinals=(0,))
    _publish(owner, index=1, ordinals=(1,))
    copy = tmp_path / "copy"
    target = shard_directory(copy, split=TRAIN, arm=1, worker=2)
    shutil.copytree(owner, target)
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
    # Worlds 2 and 1 in turn: ordinal 1 plays an odd world past the cap, and
    # ordinals 0 and 2, on world 2, end within it before capture stops.
    worker = _worker(tmp_path, decisions=100_000, max_decisions=_DECISIONS)
    source = worker.config.source
    assert isinstance(source, RandomSource.Config)
    source.env.world_seeds = (2, 1)
    with pytest.raises(CaptureError, match=f"exceeds {_DECISIONS}"):
        worker.capture()
    published = [
        (from_plain(summary.summary["episode"], int), summary.decisions)
        for split in (TRAIN, VALIDATION)
        for directory in [shard_directory(tmp_path, split=split, arm=1, worker=2)]
        for line in read_manifest(directory)
        for summary in read_summaries(directory, line)
    ]
    assert sorted(published) == [(0, _DECISIONS), (2, _DECISIONS)]


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
    """Return arm 1's worker 2 capturing random play on two rows into ``root``."""
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
    # polls that end no episode only idles.
    config.poll_seconds = 0.0
    source = config.source = RandomSource.Config(steps_per_poll=_DECISIONS)
    source.env.num_envs = 2
    source.env.max_decisions = max_decisions
    return CaptureWorker(config)


def _publish(directory: Path, *, index: int, ordinals: tuple[int, ...]) -> None:
    """Publish a replay shard of one-decision episodes, as a worker that ran before."""
    directory.mkdir(parents=True, exist_ok=True)
    write_replay_shard(
        directory,
        index=index,
        episodes=[
            ReplayEpisode(
                receipt=Receipt(
                    world_seed=1,
                    sampling_seed=ordinal,
                    initial_state_hash=0,
                    arm=1,
                    split=TRAIN,
                ),
                actions=torch.zeros(1, dtype=torch.uint8),
                hashes=torch.zeros(2, dtype=torch.int64),
                snapshots=b"",
                floors=FloorTrace(changes=((0, 0),), died=False),
                summary={"episode": ordinal},
            )
            for ordinal in ordinals
        ],
        stride=256,
        provenance={},
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
