"""Tests for the replay verifier: true shards pass, damage halts capture, and it waits.

Shards are written from ``replay.record`` episodes into a capture worker's
directory layout: random play on tiny worlds whose clock runs out three
decisions after the reset, or six on a world of odd seed, its kernels run as
Python (``eager``). ``run`` is driven by ``_Timeline``, a fake clock that plays
the launch's workers right after chosen polls, so waiting and timing out take
no wall time.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import copy
import dataclasses

import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import DEFAULT_MAX_TIMESTEPS
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    ReplayEpisode,
    read_manifest,
    read_summaries,
    token_frame,
    write_replay_shard,
    write_shard,
)
from priml.baselines.craftax.world_model.capture import verify
from priml.baselines.craftax.world_model.capture.control import (
    CaptureHaltedError,
    check_halt,
    halt,
    mark_complete,
    mark_started,
)
from priml.baselines.craftax.world_model.capture.seeds import TRAIN
from priml.baselines.craftax.world_model.capture.verify import (
    ReplayMismatchError,
    ReplayVerifier,
    ShardVerdict,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.baselines.craftax.world_model.index import floor_trace
from priml.baselines.craftax.world_model.snapshots import (
    snapshot_episode,
    verify_snapshots,
)
from priml.lib import zstd_compat
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    import numpy as np

    from priml.baselines.craftax.game.state import Array1, EnvState
    from priml.baselines.craftax.world_model.archive import Record


def _world(state: EnvState, rng: Array1[np.uint32]) -> None:
    """Fill the tiny world, its clock running out after 3 decisions, 6 on an odd seed."""
    tiny_world(state, rng, timestep=DEFAULT_MAX_TIMESTEPS - 3 * (1 + int(rng[0]) % 2))


@pytest.fixture(scope="module")
def recorded() -> list[Episode]:
    """Return three random-play episodes, the second the longest."""
    with eager(world=_world):
        episodes = [
            replay.record(world_seed=seed, sampling_seed=seed, max_decisions=6)
            for seed in (10, 13, 4)
        ]
    assert [len(e.actions) for e in episodes] == [3, 6, 3]
    return episodes


@pytest.fixture(autouse=True)
def tiny() -> Generator[None]:
    with eager(world=_world):
        yield


@pytest.fixture
def verifier(tmp_path: Path) -> ReplayVerifier:
    config = ReplayVerifier.Config()
    config.base_dir = None
    config.root = tmp_path / "archive"
    config.log_dir = tmp_path / "verifier"
    config.fraction = 1.0
    config.launch = "launch-1"
    config.workers = 2
    config.poll_seconds = 0.01
    return config.make()


def test_a_true_shard_passes_once(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    _publish(verifier.config.root, recorded)
    (verdict,) = verifier.poll_once()
    assert verdict.shard == "train/arm0/w0/shard-000000"
    assert verdict.episodes == [0, 1, 2]
    assert verdict.decisions == sum(len(e.actions) for e in recorded)
    assert not verdict.mismatch
    assert verifier.poll_once() == []


def test_a_restart_skips_logged_shards(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    _publish(verifier.config.root, recorded[:1])
    assert verifier.poll_once()
    restarted = verifier.config.make()
    assert restarted.poll_once() == []
    _publish(verifier.config.root, recorded[1:2], index=1)
    (verdict,) = restarted.poll_once()
    assert verdict.shard.endswith("shard-000001")


def test_a_later_launch_verifies_a_shard_name_an_earlier_one_logged(
    verifier: ReplayVerifier,
    recorded: list[Episode],
    tmp_path: Path,
) -> None:
    _publish(verifier.config.root, recorded[:1])
    assert verifier.poll_once()
    config = copy.deepcopy(verifier.config)
    config.root = tmp_path / "another-archive"
    config.launch = "launch-2"
    _publish(config.root, recorded[1:2])
    (verdict,) = config.make().poll_once()
    assert verdict.shard == "train/arm0/w0/shard-000000"
    assert verdict.episodes == [0]


def test_a_mutated_action_halts_capture(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    actions = recorded[1].actions.flip(0).contiguous()
    assert not torch.equal(actions, recorded[1].actions)
    mutated = dataclasses.replace(recorded[1], actions=actions)
    _publish(verifier.config.root, [recorded[0], mutated])
    with pytest.raises(
        ReplayMismatchError,
        match=f"world seed {mutated.receipt.world_seed}",
    ):
        verifier.poll_once()
    halt = from_plain(
        loads((verifier.config.root / "HALT.json").read_text()),
        dict[str, object],
    )
    assert halt["shard"] == "train/arm0/w0/shard-000000"
    with pytest.raises(CaptureHaltedError, match="shard-000000"):
        check_halt(verifier.config.root)


def test_a_mutated_token_frame_halts_capture(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    cells = recorded[2].cells.clone()
    cells[len(cells) // 2, 0, 0] ^= 1
    _publish(verifier.config.root, [dataclasses.replace(recorded[2], cells=cells)])
    with pytest.raises(ReplayMismatchError, match=r"\['cells'\] differ"):
        verifier.poll_once()


@pytest.mark.parametrize(
    ("suffix", "match"),
    [
        ("frames.zst", "SHA-256"),
        ("bin.zst", "SHA-256"),
        ("meta.jsonl", "SHA-256"),
        ("bin.zst", "No such file"),
    ],
)
def test_a_damaged_shard_halts_capture(
    verifier: ReplayVerifier,
    recorded: list[Episode],
    suffix: str,
    match: str,
) -> None:
    directory = _publish(verifier.config.root, recorded)
    path = directory / f"shard-000000.{suffix}"
    if match == "No such file":
        path.unlink()
    else:
        payload = bytearray(path.read_bytes())
        payload[len(payload) // 2] ^= 1
        path.write_bytes(bytes(payload))
    with pytest.raises(ReplayMismatchError, match=match):
        verifier.poll_once()
    halt = from_plain(
        loads((verifier.config.root / "HALT.json").read_text()),
        dict[str, object],
    )
    assert match in str(halt["reason"])


def test_damage_outside_the_sampled_episodes_halts_capture(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    verifier.config.fraction = 0.01
    directory = _publish(verifier.config.root, recorded)
    (verdict,) = verifier.poll_once()
    (sampled,) = verdict.episodes
    line = read_manifest(directory)[0]
    other = read_summaries(directory, line)[(sampled + 1) % len(recorded)]
    path = directory / "shard-000000.bin.zst"
    payload = bytearray(path.read_bytes())
    payload[other.bin.offset + other.bin.size // 2] ^= 1
    path.write_bytes(bytes(payload))
    verifier.config.launch = "launch-2"
    with pytest.raises(ReplayMismatchError, match="SHA-256"):
        verifier.config.make().poll_once()


def test_a_true_replay_shard_passes(
    verifier: ReplayVerifier,
    recorded: list[Episode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strides: list[int] = []
    monkeypatch.setattr(
        verify,
        "verify_snapshots",
        partial(_noting_stride, verify_snapshots, strides),
    )
    _publish_replay(verifier.config.root, recorded, strides=[256] * 3)
    (verdict,) = verifier.poll_once()
    assert verdict.episodes == [0, 1, 2]
    assert not verdict.mismatch
    # Each episode is checked at the shard's own stride.
    assert strides == [256] * 3


def test_a_replay_shard_whose_snapshots_miss_the_replay_halts_capture(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    # The episode is shorter than the stride of 256, so its replay yields no
    # snapshot, but it stores one.
    directory = shard_directory(verifier.config.root, split=TRAIN, arm=0, worker=0)
    directory.mkdir(parents=True)
    episode = recorded[0]
    spurious = zstd_compat.compress(bytes(replay.SNAPSHOT_BYTES))
    write_replay_shard(
        directory,
        index=0,
        episodes=[
            ReplayEpisode(
                receipt=episode.receipt,
                actions=episode.actions,
                hashes=episode.hashes,
                snapshots=spurious,
                floors=floor_trace(
                    aux=episode.aux,
                    reward=episode.reward,
                    done=episode.done,
                ),
                summary=episode.summary,
            ),
        ],
        stride=256,
        provenance={},
    )
    with pytest.raises(
        ReplayMismatchError,
        match=f"world seed {episode.receipt.world_seed}: .*snapshots differ",
    ):
        verifier.poll_once()


def test_a_replay_shard_with_an_episode_stored_as_frames_passes(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    directory = _publish_replay(verifier.config.root, recorded[:1], strides=[256])
    framed = recorded[1]
    write_replay_shard(
        directory,
        index=1,
        episodes=[
            ReplayEpisode(
                receipt=framed.receipt,
                actions=framed.actions.flip(0).contiguous(),
                hashes=framed.hashes,
                snapshots=b"",
                floors=floor_trace(
                    aux=framed.aux,
                    reward=framed.reward,
                    done=framed.done,
                ),
                summary=framed.summary,
                frames=token_frame(framed),
            ),
        ],
        stride=256,
        provenance={},
    )
    verdicts = verifier.poll_once()
    assert [v.shard[-12:] for v in verdicts] == ["shard-000000", "shard-000001"]
    assert not any(v.mismatch for v in verdicts)


def test_an_unreadable_manifest_is_retried_once_then_halts(
    verifier: ReplayVerifier,
    recorded: list[Episode],
) -> None:
    directory = _publish(verifier.config.root, recorded[:1])
    with (directory / "MANIFEST.jsonl").open("a") as manifest:
        manifest.write('{"shard": "shard-0000\n')
    assert verifier.poll_once() == []
    assert not (verifier.config.root / "HALT.json").exists()
    with pytest.raises(ReplayMismatchError, match="MANIFEST"):
        verifier.poll_once()


def test_run_does_not_finish_with_a_manifest_it_could_not_read(
    verifier: ReplayVerifier,
    recorded: list[Episode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = verifier.config.root
    _finish(root, recorded[:1], 0, 1)
    with (shard_directory(root, split=TRAIN, arm=0, worker=0) / "MANIFEST.jsonl").open(
        "a",
    ) as manifest:
        manifest.write('{"shard": "shard-0000\n')
    _timeline(verifier, monkeypatch, {})
    with pytest.raises(ReplayMismatchError, match="MANIFEST"):
        verifier.run()
    assert (root / "HALT.json").exists()


def test_run_verifies_what_the_last_worker_publishes_before_it_stops(
    verifier: ReplayVerifier,
    recorded: list[Episode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = verifier.config.root
    _start(root, 0, 1)
    mark_complete(root, launch="launch-1", arm=0, worker=1, decisions=0)
    mark_complete(root, launch="other", arm=0, worker=0, decisions=0)
    # Worker 0 publishes and marks itself complete between a poll and the next
    # marker read: the race a marker read after the poll would lose.
    _timeline(verifier, monkeypatch, {3: partial(_finish, root, recorded, 0)})
    verifier.run()
    assert "train/arm0/w0/shard-000000" in verifier.verified


def test_queued_workers_never_expire_the_verifier(
    verifier: ReplayVerifier,
    recorded: list[Episode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = verifier.config.root
    timeline = _timeline(
        verifier,
        monkeypatch,
        {100: partial(_finish, root, recorded, 0, 1)},
    )
    verifier.run()
    assert timeline.now >= 10 * 600


def test_run_stops_once_another_launch_halts_the_archive(
    verifier: ReplayVerifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = verifier.config.root
    _start(root, 0)
    # The halted worker never completes, so only the halt can end the run early.
    timeline = _timeline(
        verifier,
        monkeypatch,
        {2: lambda: halt(root, shard="train/arm1/w0/shard-000003", reason="x")},
    )
    with pytest.raises(CaptureHaltedError, match="shard-000003"):
        verifier.run()
    assert timeline.now < 600


def test_run_gives_up_when_a_started_worker_is_silent_for_the_longest_episode(
    verifier: ReplayVerifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = verifier.config.root
    _start(root, 0, 1)
    mark_complete(root, launch="launch-1", arm=0, worker=0, decisions=0)
    timeline = _timeline(verifier, monkeypatch, {})
    with pytest.raises(TimeoutError, match="1 of 2 workers of launch launch-1"):
        verifier.run()
    assert 600 < timeline.now <= 660


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("launch", ""),
        ("workers", 0),
        ("slowest_decisions_per_second", 0.0),
        ("fraction", 0.0),
    ],
)
def test_invalid_settings_are_rejected(name: str, value: object) -> None:
    config = ReplayVerifier.Config()
    config.launch = "launch-1"
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        config.make()


class _Timeline:
    """Fake clock of the verifier that runs a scripted event right after a poll."""

    def __init__(
        self,
        verifier: ReplayVerifier,
        events: dict[int, Callable[[], None]],
    ) -> None:
        self.now = 0.0
        self.polls = 0
        self.events = events
        self.poll = verifier.poll_once

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def poll_once(self) -> list[ShardVerdict]:
        verdicts = self.poll()
        self.polls += 1
        if self.polls in self.events:
            self.events.pop(self.polls)()
        return verdicts


def _timeline(
    verifier: ReplayVerifier,
    monkeypatch: pytest.MonkeyPatch,
    events: dict[int, Callable[[], None]],
) -> _Timeline:
    """Drive ``verifier.run`` by ``events``: polls every 60 s, quiet after 600 s."""
    verifier.config.poll_seconds = 60.0
    verifier.config.max_episode_decisions = 6_000
    verifier.config.slowest_decisions_per_second = 10.0
    timeline = _Timeline(verifier, events)
    monkeypatch.setattr(verify, "time", timeline)
    monkeypatch.setattr(verifier, "poll_once", timeline.poll_once)
    return timeline


def _publish(
    root: Path,
    episodes: list[Episode],
    *,
    index: int = 0,
    worker: int = 0,
) -> Path:
    """Publish ``episodes`` as a frame shard of arm 0's worker ``worker``."""
    directory = shard_directory(root, split=TRAIN, arm=0, worker=worker)
    directory.mkdir(parents=True, exist_ok=True)
    write_shard(directory, index=index, episodes=episodes, provenance={})
    return directory


def _publish_replay(root: Path, episodes: list[Episode], *, strides: list[int]) -> Path:
    """Publish ``episodes`` as a replay shard, each snapshotted at its stride."""
    directory = shard_directory(root, split=TRAIN, arm=0, worker=0)
    directory.mkdir(parents=True, exist_ok=True)
    stored = [
        ReplayEpisode(
            receipt=e.receipt,
            actions=e.actions,
            hashes=e.hashes,
            snapshots=snapshot_episode(e, stride=stride),
            floors=floor_trace(aux=e.aux, reward=e.reward, done=e.done),
            summary=e.summary,
        )
        for e, stride in zip(episodes, strides, strict=True)
    ]
    write_replay_shard(directory, index=0, episodes=stored, stride=256, provenance={})
    return directory


def _start(root: Path, *workers: int) -> None:
    """Mark ``workers`` of arm 0 started in launch-1."""
    for worker in workers:
        mark_started(root, launch="launch-1", arm=0, worker=worker)


def _finish(root: Path, episodes: list[Episode], *workers: int) -> None:
    """Start ``workers``, publish the first one's shard, and mark them all complete."""
    _start(root, *workers)
    _publish(root, episodes, worker=workers[0])
    for worker in workers:
        mark_complete(root, launch="launch-1", arm=0, worker=worker, decisions=0)


def _noting_stride(
    check: Callable[..., None],
    strides: list[int],
    record: Record,
    stored: bytes,
    *,
    stride: int,
) -> None:
    """Run ``verify_snapshots``, first noting the stride it checks at."""
    strides.append(stride)
    check(record, stored, stride=stride)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
