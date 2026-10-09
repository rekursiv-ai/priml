"""Tests for the shard writer: reference bytes, publication order, and the buffer.

Every episode is random play on a tiny world whose clock runs out two
decisions after the reset, its kernels run as Python (``eager``) on the
recorder's thread and the writer's encoders alike.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import dataclasses
import functools
import threading
import time

import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import DEFAULT_MAX_TIMESTEPS
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    read_manifest,
    read_summaries,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.capture import shards
from priml.baselines.craftax.world_model.capture.env import Captured
from priml.baselines.craftax.world_model.capture.shards import (
    ShardWriter,
)
from priml.baselines.craftax.world_model.index import floor_trace
from priml.baselines.craftax.world_model.snapshots import (
    replay_episodes,
    stored_episode,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from concurrent.futures import Future
    from pathlib import Path

    from priml.baselines.craftax.world_model.archive import (
        Episode,
        ReplayEpisode,
    )


_DECISIONS: Final = 1
"""Decisions an episode plays before its clock runs out."""


def _recorded(ordinal: int) -> Episode:
    """Return a random-play episode, with its frames, whose summary names ``ordinal``."""
    episode = replay.record(
        world_seed=100_000_000 + ordinal,
        sampling_seed=ordinal,
        max_decisions=_DECISIONS,
    )
    return dataclasses.replace(
        episode,
        receipt=dataclasses.replace(episode.receipt, arm=3),
        summary={"episode": ordinal},
    )


def _episode(ordinal: int) -> Captured:
    """Return ``_recorded(ordinal)`` as capture hands it over."""
    return _captured(_recorded(ordinal))


def _captured(episode: Episode) -> Captured:
    """Return a recorded episode as capture hands it over."""
    return Captured(
        receipt=episode.receipt,
        actions=episode.actions,
        hashes=episode.hashes,
        floors=floor_trace(aux=episode.aux, reward=episode.reward, done=episode.done),
        summary=episode.summary,
    )


@pytest.fixture(autouse=True)
def tiny() -> Iterator[None]:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - _DECISIONS)):
        yield


@pytest.fixture
def writer() -> Iterator[ShardWriter]:
    opened = ShardWriter(
        provenance={"source": "test"},
        stride=256,
        compressors=3,
        buffer_decisions=1_000,
    )
    yield opened
    opened.shutdown()


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[int, threading.Event]]:
    """Hold the encoding of an episode whose ordinal is a key until its event is set.

    Every event is set at teardown, before the writer shuts down, so a failed
    assertion cannot leave an encoder waiting forever.
    """
    events: dict[int, threading.Event] = {}
    encode: Callable[..., ReplayEpisode] = shards._encode

    def gated(episode: Captured, *, stride: int) -> ReplayEpisode:
        ordinal = episode.summary["episode"]
        assert isinstance(ordinal, int)
        if ordinal in events:
            events[ordinal].wait()
        return encode(episode, stride=stride)

    monkeypatch.setattr(shards, "_encode", gated)
    yield events
    for event in events.values():
        event.set()


def test_files_equal_the_reference_writer(tmp_path: Path, writer: ShardWriter) -> None:
    recorded = [_recorded(ordinal) for ordinal in range(2)]
    episodes = [_captured(episode) for episode in recorded]
    reference, written = tmp_path / "reference", tmp_path / "written"
    reference.mkdir()
    written.mkdir()
    write_replay_shard(
        reference,
        index=7,
        episodes=[stored_episode(e, stride=256) for e in recorded],
        stride=256,
        provenance={"source": "test"},
    )
    stream = writer.stream(written, index=7, threshold=1 << 30)
    for episode in episodes:
        stream.add(episode)
    stream.close_shard()
    assert [line.shard for _, line in writer.finish()] == ["shard-000007"]
    names = sorted(path.name for path in reference.iterdir())
    assert names == sorted(path.name for path in written.iterdir())
    for name in names:
        assert (written / name).read_bytes() == (reference / name).read_bytes(), name
    (line,) = read_manifest(written)
    served = replay_episodes(written, line, summaries=read_summaries(written, line))
    for episode, again in zip(recorded, served, strict=True):
        assert torch.equal(again.cells, episode.cells)


def test_shards_publish_in_close_order(
    tmp_path: Path,
    writer: ShardWriter,
    gate: dict[int, threading.Event],
) -> None:
    gate[0] = threading.Event()
    stream = writer.stream(tmp_path, index=0, threshold=1)
    first, second = _episode(0), _episode(1)
    stream.add(first)
    stream.add(second)
    # Once only episode 0's decisions wait for an encoder, episode 1 is encoded
    # and nothing but the publication order holds shard 1 back; the watch starts
    # there rather than at an add, so a slow encoder cannot use it up.
    deadline = time.monotonic() + 5
    while writer.raw > len(first.actions):
        assert time.monotonic() < deadline, "episode 1 was never encoded"
        time.sleep(0.001)
    deadline = time.monotonic() + 0.05
    while time.monotonic() < deadline:
        assert not read_manifest(tmp_path), "shard 1 was published before shard 0"
        time.sleep(0.01)
    gate[0].set()
    lines = [line for _, line in writer.finish()]
    assert [line.shard for line in read_manifest(tmp_path)] == [
        "shard-000000",
        "shard-000001",
    ]
    assert read_summaries(tmp_path, lines[1])[0].receipt == second.receipt


def test_the_raw_buffer_blocks_the_caller_until_encoding_frees_it(
    tmp_path: Path,
    gate: dict[int, threading.Event],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _episode(0), _episode(1)
    writer = ShardWriter(
        provenance={},
        stride=256,
        compressors=2,
        buffer_decisions=len(first.actions) + len(second.actions) - 1,
    )
    waited = threading.Event()
    monkeypatch.setattr(
        writer.room,
        "wait_for",
        functools.partial(_noting_wait, writer.room.wait_for, waited),
    )
    try:
        gate[0] = threading.Event()
        stream = writer.stream(tmp_path, index=0, threshold=1 << 30)
        stream.add(first)
        blocked = threading.Thread(target=stream.add, args=(second,))
        blocked.start()
        assert waited.wait(timeout=5), "the raw decisions exceeded their bound"
        assert blocked.is_alive()
        assert writer.raw == len(first.actions)
        gate[0].set()
        blocked.join(timeout=5)
        assert not blocked.is_alive()
        stream.close_shard()
        assert writer.finish()[0][1].episodes == 2
    finally:
        writer.shutdown()


def test_a_failed_shard_stops_publication_and_is_raised(
    tmp_path: Path,
    writer: ShardWriter,
) -> None:
    stream = writer.stream(tmp_path, index=0, threshold=1)
    stream.add(dataclasses.replace(_episode(0), summary={"episode": {1}}))
    with pytest.raises(TypeError, match="set"):
        writer.finish()
    assert not read_manifest(tmp_path)
    with pytest.raises(TypeError, match="set"):
        stream.add(_episode(1))


def test_a_failed_last_shard_is_raised_before_its_callback_runs(
    tmp_path: Path,
    writer: ShardWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A future wakes its waiters before it runs its callbacks, so ``finish`` can
    # see the last shard fail while ``failure`` is still unset; holding the
    # callback until ``finish`` has answered makes that window certain.
    answered = threading.Event()
    published = writer._published

    def held(shard: Future[None]) -> None:
        answered.wait(timeout=5)
        published(shard)

    monkeypatch.setattr(writer, "_published", held)
    stream = writer.stream(tmp_path, index=0, threshold=1)
    try:
        stream.add(dataclasses.replace(_episode(0), summary={"episode": {1}}))
        with pytest.raises(TypeError, match="set"):
            writer.finish()
    finally:
        answered.set()
    assert not read_manifest(tmp_path)


def test_an_episode_that_does_not_replay_fails_its_shard(
    tmp_path: Path,
    writer: ShardWriter,
) -> None:
    # Capture takes no frames to store instead: the game and replay are one
    # implementation, so a hash replay misses is a fault, not a corpus quirk.
    episodes = [_episode(ordinal) for ordinal in range(3)]
    hashes = episodes[1].hashes.clone()
    hashes[-1] ^= 1
    episodes[1] = dataclasses.replace(episodes[1], hashes=hashes)
    stream = writer.stream(tmp_path, index=0, threshold=1 << 30)
    for episode in episodes:
        stream.add(episode)
    stream.close_shard()
    with pytest.raises(ValueError, match="hash"):
        writer.finish()
    assert not read_manifest(tmp_path)


def test_an_episode_whose_floors_replay_does_not_give_fails_its_shard(
    tmp_path: Path,
    writer: ShardWriter,
) -> None:
    episode = _episode(0)
    wrong = dataclasses.replace(episode.floors, died=not episode.floors.died)
    stream = writer.stream(tmp_path, index=0, threshold=1)
    stream.add(dataclasses.replace(episode, floors=wrong))
    with pytest.raises(ValueError, match="floors capture took"):
        writer.finish()
    assert not read_manifest(tmp_path)


def _noting_wait(
    wait_for: Callable[[Callable[[], bool], float | None], bool],
    waited: threading.Event,
    predicate: Callable[[], bool],
    timeout: float | None = None,
) -> bool:
    """Run ``Condition.wait_for``, first setting ``waited`` if the caller must wait."""
    if not predicate():
        waited.set()
    return wait_for(predicate, timeout)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
