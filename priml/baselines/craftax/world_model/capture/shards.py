"""Shard writer of a capture worker: encode episodes in parallel, publish in order.

Captured episodes are written as replay shards (``archive.py``): each episode's
record, its game snapshots every ``stride`` decisions, and its floor trace,
but not its frames. The encoder first regenerates the frames by replay, which
checks every state hash capture took live, the episode's end, and every token
against the schema, and requires the floor trace capture took to be theirs;
then ``snapshots.snapshot_episode`` takes the snapshots by replaying it again.
So a published shard is known to replay the trajectory capture played, in
tokens the world model reads. Capture takes no frames to fall back on: the
game and replay are one implementation, so an episode that does not replay is
a fault, and its shard fails to publish.

Each episode is encoded on a thread pool as soon as the worker hands it over;
an open shard holds encoded episodes (about 2 bytes per decision).
``buffer_decisions`` bounds the decisions waiting for an encoder:
``ShardStream.add`` blocks while the bound would be exceeded, so the worker,
and the rollout it polls between, wait for the encoders. Replay and zstd
release the GIL, so encoders run in parallel.

One publisher thread writes shards in the order they close, with
``archive.write_replay_shard``, so every split's manifest lists its shards in
index order and a crash leaves no gap. After a failed shard nothing more is
published, and the failure is raised.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

import logging
import threading
import time

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    ReplayEpisode,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.index import floor_trace
from priml.baselines.craftax.world_model.snapshots import (
    snapshot_episode,
)


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.craftax.world_model.capture.env import Captured


logger = logging.getLogger(__name__)


class ShardWriter:
    """Encode episodes on a thread pool and publish closed shards in close order.

    Args:
      provenance: Recorded in every manifest line.
      stride: Decisions between the stored snapshots, a positive multiple of 256.
      compressors: Threads encoding episodes.
      buffer_decisions: Most decisions waiting for encoding; a larger
        episode is admitted only when none waits.

    """

    def __init__(
        self,
        *,
        provenance: dict[str, str],
        stride: int,
        compressors: int,
        buffer_decisions: int,
    ) -> None:
        self.provenance = dict(provenance)
        self.stride = stride
        self.buffer_decisions = buffer_decisions
        self.published: list[tuple[Path, ManifestLine]] = []
        self.started = time.monotonic()
        self.compressors = ThreadPoolExecutor(
            compressors,
            thread_name_prefix="shard-encode",
        )
        self.publisher = ThreadPoolExecutor(1, thread_name_prefix="shard-publish")
        self.shards: list[Future[None]] = []
        self.raw = 0
        self.room = threading.Condition()
        self.failure: BaseException | None = None

    def stream(self, directory: Path, *, index: int, threshold: int) -> ShardStream:
        """Open one split's stream of shards, numbered from ``index``."""
        return ShardStream(self, directory, index=index, threshold=threshold)

    def compress(self, episode: Captured) -> Future[ReplayEpisode]:
        """Queue ``episode`` for encoding, waiting while the buffer is full.

        Args:
          episode: A complete episode as capture hands it over.

        Returns:
          encoded: The episode as its replay shard stores it, once encoded.

        Raises:
          BaseException: An earlier shard failed to publish; the failure itself.

        """
        decisions = len(episode.actions)
        with self.room:
            self.room.wait_for(
                lambda: (
                    self.failure is not None
                    or not self.raw
                    or self.raw + decisions <= self.buffer_decisions
                ),
            )
            if self.failure is not None:
                raise self.failure
            self.raw += decisions
        future = self.compressors.submit(_encode, episode, stride=self.stride)
        future.add_done_callback(partial(self._compressed, decisions))
        return future

    def publish(
        self,
        directory: Path,
        *,
        index: int,
        episodes: list[Future[ReplayEpisode]],
    ) -> None:
        """Queue a closed shard; shards publish in the order they are queued."""
        shard = self.publisher.submit(self._publish, directory, index, episodes)
        shard.add_done_callback(self._published)
        self.shards.append(shard)

    def finish(self) -> list[tuple[Path, ManifestLine]]:
        """Wait for every queued shard.

        Returns:
          published: Each published shard's directory and manifest line, in order.

        Raises:
          BaseException: The first shard that failed; later ones were not published.

        """
        # Each shard's own exception, not ``failure``: a future wakes its waiters
        # before it runs the callback that sets ``failure``. Shards publish one at
        # a time, each callback before the next shard, so the first seen is first.
        for shard in self.shards:
            if (error := shard.exception()) is not None:
                raise error
        if self.failure is not None:
            raise self.failure
        return self.published

    def shutdown(self) -> None:
        """Stop both thread pools, dropping work that has not started."""
        self.compressors.shutdown(wait=True, cancel_futures=True)
        self.publisher.shutdown(wait=True, cancel_futures=True)

    def _compressed(self, decisions: int, future: Future[ReplayEpisode]) -> None:
        """Release an encoded episode's decisions from the buffer."""
        del future
        with self.room:
            self.raw -= decisions
            self.room.notify_all()

    def _publish(
        self,
        directory: Path,
        index: int,
        episodes: list[Future[ReplayEpisode]],
    ) -> None:
        """Write one shard, unless an earlier one failed."""
        if self.failure is not None:
            return
        line = write_replay_shard(
            directory,
            index=index,
            episodes=[episode.result() for episode in episodes],
            stride=self.stride,
            provenance=self.provenance,
        )
        self.published.append((directory, line))
        logger.info(
            "Published %s/%s: %d episodes, %d decisions; %.0f decisions/s overall.",
            directory,
            line.shard,
            line.episodes,
            line.decisions,
            sum(done.decisions for _, done in self.published)
            / (time.monotonic() - self.started),
        )

    def _published(self, shard: Future[None]) -> None:
        """Keep the first failure and wake any ``compress`` waiting for room."""
        if shard.cancelled() or shard.exception() is None:
            return
        with self.room:
            self.failure = self.failure or shard.exception()
            self.room.notify_all()


class ShardStream:
    """One split's open shard: encoded episodes until it reaches its threshold."""

    def __init__(
        self,
        writer: ShardWriter,
        directory: Path,
        *,
        index: int,
        threshold: int,
    ) -> None:
        self.writer = writer
        self.directory = directory
        self.index = index
        self.threshold = threshold
        self.pending: list[Future[ReplayEpisode]] = []
        self.decisions = 0

    def add(self, episode: Captured) -> None:
        """Append an episode; close the shard once it holds ``threshold`` decisions."""
        self.pending.append(self.writer.compress(episode))
        self.decisions += len(episode.actions)
        if self.decisions >= self.threshold:
            self.close_shard()

    def close_shard(self) -> None:
        """Hand the pending episodes to the publisher as the next shard, if any."""
        if not self.pending:
            return
        self.writer.publish(self.directory, index=self.index, episodes=self.pending)
        self.index += 1
        self.pending = []
        self.decisions = 0


def _encode(episode: Captured, *, stride: int) -> ReplayEpisode:
    """Return an episode as its replay shard stores it, once its frames replay."""
    frames = replay.segment(episode, start=0, stop=len(episode.actions))
    if episode.floors != floor_trace(
        aux=frames.aux,
        reward=frames.reward,
        done=frames.done,
    ):
        raise ValueError(
            f"World seed {episode.receipt.world_seed}: the floors capture took "
            "differ from the replayed frames'.",
        )
    return ReplayEpisode(
        receipt=episode.receipt,
        actions=episode.actions,
        hashes=episode.hashes,
        snapshots=snapshot_episode(episode, stride=stride),
        floors=episode.floors,
        summary=episode.summary,
        origin=episode.origin,
        truncated=episode.truncated,
    )
