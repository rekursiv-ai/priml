"""Replay shards: take, check, and serve the game snapshots frames replay from.

A replay shard (``archive.py``) stores, per episode, its record and the
snapshots before decisions ``S``, ``2 * S``, ... of its length ``T``, ``S`` the
shard's ``snapshot_stride``; the start before decision 0 comes from the seed,
or a branch's origin. Each snapshot is XORed with the episode's start, which
zeroes everything play left unchanged, and compressed as its own zstd frame.

- ``snapshot_episode`` replays an episode from its start, checking every
  state hash, and returns its stored snapshots; ``check_episode`` regenerates
  every frame from them, each stretch from its own snapshot, and compares it
  with the frames it was captured with; ``verify_snapshots`` regenerates them
  from the record alone and compares them with the stored ones;
  ``replay_episodes`` reads whole episodes with their frames from a shard of
  either format.
- ``stored_episode`` keeps an episode's snapshots once every frame replays
  from them, or else its frames (``archive.py``): it is how an episode with
  frames goes into ``write_replay_shard``; capture's, which hold none, go in
  by ``snapshot_episode`` alone (``capture/shards.py``).
- ``ReplayCache`` is the ``data.FrameSource`` of replay shards: it replays the
  aligned blocks of ``block_decisions`` a window needs, from the latest
  snapshot before them, and keeps the most recent blocks, so the windows of a
  long episode share its replay.

Replay is the game's own (``replay.py``), and a snapshot is the State's bytes
and its stream, so the first datasets' replay shards, v1 and v2 alike, read
here. A restored snapshot is checked by
the state hash of its decision, a multiple of 256, before use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import collections
import dataclasses
import logging
import threading

from torch import Tensor

import torch

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    ManifestLine,
    Record,
    ReplayEpisode,
    read_episodes,
    read_records,
    read_snapshots,
    read_summaries,
    token_frame,
)
from priml.baselines.craftax.world_model.batch import Segment
from priml.baselines.craftax.world_model.index import (
    floor_trace,
    trace_strata,
)
from priml.lib import zstd_compat


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


logger = logging.getLogger(__name__)


def snapshot_episode(
    episode: replay.Replayable,
    *,
    stride: int,
    level: int = 3,
) -> bytes:
    """Replay one episode, checking every hash, and return its stored snapshots.

    Args:
      episode: The episode's record.
      stride: Decisions between snapshots, a positive multiple of 256.
      level: zstd level.

    Returns:
      stored: The snapshots before decisions ``stride``, ``2 * stride``, ...,
        each XORed with the episode's start and compressed as its own zstd
        frame, back to back.

    """
    base, *rest = replay.snapshots(episode, stride=stride)
    return b"".join(
        zstd_compat.compress(replay.xor_bytes(s.state, right=base.state), level=level)
        for s in rest
    )


def decode_snapshots(
    stored: bytes,
    *,
    base: replay.Snapshot,
    stride: int,
) -> list[replay.Snapshot]:
    """Return an episode's snapshots from its stored bytes.

    Args:
      stored: The episode's stored snapshots, from ``snapshot_episode``.
      base: The episode's start, ``replay.initial``.
      stride: The shard's snapshot stride.

    Returns:
      snapshots: The snapshots before decisions ``stride``, ``2 * stride``, ...

    Raises:
      ValueError: The bytes do not hold whole snapshots of this game.

    """
    raw = zstd_compat.decompress(stored) if stored else b""
    size = len(base.state)
    if len(raw) % size:
        raise ValueError("Stored snapshots are not whole snapshots of this game.")
    return [
        replay.Snapshot(
            decision=(k + 1) * stride,
            state=replay.xor_bytes(
                raw[k * size : (k + 1) * size],
                right=base.state,
            ),
        )
        for k in range(len(raw) // size)
    ]


def check_episode(episode: Episode, stored: bytes, *, stride: int) -> int:
    """Regenerate every frame of an episode from its stored snapshots and compare.

    Each stretch between snapshots is replayed from its own decoded snapshot,
    the first from the start, so every stored snapshot is exercised.

    Args:
      episode: The episode with the frames it was captured with.
      stored: The episode's stored snapshots, from ``snapshot_episode``.
      stride: The shard's snapshot stride.

    Returns:
      decisions: Decisions compared, the whole episode.

    Raises:
      ValueError: A hash does not match or a regenerated frame differs.

    """
    decisions = len(episode.actions)
    decoded = decode_snapshots(stored, base=replay.initial(episode), stride=stride)
    starts = range(0, decisions, stride)
    if len(starts) != len(decoded) + 1:
        raise ValueError(
            f"{len(decoded)} snapshots do not cover {decisions} decisions.",
        )
    for start, snapshot in zip(starts, [None, *decoded], strict=True):
        stop = min(start + stride, decisions)
        frames = replay.segment(episode, start=start, stop=stop, snapshot=snapshot)
        for name in ("cells", "aux", "reward", "done"):
            if not torch.equal(
                cast("torch.Tensor", getattr(frames, name)),
                cast("torch.Tensor", getattr(episode, name))[start:stop],
            ):
                raise ValueError(
                    f"World seed {episode.receipt.world_seed}: {name} frames differ "
                    f"in decisions [{start}, {stop}).",
                )
    return decisions


def verify_snapshots(record: replay.Replayable, stored: bytes, *, stride: int) -> None:
    """Replay a record from its start and require the stored snapshots it yields.

    Args:
      record: The episode's record; every state hash is checked.
      stored: The episode's stored snapshots.
      stride: The shard's snapshot stride.

    Raises:
      ValueError: A hash does not match, or a stored snapshot differs from the
        state replay reaches at its decision.

    """
    base, *replayed = replay.snapshots(record, stride=stride)
    if decode_snapshots(stored, base=base, stride=stride) != replayed:
        raise ValueError(
            f"World seed {record.receipt.world_seed}: stored snapshots differ from "
            "the replayed states.",
        )


def replay_episodes(
    directory: Path,
    line: ManifestLine,
    *,
    summaries: Sequence[EpisodeSummary],
) -> list[Episode]:
    """Read chosen whole episodes of a shard of either format, with their frames.

    Args:
      directory: The shard's worker directory.
      line: The shard's manifest line.
      summaries: The episodes to read, from ``read_summaries``.

    Returns:
      episodes: The chosen episodes in the order given, with their stored frames
        or, where a replay shard stores none, frames replayed from the start
        with every hash checked.

    """
    if not line.snapshot_stride:
        return read_episodes(directory, line, summaries=summaries)
    empty = torch.empty(0)
    records = read_records(directory, line, summaries=summaries)
    return [
        replay.replay(
            Episode(
                receipt=r.receipt,
                actions=r.actions,
                hashes=r.hashes,
                cells=empty,
                aux=empty,
                reward=empty,
                done=empty,
                summary=s.summary,
                origin=r.origin,
                truncated=r.truncated,
            ),
        )
        if s.replayed
        else read_episodes(directory, line, summaries=[s])[0]
        for r, s in zip(records, summaries, strict=True)
    ]


def stored_episode(episode: Episode, *, stride: int) -> ReplayEpisode:
    """Return an episode as a replay shard stores it: snapshots, else its frames.

    Its snapshots are kept once every frame replays from them
    (``check_episode``). An episode whose frames replay does not reproduce
    keeps its frames instead, logged as a warning.

    Args:
      episode: A complete episode with the frames it was captured with.
      stride: The shard's snapshot stride, a positive multiple of 256.

    Returns:
      stored: The episode with its snapshots, or with its frames.

    Raises:
      ValueError: The stride is not a positive multiple of 256.

    """
    # Checked here: the replay below would reject it too, as a replay failure.
    if stride <= 0 or stride % replay.HASH_STRIDE:
        raise ValueError(f"Snapshot stride {stride} is not a multiple of 256.")
    stored = ReplayEpisode(
        receipt=episode.receipt,
        actions=episode.actions,
        hashes=episode.hashes,
        snapshots=b"",
        floors=floor_trace(aux=episode.aux, reward=episode.reward, done=episode.done),
        summary=episode.summary,
        origin=episode.origin,
        truncated=episode.truncated,
    )
    try:
        snapshots = snapshot_episode(episode, stride=stride)
        check_episode(episode, snapshots, stride=stride)
    except ValueError as error:
        logger.warning(
            "World seed %d, %d decisions, does not replay; its frames are stored: %s",
            episode.receipt.world_seed,
            len(episode.actions),
            error,
        )
        return dataclasses.replace(stored, frames=token_frame(episode))
    return dataclasses.replace(stored, snapshots=snapshots)


class ReplayCache:
    """Regenerate replay shards' frames on demand, keeping recent blocks; thread-safe.

    Frames are regenerated in aligned blocks of ``block_decisions``, from the
    latest stored snapshot at or before a missing block, or from the start.
    Strata come from the summaries' floor traces.

    Args:
      entries: Each replay shard's directory and manifest line.
      capacity: Decisions of regenerated blocks to hold; the blocks of the
        latest request are kept even when they alone exceed it.
      block_decisions: Decisions per block.
      cached_episodes: Episodes whose records, snapshots, and strata are held.

    """

    def __init__(
        self,
        entries: Sequence[tuple[Path, ManifestLine]],
        *,
        capacity: int,
        block_decisions: int = 4_096,
        cached_episodes: int = 1_024,
    ) -> None:
        if not all(line.snapshot_stride for _, line in entries):
            raise ValueError("A replay cache serves replay shards only.")
        if block_decisions <= 0:
            raise ValueError(f"block_decisions={block_decisions} must be positive.")
        self.entries = list(entries)
        self.capacity = capacity
        self.block_decisions = block_decisions
        self.cached_episodes = cached_episodes
        self._lock = threading.Lock()
        self._summaries: dict[int, list[EpisodeSummary]] = {}
        self._episodes: collections.OrderedDict[tuple[int, int], _Replayable] = (
            collections.OrderedDict()
        )
        self._blocks: collections.OrderedDict[tuple[int, int, int], replay.Frames] = (
            collections.OrderedDict()
        )
        self._decisions = 0

    @property
    def resident(self) -> list[tuple[int, int, int]]:
        """Return the held ``(shard, episode, block)`` keys, least recent first."""
        return list(self._blocks)

    def episodes(self, shard: int) -> int:
        """Return the episode count of one shard."""
        with self._lock:
            return len(self._shard_summaries(shard))

    def decisions(self, shard: int, episode: int) -> int:
        """Return the decision count of one episode."""
        with self._lock:
            return self._shard_summaries(shard)[episode].decisions

    def starts_at_reset(self, shard: int, episode: int) -> bool:
        """Return whether an episode starts at its world's reset, not a branch's origin."""
        with self._lock:
            return not self._episode(shard, episode).record.origin

    def segment(
        self,
        shard: int,
        episode: int,
        *,
        start: int,
        stop: int,
    ) -> tuple[Segment, Tensor]:
        """Return decisions ``[start, stop)`` of one episode; see ``FrameSource``.

        Args:
          shard: Position of the shard in ``entries``.
          episode: Episode within the shard.
          start: First decision.
          stop: One past the last decision; may equal ``start``.

        Returns:
          segment: The decisions and the frames of ``[start, stop]`` that exist.
          strata: Stratum of each of those frames' decisions.

        """
        size = self.block_decisions
        with self._lock:
            replayable = self._episode(shard, episode)
            end = min(stop + 1, len(replayable.record.actions))
            blocks = range(start // size, (end - 1) // size + 1)
            missing = [b for b in blocks if (shard, episode, b) not in self._blocks]
            if missing:
                self._replay(shard, episode, replayable, missing[0], missing[-1])
            parts = [self._block(shard, episode, b) for b in blocks]
            self._evict(keep=len(parts))
        first = start - blocks[0] * size
        frames = slice(first, first + end - start)
        count = stop - start
        segment = Segment(
            cells=torch.cat([p.cells for p in parts])[frames],
            aux=torch.cat([p.aux for p in parts])[frames],
            actions=replayable.record.actions[start:stop],
            reward=torch.cat([p.reward for p in parts])[frames][:count],
            done=torch.cat([p.done for p in parts])[frames][:count],
            starts_episode=start == 0 and not replayable.record.origin,
        )
        return segment, replayable.strata[start:end].long()

    def _episode(self, shard: int, episode: int) -> _Replayable:
        """Return one episode's record, snapshots, and strata; hold the lock."""
        key = (shard, episode)
        if key in self._episodes:
            self._episodes.move_to_end(key)
            return self._episodes[key]
        directory, line = self.entries[shard]
        summary = self._shard_summaries(shard)[episode]
        (record,) = read_records(directory, line, summaries=[summary])
        if summary.floors is None:
            raise ValueError("Only a replay shard's summaries hold floors.")
        replayable = _Replayable(
            record=record,
            summary=summary,
            strata=trace_strata(summary.floors, decisions=summary.decisions).to(
                torch.uint8,
            ),
        )
        self._episodes[key] = replayable
        while len(self._episodes) > self.cached_episodes:
            self._episodes.popitem(last=False)
        return replayable

    def _replay(
        self,
        shard: int,
        episode: int,
        replayable: _Replayable,
        first: int,
        last: int,
    ) -> None:
        """Regenerate blocks ``first`` through ``last`` of one episode; hold the lock."""
        size = self.block_decisions
        start = first * size
        stop = min((last + 1) * size, len(replayable.record.actions))
        frames = self._frames(shard, replayable, start=start, stop=stop)
        for block in range(first, last + 1):
            rows = slice((block - first) * size, (block - first + 1) * size)
            part = replay.Frames(
                cells=frames.cells[rows],
                aux=frames.aux[rows],
                reward=frames.reward[rows],
                done=frames.done[rows],
            )
            key = (shard, episode, block)
            if key in self._blocks:
                self._decisions -= len(self._blocks[key].done)
            self._blocks[key] = part
            self._decisions += len(part.done)

    def _frames(
        self,
        shard: int,
        replayable: _Replayable,
        *,
        start: int,
        stop: int,
    ) -> replay.Frames:
        """Return frames ``[start, stop)``, replayed or stored; hold the lock."""
        directory, line = self.entries[shard]
        summary = replayable.summary
        if not summary.replayed:
            (stored,) = read_episodes(directory, line, summaries=[summary])
            return replay.Frames(
                cells=stored.cells[start:stop],
                aux=stored.aux[start:stop],
                reward=stored.reward[start:stop],
                done=stored.done[start:stop],
            )
        k = start // line.snapshot_stride
        snapshot = None
        if k:
            (stored_snapshots,) = read_snapshots(directory, line, summaries=[summary])
            snapshot = decode_snapshots(
                stored_snapshots,
                base=replay.initial(replayable.record),
                stride=line.snapshot_stride,
            )[k - 1]
        return replay.segment(
            replayable.record,
            start=start,
            stop=stop,
            snapshot=snapshot,
        )

    def _block(self, shard: int, episode: int, block: int) -> replay.Frames:
        """Return one held block, marking it most recent; hold the lock."""
        key = (shard, episode, block)
        self._blocks.move_to_end(key)
        return self._blocks[key]

    def _evict(self, *, keep: int) -> None:
        """Drop least recent blocks past the capacity, keeping the last ``keep``."""
        while self._decisions > self.capacity and len(self._blocks) > keep:
            _, evicted = self._blocks.popitem(last=False)
            self._decisions -= len(evicted.done)

    def _shard_summaries(self, shard: int) -> list[EpisodeSummary]:
        """Return one shard's summaries, reading them on first use; hold the lock."""
        if shard not in self._summaries:
            self._summaries[shard] = [
                dataclasses.replace(s, summary={})
                for s in read_summaries(*self.entries[shard])
            ]
        return self._summaries[shard]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Replayable:
    """What a cache holds of one episode to replay it.

    Attributes:
      record: The episode's record.
      summary: Its summary, locating its snapshots or stored frames.
      strata: Stratum of each decision, uint8 ``[T]``.

    """

    record: Record
    summary: EpisodeSummary
    strata: Tensor
