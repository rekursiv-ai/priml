"""Per-shard stratum index: every decision's floor and event, as run-length spans.

A stratum is a floor 0–8 crossed with an event:

- ``entry``: one of the first 64 decisions the episode spends on that floor,
  counted over all its visits to the floor;
- ``pre_death``: one of the last 64 decisions of an episode that ends in death;
  it wins where it overlaps ``entry``, because deaths are what the stratum
  exists to oversample;
- ``ordinary``: everything else.

A death is a terminal decision whose realized reward is -1: the game's reward
is replaced by -1 on death and is otherwise never negative. The strata come
from the token frames alone, so the index never depends on the capture's
summary record; the split comes from the episode records and the chunk
offsets from the meta lines. A replay shard stores no frames: its writer
derives each episode's ``floor_trace`` from them, the floor-token changes and
the death, and the index is built from those (``trace_index``), equal to the
index of the frames.

The index of one shard is a few spans per episode, small enough that a loader
holds every shard's index at once. It is cached beside the corpus under
``shard_key``, which covers every file of the shard: the frames give the
strata, but the split comes from the episode records, so a shard republished
with only its split changed must not read a stale index. It is built from
whole-episode chunks of the shard, decoded episode by episode, in a process
pool (``map_chunks``).

Four workers is the measured fastest pool. Cold builds of a synthetic
10M-decision shard (2,466 episodes, 40 chunks of 250k decisions) on an
18-core Mac at load average 13-18 took 8.3-9.0 s with 1 worker, 7.7-8.2 s
with 2, 5.8-5.9 s with 4, 7.0-7.4 s with 6, and 10.5-12.9 s with 8.
"""

from collections.abc import Callable, Sequence
from concurrent import futures
from pathlib import Path
from typing import Final, Protocol, cast

import dataclasses
import enum
import functools
import hashlib
import uuid

from torch import Tensor

import torch

from priml.baselines.craftax.game.state import NUM_LEVELS
from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    FloorTrace,
    ManifestLine,
    read_episodes,
    read_summaries,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.lib.codec import from_plain


FLOOR_AUX: Final = craftax_schema().scalar_names.index("floor")
"""Auxiliary index of the floor token."""

FLOORS: Final = NUM_LEVELS
"""Floors 0 overworld through 8 Graveyard."""

EVENT_DECISIONS: Final = 64
"""Decisions after a floor entry, or before a death, that form those events."""


class EpisodeReader(Protocol):
    """Reads chosen whole episodes of a shard, with their frames.

    ``archive.read_episodes`` reads a frame shard; ``snapshots.replay_episodes``,
    its replay build bound, reads either format.
    """

    def __call__(
        self,
        directory: Path,
        line: ManifestLine,
        *,
        summaries: Sequence[EpisodeSummary],
    ) -> list[Episode]:
        """Return the chosen episodes, in the order given."""
        ...


class Event(enum.IntEnum):
    """The event half of a stratum."""

    ENTRY = 0
    PRE_DEATH = 1
    ORDINARY = 2


STRATA: Final = FLOORS * len(Event)
"""Stratum count; stratum ``floor * len(Event) + event``."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ShardIndex:
    """Stratum spans of one shard's episodes.

    Attributes:
      decisions: Decision count of each episode, int64 ``[E]``.
      split: Receipt split of each episode, 0 training and 1 validation,
        int64 ``[E]``.
      span_episode: Episode of each span, int64 ``[S]``.
      span_start: First decision of each span, int64 ``[S]``.
      span_length: Decisions in each span, int64 ``[S]``.
      span_stratum: Stratum of each span, int64 ``[S]``.

    """

    decisions: Tensor
    split: Tensor
    span_episode: Tensor
    span_start: Tensor
    span_length: Tensor
    span_stratum: Tensor

    def counts(self) -> Tensor:
        """Return decisions per stratum, int64 ``[STRATA]``."""
        return torch.zeros(STRATA, dtype=torch.int64).index_add_(
            0,
            self.span_stratum,
            self.span_length,
        )


def decision_strata(*, aux: Tensor, reward: Tensor, done: Tensor) -> Tensor:
    """Return the stratum of every decision of one complete episode.

    Args:
      aux: Auxiliary token values, int16 ``[T, 51]``.
      reward: Realized rewards, int16 ``[T]``.
      done: Terminal flags, bool ``[T]``.

    Returns:
      strata: Stratum of each decision, int64 ``[T]``.

    """
    return _strata(aux[:, FLOOR_AUX].long(), died=bool(done[-1] & (reward[-1] == -1)))


def floor_trace(*, aux: Tensor, reward: Tensor, done: Tensor) -> FloorTrace:
    """Return what the strata of one complete episode depend on, for its summary.

    Args:
      aux: Auxiliary token values, int16 ``[T, 51]``.
      reward: Realized rewards, int16 ``[T]``.
      done: Terminal flags, bool ``[T]``.

    Returns:
      trace: Where the floor token changes, and whether the episode died.

    """
    floor = aux[:, FLOOR_AUX].long()
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64),
            (floor[1:] != floor[:-1]).nonzero()[:, 0] + 1,
        ],
    )
    return FloorTrace(
        changes=tuple(
            zip(
                from_plain(starts.tolist(), list[int]),
                from_plain(floor[starts].tolist(), list[int]),
                strict=True,
            ),
        ),
        died=bool(done[-1] & (reward[-1] == -1)),
    )


def trace_strata(trace: FloorTrace, *, decisions: int) -> Tensor:
    """Return the stratum of every decision of one episode from its floor trace.

    Args:
      trace: The episode's floor trace, from its replay-shard summary.
      decisions: The episode's decision count.

    Returns:
      strata: Stratum of each decision, int64 ``[T]``, equal to
        ``decision_strata`` of its frames.

    """
    starts, floors = torch.tensor(trace.changes, dtype=torch.int64).T
    lengths = torch.diff(starts, append=torch.tensor([decisions]))
    return _strata(torch.repeat_interleave(floors, lengths), died=trace.died)


def build_index(episodes: list[Episode]) -> ShardIndex:
    """Index one shard's episodes.

    Args:
      episodes: The shard's episodes, in written order.

    Returns:
      index: Their stratum spans.

    """
    return _index(
        [decision_strata(aux=e.aux, reward=e.reward, done=e.done) for e in episodes],
        splits=[e.receipt.split for e in episodes],
    )


def trace_index(summaries: list[EpisodeSummary]) -> ShardIndex:
    """Index episodes of a replay shard from their summaries alone.

    Args:
      summaries: The episodes' summaries, each with its floor trace.

    Returns:
      index: Their stratum spans, equal to ``build_index`` of their frames.

    """
    strata: list[Tensor] = []
    for s in summaries:
        if s.floors is None:
            raise ValueError("Only a replay shard's summaries hold floors.")
        strata.append(trace_strata(s.floors, decisions=s.decisions))
    return _index(strata, splits=[s.receipt.split for s in summaries])


def load_index(
    directory: Path,
    line: ManifestLine,
    *,
    index_dir: Path,
    workers: int = 4,
    chunk_decisions: int = 250_000,
) -> ShardIndex:
    """Return a shard's index from the cache, building and caching it if absent.

    Args:
      directory: The shard's worker directory.
      line: The shard's manifest line.
      index_dir: Directory of cached indexes, created if absent.
      workers: Processes that build it; see ``map_chunks``. Four is the measured
        fastest; see the module docstring.
      chunk_decisions: Decisions each process decodes at once.

    Returns:
      index: The shard's stratum spans; a replay shard's come from its
        summaries' floor traces, in this process.

    """
    path = index_dir / f"index-v1-{shard_key(line)}.pt"
    if path.exists():
        return ShardIndex(**load_tensors(path))
    parts = (
        [
            trace_index(chunk)
            for chunk in _chunks(read_summaries(directory, line), chunk_decisions)
        ]
        if line.snapshot_stride
        else map_chunks(
            directory,
            line,
            build_index,
            workers=workers,
            chunk_decisions=chunk_decisions,
        )
    )
    episodes = torch.tensor([len(p.decisions) for p in parts])
    offsets = from_plain((torch.cumsum(episodes, 0) - episodes).tolist(), list[int])
    index = ShardIndex(
        decisions=torch.cat([p.decisions for p in parts]),
        split=torch.cat([p.split for p in parts]),
        span_episode=torch.cat(
            [p.span_episode + o for p, o in zip(parts, offsets, strict=True)],
        ),
        span_start=torch.cat([p.span_start for p in parts]),
        span_length=torch.cat([p.span_length for p in parts]),
        span_stratum=torch.cat([p.span_stratum for p in parts]),
    )
    save_tensors(
        {f.name: getattr(index, f.name) for f in dataclasses.fields(index)},
        path,
    )
    return index


def shard_key(line: ManifestLine) -> str:
    """Return a cache key that changes when any file of the shard changes.

    Args:
      line: The shard's manifest line.

    Returns:
      key: SHA-256 hex digest of the file digests, in suffix order.

    """
    digests = "".join(line.sha256[suffix] for suffix in sorted(line.sha256))
    return hashlib.sha256(digests.encode()).hexdigest()


def map_chunks[T](
    directory: Path,
    line: ManifestLine,
    function: Callable[[list[Episode]], T],
    *,
    workers: int,
    chunk_decisions: int,
    read: EpisodeReader = read_episodes,
) -> list[T]:
    """Apply ``function`` to consecutive whole-episode chunks of a shard, in order.

    Each chunk decodes only its own episodes, so a 10M-decision shard never sits
    decoded in memory at once, and chunks run in a pool of ``workers`` processes
    when there is more than one of each.

    Args:
      directory: The shard's worker directory.
      line: The shard's manifest line.
      function: Picklable function of one chunk's episodes.
      workers: Processes to run chunks in; 1 runs them in this process.
      chunk_decisions: A chunk closes at the first episode boundary at or after
        this many decisions.
      read: Reads a chunk's episodes; picklable when ``workers`` exceeds 1.

    Returns:
      results: ``function`` of each chunk, in shard order.

    """
    chunks = _chunks(read_summaries(directory, line), chunk_decisions)
    apply = functools.partial(_apply, function, read, directory, line)
    workers = min(workers, len(chunks))
    if workers == 1:
        return [apply(chunk) for chunk in chunks]
    # One torch thread per process: the pool is the parallelism, and each
    # process otherwise starts a thread per core.
    with futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=torch.set_num_threads,
        initargs=(1,),
    ) as pool:
        return list(pool.map(apply, chunks))


def save_tensors(tensors: dict[str, Tensor], path: Path) -> None:
    """Write ``tensors`` to ``path`` so concurrent readers never see a partial file.

    Args:
      tensors: Tensors to save with ``torch.save``.
      path: Destination; its directory is created if absent.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not ``mkstemp``: its file is mode 0600, so a cache shared on the datasets
    # filesystem could not be read by the next user. ``x`` takes the umask's mode.
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        torch.save(tensors, handle)
    temporary.replace(path)


def load_tensors(path: Path) -> dict[str, Tensor]:
    """Read a mapping written by ``save_tensors``.

    Args:
      path: File to read.

    Returns:
      tensors: The saved tensors by name; entries that are not tensors are
        dropped, so a caller missing a field fails on construction.

    """
    return {
        key: value
        for key, value in from_plain(
            cast("object", torch.load(path, weights_only=True)),
            dict[str, object],
        ).items()
        if isinstance(value, Tensor)
    }


def _apply[T](
    function: Callable[[list[Episode]], T],
    read: EpisodeReader,
    directory: Path,
    line: ManifestLine,
    chunk: list[EpisodeSummary],
) -> T:
    """Decode one chunk's episodes and apply ``function`` to them."""
    return function(read(directory, line, summaries=chunk))


def _strata(floor: Tensor, *, died: bool) -> Tensor:
    """Return the stratum of every decision from its floor and the episode's end."""
    visits = torch.nn.functional.one_hot(floor, FLOORS).cumsum(0)
    entry = visits.gather(1, floor[:, None]).squeeze(1) <= EVENT_DECISIONS
    decisions = len(floor)
    pre_death = died & (torch.arange(decisions) >= decisions - EVENT_DECISIONS)
    event = torch.where(entry, Event.ENTRY, Event.ORDINARY)
    event = torch.where(pre_death, Event.PRE_DEATH, event)
    return floor * len(Event) + event


def _index(strata: list[Tensor], *, splits: list[int]) -> ShardIndex:
    """Run-length encode consecutive episodes' decision strata."""
    decisions = torch.tensor([len(s) for s in strata], dtype=torch.int64)
    flat = torch.cat(strata)
    episode = torch.repeat_interleave(torch.arange(len(strata)), decisions)
    first = torch.cumsum(decisions, 0) - decisions
    changed = (flat[1:] != flat[:-1]) | (episode[1:] != episode[:-1])
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), changed.nonzero()[:, 0] + 1])
    ends = torch.cat([starts[1:], torch.tensor([len(flat)])])
    return ShardIndex(
        decisions=decisions,
        split=torch.tensor(splits, dtype=torch.int64),
        span_episode=episode[starts],
        span_start=starts - first[episode[starts]],
        span_length=ends - starts,
        span_stratum=flat[starts],
    )


def _chunks(
    summaries: list[EpisodeSummary],
    chunk_decisions: int,
) -> list[list[EpisodeSummary]]:
    """Split summaries into runs closing at the first episode at or after the size."""
    chunks: list[list[EpisodeSummary]] = [[]]
    size = 0
    for summary in summaries:
        if size >= chunk_decisions:
            chunks.append([])
            size = 0
        chunks[-1].append(summary)
        size += summary.decisions
    return chunks
