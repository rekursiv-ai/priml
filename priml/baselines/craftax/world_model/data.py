"""Replay stream: stratified windows of archived episodes, packed for training.

The sampler is a pure function of ``(seed, rank, step, micro_step)``: a Philox
generator keyed by those four integers draws, per window,

1. a stratum with probability proportional to ``n_s ** power``, where ``n_s``
   is the stratum's decision count in the corpus split;
2. an episode uniformly among the episodes with decisions in that stratum;
3. an anchor uniformly among that episode's decisions in the stratum;
4. a start ``0 … W−1`` decisions before the anchor, ``W = t_g // 2``, clipped
   at the episode start.

The window then runs to the end of that episode and continues with the next
episodes of the same shard, each from its ``start``, until ``t_g`` positions or
``s_max`` segments are used; it is padded only when the shard runs out.
Resuming therefore needs only the count of micro-batches served.

Each episode is its own zstd frame, so a window decodes only the episodes it
uses, and decoded episodes are held in an LRU cache bounded by decisions. A
corpus of replay shards (``archive.py``), which store no frames, is served by
``snapshots.ReplayCache`` instead: it replays the blocks of an episode a window
needs from the latest stored snapshot and keeps the most recent blocks. One
worker thread builds micro-batches one ahead of the consumer, into pinned
memory on CUDA, so the host-to-device copy is asynchronous.

The validation set is one of two kinds. ``StratifiedWindows``, the default, is
fixed micro-batches of the same stratified windows, keyed by ``sampler_seed``,
rank, and index, so the scored decisions change with the window length and the
GPU count, and within a stratum an episode's decisions count less the longer it
is. ``EvalSpans`` instead fixes the scored decisions by the corpus and its own
seed alone and weights each by the inverse of its draw probability, so every
ratio over them estimates the natural distribution.

Every batch carries ``weight``, float32 ``[B, t_g]``: what each position's
decision counts in validation metrics, 0 at padding and at unscored history.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, override

import collections
import dataclasses
import queue
import threading

from configgle import Fig
from torch import Tensor
from torch.nn import functional

import numpy as np
import torch

from priml.baselines.craftax.lib.arrays import ints, typed
from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    ManifestLine,
    read_corpus,
    read_summaries,
)
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack,
)
from priml.baselines.craftax.world_model.index import (
    STRATA,
    ShardIndex,
    decision_strata,
    load_index,
)
from priml.baselines.craftax.world_model.snapshots import (
    ReplayCache,
    replay_episodes,
)
from priml.lib.codec import from_plain
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping, Sequence

    from numpy.typing import NDArray


class StratifiedWindows:
    """A fixed validation set of the stratified windows training draws.

    Micro-batch ``index`` is keyed by the sampler seed, the rank, and
    ``index``, never by a step, so every evaluation scores the same windows.
    Which decisions they hold changes with the window length and the GPU count,
    and within a stratum an episode's decisions count less the longer it is.
    Every position counts 1 but padding.

    Args:
      config: How many micro-batches.
      sampler: Anchor sampler over the validation shards.
      cache: Decoder of the same shards, in the same order.
      seed: The stream's sampler seed, the first word of every key.
      rank: This process's rank, the second.
      windows: Windows per micro-batch.
      t_g: Global positions per window.
      s_max: Segments allowed per window.

    """

    class Config(Fig["StratifiedWindows"]):
        """How many micro-batches."""

        batches: int = 8
        """Validation micro-batches per evaluation."""

    def __init__(
        self,
        config: Config,
        *,
        sampler: CorpusSampler,
        cache: FrameSource,
        seed: int,
        rank: int,
        windows: int,
        t_g: int,
        s_max: int,
    ) -> None:
        self.batches = config.batches
        self.sampler, self.cache = sampler, cache
        self.seed, self.rank = seed, rank
        self.windows, self.t_g, self.s_max = windows, t_g, s_max

    def __len__(self) -> int:
        """Return the micro-batches this rank serves."""
        return self.batches

    def micro_batch(self, index: int) -> tuple[PackedBatch, Tensor, Tensor]:
        """Pack this rank's validation micro-batch ``index``.

        Args:
          index: Micro-batch of this rank, below ``len(self)``.

        Returns:
          batch: The packed micro-batch.
          stratum: Decision stratum of every position, -1 at padding.
          weight: What every position's decision counts, 0 at padding.

        """
        batch, stratum = micro_batch(
            self.sampler,
            self.cache,
            key=(self.seed, self.rank, index, 0, 1),
            windows=self.windows,
            t_g=self.t_g,
            s_max=self.s_max,
        )
        return batch, stratum, (stratum >= 0).float()


class ReplayStream:
    """Serve packed micro-batches of a named corpus: ``media``, ``stratum``, ``weight``.

    ``media`` is a ``PackedBatch``; ``stratum`` is int64 ``[B, t_g]``, each
    position's decision stratum, or -1 at padding; ``weight`` is what each
    position's decision counts in validation metrics. Training reads the
    corpus's training shards forever; evaluation reads the fixed micro-batches
    of its ``validation`` set. ``eval_sampler.counts`` holds the validation
    split's natural decisions per stratum.
    """

    class Config(Fig["ReplayStream"]):
        """Where the corpus lives, the window geometry, and the sampler."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/craftax/world-model/archive-v1"
        """Archive root, resolved beneath ``base_dir``."""

        corpus: Path | str = "corpora/base.json"
        """Corpus file, relative to ``working_dir``."""

        index_dir: Path | str = "index"
        """Cache of per-shard stratum indexes, relative to ``working_dir``."""

        windows: int = 1
        """Windows per micro-batch."""

        t_g: int = 8_192
        """Global positions per window."""

        s_max: int = 64
        """Episode segments allowed per window."""

        micro_batches_per_step: int = 1
        """Micro-batches per optimizer step, for the sampler's key."""

        sampler_seed: int = 0
        """First word of every sampler key; fixed so a resume redraws nothing."""

        stratum_power: float = 0.5
        """Exponent of a stratum's decision count in its sampling weight."""

        cached_decisions: int = 1_000_000
        """Decisions of decoded episodes held per split, about 0.9 GB."""

        replay_block_decisions: int = 4_096
        """Decisions per block a replay shard's episode is regenerated in."""

        replay_cached_episodes: int = 1_024
        """Replay shards' episodes whose records, snapshots and strata are held."""

        validation: StratifiedWindows.Config | EvalSpans.Config = field(
            default_factory=StratifiedWindows.Config,
        )
        """The fixed validation set: stratified windows, or natural-weighted spans."""

        train_spans: EvalSpans.Config | None = None
        """Spans drawn the same way from the training split, which
        ``eval_dataloader`` serves inside ``training_split``; scored beside
        validation, a train-validation gap that grows shows overfitting."""

        count_multiple: int = 1
        """Frames and jobs of every micro-batch are padded with inert entries to a
        multiple of this, so compiled shapes and attention plans repeat."""

        device: torch.device | str | None = None
        """Device batches land on; see ``get_device``."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            self.corpus = Path(self.working_dir) / self.corpus
            self.index_dir = Path(self.working_dir) / self.index_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.windows <= 0 or config.s_max <= 0 or config.t_g < 2:
            raise ValueError("windows and s_max must be positive and t_g at least 2.")
        if (
            config.micro_batches_per_step <= 0
            or config.cached_decisions <= 0
            or config.count_multiple <= 0
        ):
            raise ValueError(
                "micro_batches_per_step, cached_decisions, and count_multiple must "
                "be positive.",
            )
        self.config = config
        self.device = get_device(config.device)
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the data; never ticked, because training never runs out."""
        distributed = torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if distributed else 0
        world = torch.distributed.get_world_size() if distributed else 1
        splits: tuple[list[tuple[Path, ManifestLine]], ...] = ([], [])
        indexes: tuple[list[ShardIndex], ...] = ([], [])
        for directory, line in read_corpus(Path(config.corpus)):
            index = load_index(directory, line, index_dir=Path(config.index_dir))
            split = {int(s) for s in index.split}
            if len(split) != 1:
                raise ValueError(f"Shard {directory / line.shard} mixes splits.")
            (side,) = split
            splits[side].append((directory, line))
            indexes[side].append(index)
        if not splits[0] or not splits[1]:
            raise ValueError(f"{config.corpus} needs training and validation shards.")
        power = config.stratum_power
        self.train_sampler = CorpusSampler(indexes[0], power=power)
        self.train_cache = self._frame_source(splits[0])
        self.eval_sampler = CorpusSampler(indexes[1], power=power)
        self.eval_cache = self._frame_source(splits[1])
        geometry = {
            "t_g": config.t_g,
            "s_max": config.s_max,
            "windows": config.windows,
            "rank": self.rank,
            "world": world,
        }
        validation = config.validation
        self.validation: EvalSpans | StratifiedWindows
        if isinstance(validation, EvalSpans.Config):
            self.validation = EvalSpans(
                validation,
                indexes=indexes[1],
                cache=self.eval_cache,
                **geometry,
            )
        else:
            assert isinstance(validation, StratifiedWindows.Config)
            self.validation = StratifiedWindows(
                validation,
                sampler=self.eval_sampler,
                cache=self.eval_cache,
                seed=config.sampler_seed,
                rank=self.rank,
                windows=config.windows,
                t_g=config.t_g,
                s_max=config.s_max,
            )
        # Its own cache: sharing the stream's would evict the episodes the
        # prefetch thread is packing.
        self.train_spans = (
            EvalSpans(
                config.train_spans,
                indexes=indexes[0],
                cache=self._frame_source(splits[0]),
                **geometry,
            )
            if config.train_spans is not None
            else None
        )
        self._scoring_training = False
        self._served = 0
        self._live: _Stream | None = None

    def train_dataloader(self) -> Iterator[dict[str, PackedBatch | Tensor]]:
        """Return the endless training stream, resumed at the served count."""
        self._live = _Stream(self._train_batch, first=self._served, device=self.device)
        return iter(self._live)

    def eval_dataloader(self) -> Iterator[dict[str, PackedBatch | Tensor]]:
        """Return the fixed validation micro-batches, the same on every call.

        Returns:
          batches: This rank's micro-batches of ``validation``; inside
            ``training_split``, of ``train_spans``.

        """
        if self._scoring_training:
            if self.train_spans is None:
                raise ValueError("Expected self.train_spans is not None.")
            count, produce = len(self.train_spans), self._train_span_batch
        else:
            count, produce = len(self.validation), self._eval_batch
        return iter(_Stream(produce, first=0, count=count, device=self.device))

    @contextmanager
    def training_split(self) -> Generator[None]:
        """Serve ``train_spans`` from ``eval_dataloader`` while inside.

        Yields:
          None: Control, with ``eval_dataloader`` serving ``train_spans``.

        Raises:
          ValueError: No ``train_spans`` are configured.

        """
        if self.train_spans is None:
            raise ValueError("Scoring the training split needs train_spans.")
        self._scoring_training = True
        try:
            yield
        finally:
            self._scoring_training = False

    def state_dict(self) -> dict[str, object]:
        """Return the micro-batches served and the epoch timer."""
        served = self._live.served if self._live is not None else self._served
        return {"batches": served, "timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Resume at a checkpoint's served count.

        Args:
          state_dict: Checkpoint written by ``state_dict``.

        """
        self.timer_epoch.load_state_dict(
            from_plain(state_dict["timer_epoch"], dict[str, object]),
        )
        self._served = from_plain(state_dict["batches"], int)
        self._live = None

    def _frame_source(self, entries: list[tuple[Path, ManifestLine]]) -> FrameSource:
        """Return the frames of one split's shards: decoded, or replayed."""
        replay = {bool(line.snapshot_stride) for _, line in entries}
        if len(replay) > 1:
            raise ValueError(
                f"{self.config.corpus} mixes frame and replay shards; a split's "
                "shards must share one format.",
            )
        if replay == {True}:
            return ReplayCache(
                entries,
                capacity=self.config.cached_decisions,
                block_decisions=self.config.replay_block_decisions,
                cached_episodes=self.config.replay_cached_episodes,
            )
        return EpisodeCache(entries, capacity=self.config.cached_decisions)

    def _train_batch(self, index: int) -> _Item:
        """Build training micro-batch ``index``."""
        step, micro_step = divmod(index, self.config.micro_batches_per_step)
        key = (self.config.sampler_seed, self.rank, step, micro_step)
        return self._batch(self.train_sampler, self.train_cache, key)

    def _eval_batch(self, index: int) -> _Item:
        """Build micro-batch ``index`` of the validation set."""
        batch, stratum, weight = self.validation.micro_batch(index)
        return _pad_counts(batch, self.config.count_multiple), stratum, weight

    def _train_span_batch(self, index: int) -> _Item:
        """Build micro-batch ``index`` of the training split's fixed spans."""
        if self.train_spans is None:
            raise ValueError("Expected self.train_spans is not None.")
        batch, stratum, weight = self.train_spans.micro_batch(index)
        return _pad_counts(batch, self.config.count_multiple), stratum, weight

    def _batch(
        self,
        sampler: CorpusSampler,
        cache: FrameSource,
        key: Sequence[int],
    ) -> _Item:
        """Build one stratified micro-batch with this stream's geometry."""
        batch, stratum = micro_batch(
            sampler,
            cache,
            key=key,
            windows=self.config.windows,
            t_g=self.config.t_g,
            s_max=self.config.s_max,
        )
        batch = _pad_counts(batch, self.config.count_multiple)
        return batch, stratum, (stratum >= 0).float()


class FrameSource(Protocol):
    """Where windows are cut from: one split's shards, episode by episode."""

    def episodes(self, shard: int) -> int:
        """Return the episode count of one shard."""
        ...

    def decisions(self, shard: int, episode: int) -> int:
        """Return the decision count of one episode."""
        ...

    def starts_at_reset(self, shard: int, episode: int) -> bool:
        """Return whether an episode starts at its world's reset, not a branch's origin.

        Args:
          shard: Position of the shard in the split.
          episode: Episode within the shard.

        Returns:
          at_reset: Whether its segment from decision 0 begins with a ``start``
            position.

        """
        ...

    def segment(
        self,
        shard: int,
        episode: int,
        *,
        start: int,
        stop: int,
    ) -> tuple[Segment, Tensor]:
        """Return decisions ``[start, stop)`` of one episode.

        Args:
          shard: Position of the shard in the split.
          episode: Episode within the shard.
          start: First decision.
          stop: One past the last decision, at most the episode's length; may
            equal ``start``.

        Returns:
          segment: The decisions, with the frames of ``[start, stop]`` that the
            episode has, starting the episode when ``start`` is 0.
          strata: Stratum of each of those frames' decisions.

        """
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DecodedEpisode:
    """A decoded episode and the stratum of each of its decisions.

    Attributes:
      episode: The episode; its summary is dropped, since packing never reads it.
      strata: Stratum of each decision, int64 ``[T]``.

    """

    episode: Episode
    strata: Tensor


class EpisodeCache:
    """Decode single whole episodes on demand, keeping the most recent; thread-safe.

    Each shard's summaries, which locate its episodes' bytes, are read on first
    use and kept without their capture records. A replay shard's episode is
    replayed whole (``snapshots.replay_episodes``); ``ReplayCache`` serves
    windows of long episodes more cheaply.

    Args:
      entries: Each shard's directory and manifest line.
      capacity: Decisions of decoded episodes to hold; the most recent episode
        is kept even when it alone exceeds this.

    """

    def __init__(
        self,
        entries: Sequence[tuple[Path, ManifestLine]],
        *,
        capacity: int,
    ) -> None:
        self.entries = list(entries)
        self.capacity = capacity
        self._episodes: collections.OrderedDict[tuple[int, int], DecodedEpisode] = (
            collections.OrderedDict()
        )
        self._decisions = 0
        self._summaries: dict[int, list[EpisodeSummary]] = {}
        self._lock = threading.Lock()

    @property
    def resident(self) -> list[tuple[int, int]]:
        """Return the decoded ``(shard, episode)`` pairs, least recent first."""
        return list(self._episodes)

    def episodes(self, shard: int) -> int:
        """Return the episode count of shard ``shard`` of ``entries``."""
        with self._lock:
            return len(self._shard_summaries(shard))

    def decisions(self, shard: int, episode: int) -> int:
        """Return the decision count of one episode."""
        with self._lock:
            return self._shard_summaries(shard)[episode].decisions

    def starts_at_reset(self, shard: int, episode: int) -> bool:
        """Return whether an episode starts at its world's reset; see ``FrameSource``."""
        return not self.get(shard, episode).episode.origin

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
          shard: Position of the shard in the split.
          episode: Episode within the shard.
          start: First decision.
          stop: One past the last decision; may equal ``start``.

        Returns:
          segment: The decisions and the frames of ``[start, stop]`` that exist.
          strata: Stratum of each of those frames' decisions.

        """
        decoded = self.get(shard, episode)
        e = decoded.episode
        frames = slice(start, min(stop + 1, len(e.actions)))
        segment = Segment(
            cells=e.cells[frames],
            aux=e.aux[frames],
            actions=e.actions[start:stop],
            reward=e.reward[start:stop],
            done=e.done[start:stop],
            starts_episode=start == 0 and not e.origin,
        )
        return segment, decoded.strata[frames]

    def get(self, shard: int, episode: int) -> DecodedEpisode:
        """Return one episode, decoding only its own bytes if absent.

        Args:
          shard: Position of the shard in ``entries``.
          episode: Episode within the shard.

        Returns:
          decoded: The episode and its decision strata.

        """
        key = (shard, episode)
        with self._lock:
            if key in self._episodes:
                self._episodes.move_to_end(key)
                return self._episodes[key]
            summary = self._shard_summaries(shard)[episode]
            (e,) = replay_episodes(*self.entries[shard], summaries=[summary])
            decoded = DecodedEpisode(
                episode=e,
                strata=decision_strata(aux=e.aux, reward=e.reward, done=e.done),
            )
            self._episodes[key] = decoded
            self._decisions += len(e.actions)
            while self._decisions > self.capacity and len(self._episodes) > 1:
                _, evicted = self._episodes.popitem(last=False)
                self._decisions -= len(evicted.episode.actions)
            return decoded

    def _shard_summaries(self, shard: int) -> list[EpisodeSummary]:
        """Return one shard's summaries, reading them on first use; hold the lock."""
        if shard not in self._summaries:
            self._summaries[shard] = [
                dataclasses.replace(s, summary={})
                for s in read_summaries(*self.entries[shard])
            ]
        return self._summaries[shard]


class CorpusSampler:
    """Draw window anchors: stratum by weight, episode uniformly, then decision.

    Attributes:
      counts: Natural decisions per stratum in these shards, int64 ``[STRATA]``;
        normalized, the weights that reweight per-stratum metrics to the natural
        distribution.
      probabilities: Probability of each stratum, float64 ``[STRATA]``.

    """

    def __init__(self, indexes: Sequence[ShardIndex], *, power: float) -> None:
        shard = np.concatenate(
            [
                np.full(len(ix.span_start), i, dtype=np.int64)
                for i, ix in enumerate(indexes)
            ],
        )
        episode = typed(
            torch.cat([ix.span_episode for ix in indexes]).numpy(),
            np.int64,
        )
        start = typed(torch.cat([ix.span_start for ix in indexes]).numpy(), np.int64)
        length = typed(torch.cat([ix.span_length for ix in indexes]).numpy(), np.int64)
        stratum = typed(
            torch.cat([ix.span_stratum for ix in indexes]).numpy(),
            np.int64,
        )
        most = max(len(ix.decisions) for ix in indexes)
        key = (stratum * len(indexes) + shard) * most + episode
        order = np.argsort(key, kind="stable")
        self._start = start[order]
        self._length = length[order]
        self._end = np.cumsum(self._length)
        first = np.flatnonzero(
            np.concatenate(
                [np.ones(1, np.bool), np.not_equal(key[order][1:], key[order][:-1])],
            ),
        )
        self._group_shard = shard[order][first]
        self._group_episode = episode[order][first]
        self._group_offset = self._end[first] - self._length[first]
        self._group_size = (
            np.concatenate([self._group_offset[1:], self._end[-1:]])
            - self._group_offset
        )
        group_stratum = stratum[order][first]
        self._stratum_first = np.searchsorted(group_stratum, np.arange(STRATA))
        self._stratum_groups = np.bincount(group_stratum, minlength=STRATA)
        self.counts = torch.stack([ix.counts() for ix in indexes]).sum(0)
        counts = self.counts.numpy()
        # Masked, not multiplied: under a negative power an empty stratum's
        # ``0 ** power`` is inf, and ``inf * 0`` would make every probability NaN.
        weights = np.zeros(len(counts))
        np.power(counts, power, out=weights, where=counts > 0)
        self.probabilities = weights / np.sum(weights)

    def anchor(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """Draw one anchor.

        Args:
          rng: Generator to draw from.

        Returns:
          shard: Position of the shard in ``indexes``.
          episode: Episode within the shard.
          decision: Anchor decision within the episode.

        """
        stratum = int(rng.choice(STRATA, p=self.probabilities))
        groups = self._stratum_groups.item(stratum)
        group = self._stratum_first.item(stratum) + int(rng.integers(groups))
        size = self._group_size.item(group)
        position = int(self._group_offset.item(group)) + int(rng.integers(size))
        span = int(np.searchsorted(self._end, position, side="right"))
        offset = position - int(self._end.item(span) - self._length.item(span))
        return (
            int(self._group_shard.item(group)),
            int(self._group_episode.item(group)),
            int(self._start.item(span)) + offset,
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Span:
    """One drawn validation span.

    Attributes:
      shard: Position of the shard in the validation indexes.
      episode: Episode within the shard.
      first: First target decision, a multiple of ``span_decisions``.
      length: Target decisions, ``span_decisions`` unless the episode ends first.
      weight: What each target counts: the inverse of the span's draw
        probability, over the number of draws.

    """

    shard: int
    episode: int
    first: int
    length: int
    weight: float


class EvalSpans:
    """A fixed validation set of aligned episode spans, weighted to the natural mix.

    Each validation episode is cut into consecutive spans of ``span_decisions``
    decisions, the last one shorter, so every decision's record lies in exactly
    one span, and the episode's first frame in its first. ``spans`` of them are
    drawn once, with replacement, from ``seed`` alone: each with probability
    proportional to the sum over its decisions of ``n_s ** (stratum_power - 1)``,
    ``n_s`` the count of the decision's stratum, so strata are drawn about as
    the training sampler draws them. Every target of a drawn span is weighted
    by the inverse of that probability (Hansen-Hurwitz), so weighted sums
    estimate natural totals, per stratum and overall, however long the episode.

    A span is scored after as much of its episode as precedes it and fits in a
    window; that history is unscored. Spans are packed into windows, windows
    into micro-batches, and micro-batch ``m`` goes to rank ``m % world``; a rank
    left short repeats one at weight 0, so every rank serves as many. The
    targets thus depend only on the validation split and the config: ``t_g``
    sets how much history precedes them, and ``t_g``, ``windows``, and the rank
    count only how they are packed.

    Args:
      config: The draw.
      indexes: Stratum indexes of the validation shards, as the cache orders them.
      cache: Decoder of the same shards, which the micro-batches are cut from.
      t_g: Global positions per window.
      s_max: Segments allowed per window.
      windows: Windows per micro-batch.
      rank: This process's rank.
      world: Ranks sharing the micro-batches.

    Raises:
      ValueError: No span is drawn, or a whole span, with its episode's start,
        would not fit in one window.

    """

    class Config(Fig["EvalSpans"]):
        """How many spans, how long, and the seed that draws them."""

        seed: int = 0
        """Seeds the draw; no other seed in the run reaches it."""

        spans: int = 64
        """Spans drawn, with replacement."""

        span_decisions: int = 2_048
        """Target decisions per span; at most ``(t_g - 1) // 2``."""

        stratum_power: float = 0.5
        """Exponent of a stratum's decision count in its drawing weight."""

    def __init__(
        self,
        config: Config,
        *,
        indexes: Sequence[ShardIndex],
        cache: FrameSource,
        t_g: int,
        s_max: int,
        windows: int,
        rank: int = 0,
        world: int = 1,
    ) -> None:
        if config.spans <= 0:
            raise ValueError(f"spans={config.spans} must be positive.")
        if config.span_decisions <= 0 or config.span_decisions > (t_g - 1) // 2:
            raise ValueError(
                f"span_decisions={config.span_decisions} must be positive and fit "
                f"a window of t_g={t_g} positions with its episode's start.",
            )
        self.cache = cache
        self.t_g, self.s_max, self.rank, self.world = t_g, s_max, rank, world
        self.drawn = _draw_spans(config, indexes=indexes)
        """The drawn spans, in shard, episode, and decision order."""
        placed = [
            _place(
                span,
                t_g=t_g,
                head=int(cache.starts_at_reset(span.shard, span.episode)),
            )
            for span in self.drawn
        ]
        rows = _next_fit(placed, t_g=t_g, s_max=s_max)
        # Empty windows fill the last micro-batch, so every one has the same shapes.
        rows += [[] for _ in range(-len(rows) % windows)]
        self._batches = [rows[i : i + windows] for i in range(0, len(rows), windows)]

    def __len__(self) -> int:
        """Return the micro-batches this rank serves."""
        return (len(self._batches) + self.world - 1) // self.world

    def micro_batch(self, index: int) -> tuple[PackedBatch, Tensor, Tensor]:
        """Pack this rank's validation micro-batch ``index``.

        Args:
          index: Micro-batch of this rank, below ``len(self)``.

        Returns:
          batch: The packed micro-batch.
          stratum: Decision stratum of every position, -1 at padding.
          weight: What every position's decision counts, 0 at padding, history,
            and a repeated micro-batch.

        """
        number = index * self.world + self.rank
        rows = self._batches[number % len(self._batches)]
        parts = [
            [
                self.cache.segment(p.shard, p.episode, start=p.start, stop=p.end)
                for p in row
            ]
            for row in rows
        ]
        batch, stratum = pack(parts, t_g=self.t_g, s_max=self.s_max)
        weight = _span_weights(batch, rows, s_max=self.s_max)
        if number >= len(self._batches):
            weight.zero_()
        return batch, stratum, weight


def micro_batch(
    sampler: CorpusSampler,
    cache: FrameSource,
    *,
    key: Sequence[int],
    windows: int,
    t_g: int,
    s_max: int,
) -> tuple[PackedBatch, Tensor]:
    """Draw and pack one micro-batch; the same key always gives the same batch.

    Args:
      sampler: Anchor sampler over the split's shards.
      cache: Decoder of the same shards, in the same order.
      key: Non-negative integers keying the Philox generator.
      windows: Windows in the micro-batch.
      t_g: Global positions per window.
      s_max: Segments allowed per window.

    Returns:
      batch: The packed micro-batch.
      stratum: Decision stratum of every position, -1 at padding.

    """
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(list(key))))
    parts: list[list[tuple[Segment, Tensor]]] = []
    for _ in range(windows):
        shard, episode, anchor = sampler.anchor(rng)
        start = max(0, anchor - int(rng.integers(t_g // 2)))
        parts.append(
            window(
                cache,
                shard=shard,
                episode=episode,
                start=start,
                t_g=t_g,
                s_max=s_max,
            ),
        )
    return pack(parts, t_g=t_g, s_max=s_max)


def window(
    cache: FrameSource,
    *,
    shard: int,
    episode: int,
    start: int,
    t_g: int,
    s_max: int,
) -> list[tuple[Segment, Tensor]]:
    """Cut one window's segments from ``start`` of ``episode`` onward.

    Args:
      cache: Decoder of the split's shards.
      shard: Position of the shard in ``cache.entries``.
      episode: First episode of the window.
      start: First decision of that episode.
      t_g: Global positions per window.
      s_max: Segments allowed per window.

    Returns:
      parts: Each segment with its decisions' strata, sliced to what fits.

    """
    parts: list[tuple[Segment, Tensor]] = []
    cursor = 0
    episodes = cache.episodes(shard)
    while cursor < t_g and len(parts) < s_max and episode < episodes:
        head = int(start == 0 and cache.starts_at_reset(shard, episode))
        room = t_g - cursor
        stop = start + min(
            cache.decisions(shard, episode) - start,
            (room - head + 1) // 2,
        )
        parts.append(cache.segment(shard, episode, start=start, stop=stop))
        cursor += min(room, head + 2 * (stop - start))
        episode += 1
        start = 0
    return parts


type _Item = tuple[PackedBatch, Tensor, Tensor]
"""A micro-batch, its position strata, and its position weights."""


class _Stream:
    """Iterate micro-batches built one ahead on a worker thread."""

    def __init__(
        self,
        produce: Callable[[int], _Item],
        *,
        first: int,
        device: torch.device,
        count: int | None = None,
    ) -> None:
        self.produce = produce
        self.served = first
        # CUDA's current device is per thread, and the prefetch thread's starts at
        # GPU 0: pinning there opened a context on GPU 0 for every rank of a node,
        # 522 MiB each. Naming the consumer's GPU lets the thread select it.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.count = count

    def __iter__(self) -> Iterator[dict[str, PackedBatch | Tensor]]:
        ready: queue.Queue[_Item | BaseException | None] = queue.Queue(maxsize=1)
        done = threading.Event()
        worker = threading.Thread(
            target=self._fill,
            args=(ready, done, self.served),
            name="craftax-replay-stream",
            daemon=True,
        )
        worker.start()
        try:
            while (item := ready.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                batch, stratum, weight = item
                self.served += 1
                yield {
                    "media": batch.to(self.device),
                    "stratum": stratum.to(self.device, non_blocking=True),
                    "weight": weight.to(self.device, non_blocking=True),
                }
        finally:
            done.set()

    def _fill(
        self,
        ready: queue.Queue[_Item | BaseException | None],
        done: threading.Event,
        index: int,
    ) -> None:
        """Produce batches from ``index`` until the count or the consumer stops."""
        pinned = self.device.type == "cuda"
        try:
            if pinned:
                torch.cuda.set_device(self.device)
            while self.count is None or index < self.count:
                batch, stratum, weight = self.produce(index)
                if pinned:
                    batch = PackedBatch(
                        **{
                            field.name: cast(
                                "torch.Tensor",
                                getattr(batch, field.name),
                            ).pin_memory()
                            for field in dataclasses.fields(batch)
                        },
                    )
                    stratum, weight = stratum.pin_memory(), weight.pin_memory()
                if not _offer(ready, done, (batch, stratum, weight)):
                    return
                index += 1
        except BaseException as error:  # noqa: BLE001 -- The consumer re-raises it.
            _offer(ready, done, error)
            return
        _offer(ready, done, None)


def _offer(
    ready: queue.Queue[_Item | BaseException | None],
    done: threading.Event,
    item: _Item | BaseException | None,
) -> bool:
    """Put ``item`` once there is room; give up when the consumer has stopped."""
    while not done.is_set():
        try:
            ready.put(item, timeout=0.05)
        except queue.Full:
            continue
        return True
    return False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Placed:
    """A drawn span cut from its history's first decision, ready to pack.

    Attributes:
      shard: Position of the shard in the cache.
      episode: Episode within the shard.
      start: First decision of the segment, where its history begins.
      end: One past the span's last decision.
      positions: Global positions of the segment.
      threshold: First segment position whose job is scored.
      weight: What each scored position counts.

    """

    shard: int
    episode: int
    start: int
    end: int
    positions: int
    threshold: int
    weight: float


def _pad_counts(batch: PackedBatch, multiple: int) -> PackedBatch:
    """Pad frames and jobs to ``multiple`` with zero frames and unscored start jobs."""
    frames, jobs = -len(batch.aux) % multiple, -len(batch.job_at) % multiple
    if not frames and not jobs:
        return batch
    pad = functional.pad
    return dataclasses.replace(
        batch,
        cells=pad(batch.cells, (0, 0, 0, 0, 0, frames)),
        aux=pad(batch.aux, (0, 0, 0, frames)),
        job_at=pad(batch.job_at, (0, jobs)),
        job_memory=pad(batch.job_memory, (0, jobs), value=-1),
        job_next=pad(batch.job_next, (0, jobs), value=-1),
        job_reward=pad(batch.job_reward, (0, jobs)),
        job_done=pad(batch.job_done, (0, jobs)),
        job_is_start=pad(batch.job_is_start, (0, jobs), value=True),
    )


def _draw_spans(
    config: EvalSpans.Config,
    *,
    indexes: Sequence[ShardIndex],
) -> list[Span]:
    """Cut every episode into aligned spans and draw ``config.spans`` of them."""
    counts = torch.stack([ix.counts() for ix in indexes]).sum(0).double().numpy()
    stratum_weight = np.power(
        counts,
        config.stratum_power - 1,
        out=np.zeros_like(counts),
        where=counts > 0,
    )
    tables = [
        _spans_of(ix, size=config.span_decisions, stratum_weight=stratum_weight)
        for ix in indexes
    ]
    shard = np.concatenate(
        [np.full(len(t[0]), i, dtype=np.int64) for i, t in enumerate(tables)],
    )
    episode = np.concatenate([t[0] for t in tables])
    first = np.concatenate([t[1] for t in tables])
    length = np.concatenate([t[2] for t in tables])
    mass = np.concatenate([t[3] for t in tables])
    probability = mass / np.sum(mass)
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([config.seed])))
    picks = np.sort(rng.choice(len(probability), size=config.spans, p=probability))
    return [
        Span(
            shard=shard.item(i),
            episode=episode.item(i),
            first=first.item(i),
            length=length.item(i),
            weight=float(1 / (config.spans * probability.item(i))),
        )
        for i in ints(picks)
    ]


def _spans_of(
    index: ShardIndex,
    *,
    size: int,
    stratum_weight: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Return each aligned span's episode, first decision, length, and drawing mass."""
    lengths = typed(index.decisions.numpy(), np.int64)
    per_episode = -(-lengths // size)
    episode = np.repeat(np.arange(len(lengths)), per_episode)
    rank = np.arange(len(episode)) - np.repeat(
        np.cumsum(per_episode) - per_episode,
        per_episode,
    )
    first = rank * size
    length = np.minimum(size, lengths[episode] - first)
    # Stratum runs tile the shard's decisions in episode order, so the drawing
    # mass before any decision is piecewise linear in it; a span's mass is the
    # difference at its ends, with no per-decision array.
    order = np.lexsort((index.span_start.numpy(), index.span_episode.numpy()))
    run_length = index.span_length.numpy()[order]
    run_weight = stratum_weight[index.span_stratum.numpy()[order]]
    if run_length.sum() != lengths.sum():
        raise ValueError("Stratum runs must tile episodes.")
    run_first = np.cumsum(run_length) - run_length
    before = np.concatenate([np.zeros(1), np.cumsum(run_length * run_weight)])
    start = (np.cumsum(lengths) - lengths)[episode] + first

    def mass_before(decision: NDArray[np.int64]) -> NDArray[np.float64]:
        run = np.searchsorted(run_first, decision, side="right") - 1
        return before[run] + (decision - run_first[run]) * run_weight[run]

    return episode, first, length, mass_before(start + length) - mass_before(start)


# ``head`` is 1 when the episode starts at its world's reset, so a segment from its
# decision 0 opens with a ``start`` position, and 0 for a branch.
def _place(span: Span, *, t_g: int, head: int) -> _Placed:
    """Start a span's segment as early in its episode as one window allows."""
    end = span.first + span.length
    start = 0 if head + 2 * end <= t_g else max(1, end - t_g // 2)
    head = head if start == 0 else 0
    return _Placed(
        shard=span.shard,
        episode=span.episode,
        start=start,
        end=end,
        positions=head + 2 * (end - start),
        threshold=0 if start == span.first else head + 2 * (span.first - start),
        weight=span.weight,
    )


def _next_fit(
    placed: Sequence[_Placed],
    *,
    t_g: int,
    s_max: int,
) -> list[list[_Placed]]:
    """Fill windows in order, opening one when a segment does not fit."""
    rows: list[list[_Placed]] = []
    room = 0
    for segment in placed:
        if not rows or segment.positions > room or len(rows[-1]) == s_max:
            rows.append([])
            room = t_g
        rows[-1].append(segment)
        room -= segment.positions
    return rows


def _span_weights(
    batch: PackedBatch,
    rows: Sequence[Sequence[_Placed]],
    *,
    s_max: int,
) -> Tensor:
    """Return each position's weight: its span's from the threshold on, else 0."""
    threshold = torch.full((len(rows), s_max + 1), batch.kind.shape[-1])
    value = torch.zeros(len(rows), s_max + 1)
    for row, placed in enumerate(rows):
        threshold[row, : len(placed)] = torch.tensor([p.threshold for p in placed])
        value[row, : len(placed)] = torch.tensor([p.weight for p in placed])
    segment = batch.segment.long()
    scored = batch.pos >= threshold.gather(1, segment)
    return torch.where(scored, value.gather(1, segment), 0.0)
