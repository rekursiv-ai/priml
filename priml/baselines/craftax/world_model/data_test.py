"""Check the stratified sampler, window building, packing, and the replay stream."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast, override

import collections
import dataclasses
import itertools
import queue
import threading

import numpy as np
import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    Receipt,
    read_summaries,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.batch import (
    Kind,
    PackedBatch,
    Segment,
    pack,
)
from priml.baselines.craftax.world_model.data import (
    CorpusSampler,
    EpisodeCache,
    EvalSpans,
    ReplayStream,
    StratifiedWindows,
    _Item,
    _Stream,
    micro_batch,
    window,
)
from priml.baselines.craftax.world_model.index import (
    FLOOR_AUX,
    STRATA,
    Event,
    ShardIndex,
    build_index,
    decision_strata,
)


def _episode(
    decisions: int,
    *,
    tag: int = 0,
    floor: int = 0,
    split: int = 0,
) -> Episode:
    cells = torch.zeros(decisions, 99, 8, dtype=torch.uint8)
    cells[:, 0, 0] = tag
    cells[:, 1, 0] = torch.arange(decisions, dtype=torch.uint8)
    aux = torch.zeros(decisions, 51, dtype=torch.int16)
    aux[:, FLOOR_AUX] = floor
    return Episode(
        receipt=Receipt(
            world_seed=tag,
            sampling_seed=0,
            initial_state_hash=0,
            arm=0,
            split=split,
        ),
        actions=torch.arange(decisions, dtype=torch.uint8) % 43,
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=cells,
        aux=aux,
        reward=torch.ones(decisions, dtype=torch.int16),
        done=torch.arange(decisions) == decisions - 1,
        summary={},
    )


def _cache(
    directory: Path,
    episodes: list[Episode],
    *,
    capacity: int = 10_000,
) -> EpisodeCache:
    directory.mkdir(parents=True, exist_ok=True)
    line = write_shard(directory, index=0, episodes=episodes, provenance={})
    return EpisodeCache([(directory, line)], capacity=capacity)


def _segment(decisions: int, *, frames: int, starts: bool, done: bool) -> Segment:
    generator = torch.Generator().manual_seed(decisions * 7 + frames)
    return Segment(
        cells=torch.randint(0, 9, (frames, 99, 8), generator=generator).byte(),
        aux=torch.randint(0, 9, (frames, 51), generator=generator).short(),
        actions=torch.randint(0, 43, (decisions,), generator=generator).byte(),
        reward=torch.randint(-1, 3, (decisions,), generator=generator).short(),
        done=torch.arange(decisions) == decisions - 1
        if done
        else torch.zeros(decisions, dtype=torch.bool),
        starts_episode=starts,
    )


def _assert_same_batch(left: PackedBatch, right: PackedBatch) -> None:
    for field in dataclasses.fields(PackedBatch):
        a = cast("torch.Tensor", getattr(left, field.name))
        b = cast("torch.Tensor", getattr(right, field.name))
        assert a.dtype == b.dtype, field.name
        assert torch.equal(a, b), field.name


def test_pack_marks_each_position_with_its_decision_stratum() -> None:
    first = _segment(2, frames=2, starts=False, done=True)
    second = _segment(3, frames=4, starts=True, done=False)
    parts = [[(first, torch.tensor([4, 5])), (second, torch.tensor([7, 8, 9]))]]
    _, stratum = pack(parts, t_g=10, s_max=4)
    assert stratum[0].tolist() == [4, 4, 5, 5, 7, 7, 7, 8, 8, 9]
    _, padded = pack([[(first, torch.tensor([4, 5]))]], t_g=6, s_max=4)
    assert padded[0].tolist() == [4, 4, 5, 5, -1, -1]


def test_stratum_probabilities_follow_square_root_counts() -> None:
    index = build_index([_episode(100, floor=2)])
    sampler = CorpusSampler([index], power=0.5)
    entry = 2 * len(Event) + Event.ENTRY
    ordinary = 2 * len(Event) + Event.ORDINARY
    probabilities = sampler.probabilities
    assert probabilities.shape == (STRATA,)
    assert probabilities[entry] == pytest.approx(8 / (8 + 6))
    assert probabilities[ordinary] == pytest.approx(6 / (8 + 6))


def test_a_negative_stratum_power_draws_only_present_strata() -> None:
    # Rarer strata weigh more: the 64 entry decisions against the 36 ordinary.
    sampler = CorpusSampler([build_index([_episode(100, floor=2)])], power=-0.5)
    entry = 2 * len(Event) + Event.ENTRY
    ordinary = 2 * len(Event) + Event.ORDINARY
    probabilities = sampler.probabilities
    assert probabilities[entry] == pytest.approx(6 / (8 + 6))
    assert probabilities[ordinary] == pytest.approx(8 / (8 + 6))
    assert probabilities.sum() == pytest.approx(1)
    assert sampler.anchor(np.random.default_rng(0))[2] < 100


def test_episodes_are_drawn_uniformly_within_a_stratum() -> None:
    long, short = _episode(1_000), _episode(70)
    index = build_index([long, short])
    sampler = CorpusSampler([build_index([_episode(64)]), index], power=0.5)
    rng = np.random.default_rng(0)
    anchors = [sampler.anchor(rng) for _ in range(3_000)]
    ordinary = [(s, e) for s, e, d in anchors if s == 1 and d >= 64]
    short_share = sum(e == 1 for _, e in ordinary) / len(ordinary)
    assert short_share == pytest.approx(0.5, abs=0.05)
    assert all(d < (1_000, 70)[e] for s, e, d in anchors if s == 1)


class _FixedAnchor(CorpusSampler):
    """Always anchor at one decision of episode 0 of shard 0."""

    def __init__(self, index: ShardIndex, *, decision: int) -> None:
        super().__init__([index], power=0.5)
        self.decision = decision

    @override
    def anchor(self, rng: np.random.Generator) -> tuple[int, int, int]:
        del rng
        return 0, 0, self.decision


def _window_starts(directory: Path, *, anchor: int) -> list[int]:
    """Return the first decision of each of 400 windows drawn around ``anchor``."""
    episode = _episode(200)
    cache = _cache(directory, [episode])
    sampler = _FixedAnchor(build_index([episode]), decision=anchor)
    batch, _ = micro_batch(sampler, cache, key=(0,), windows=400, t_g=16, s_max=4)
    first = [row[row >= 0][0] for row in batch.frame_of]
    return [int(batch.cells[frame, 1, 0]) for frame in first]


def test_window_starts_uniformly_within_w_decisions_before_the_anchor(
    tmp_path: Path,
) -> None:
    offsets = collections.Counter(100 - s for s in _window_starts(tmp_path, anchor=100))
    # W = t_g // 2 = 8: offsets 0 ... 7, each about 400 / 8 = 50 times.
    assert sorted(offsets) == list(range(8))
    assert all(25 <= count <= 75 for count in offsets.values()), offsets


def test_window_start_is_clipped_at_the_episode_start(tmp_path: Path) -> None:
    starts = _window_starts(tmp_path, anchor=2)
    assert set(starts) == {0, 1, 2}
    # Offsets 2 ... 7 of the 8 all clip to decision 0.
    assert starts.count(0) / len(starts) == pytest.approx(6 / 8, abs=0.08)


def test_samplers_expose_the_natural_stratum_counts(tmp_path: Path) -> None:
    stream = _stream(_corpus(tmp_path))
    train, val = stream.train_sampler.counts, stream.eval_sampler.counts
    assert train.dtype == val.dtype == torch.int64
    assert train.shape == val.shape == (STRATA,)
    entry = int(Event.ENTRY)
    expected = {entry: 40, len(Event) + entry: 9, 2 * len(Event) + entry: 30}
    assert {s: int(c) for s, c in enumerate(train) if c} == expected
    assert {s: int(c) for s, c in enumerate(val) if c} == {entry: 20}


def test_episode_cache_evicts_least_recently_used_beyond_its_decisions(
    tmp_path: Path,
) -> None:
    episodes = [_episode(40, tag=1), _episode(9, tag=2), _episode(30, tag=3)]
    cache = _cache(tmp_path, episodes, capacity=40)
    assert cache.episodes(0) == 3
    decoded = cache.get(0, 0)
    assert torch.equal(decoded.episode.cells, episodes[0].cells)
    strata = decision_strata(
        aux=episodes[0].aux,
        reward=episodes[0].reward,
        done=episodes[0].done,
    )
    assert torch.equal(decoded.strata, strata)
    cache.get(0, 2)
    assert cache.resident == [(0, 2)]
    cache.get(0, 1)
    assert cache.resident == [(0, 2), (0, 1)]
    cache.get(0, 2)
    assert cache.resident == [(0, 1), (0, 2)]
    cache.get(0, 0)
    assert cache.resident == [(0, 0)]


def test_episode_cache_holds_exactly_its_capacity(tmp_path: Path) -> None:
    episodes = [_episode(40, tag=1), _episode(9, tag=2), _episode(30, tag=3)]
    full = _cache(tmp_path / "full", episodes, capacity=39)
    full.get(0, 2)
    full.get(0, 1)
    assert full.resident == [(0, 2), (0, 1)]
    # An episode larger than the whole capacity is still kept, alone.
    small = _cache(tmp_path / "small", episodes, capacity=8)
    small.get(0, 1)
    assert small.resident == [(0, 1)]
    small.get(0, 2)
    assert small.resident == [(0, 2)]


def test_episode_cache_segment_holds_its_decisions_and_the_next_frame(
    tmp_path: Path,
) -> None:
    episodes = [_episode(10, tag=1, floor=2), _episode(4, tag=2)]
    cache = _cache(tmp_path, episodes)
    assert [cache.decisions(0, 0), cache.decisions(0, 1)] == [10, 4]
    segment, strata = cache.segment(0, 0, start=3, stop=6)
    first = episodes[0]
    assert not segment.starts_episode
    assert torch.equal(segment.cells, first.cells[3:7])
    assert torch.equal(segment.aux, first.aux[3:7])
    for name in ("actions", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(segment, name)),
            cast("torch.Tensor", getattr(first, name))[3:6],
        ), name
    expected = decision_strata(aux=first.aux, reward=first.reward, done=first.done)
    assert torch.equal(strata, expected[3:7])
    whole, _ = cache.segment(0, 1, start=0, stop=4)
    assert whole.starts_episode
    assert len(whole.cells) == len(whole.actions) == 4
    head, _ = cache.segment(0, 1, start=0, stop=0)
    assert (len(head.cells), len(head.actions)) == (1, 0)


def test_a_branch_stored_as_frames_starts_mid_episode(tmp_path: Path) -> None:
    branch = dataclasses.replace(_episode(5, tag=1), origin=b"state")
    cache = _cache(tmp_path, [branch, _episode(4, tag=2)])
    first, _ = cache.segment(0, 0, start=0, stop=5)
    assert not first.starts_episode
    assert cache.segment(0, 1, start=0, stop=4)[0].starts_episode


def test_window_decodes_only_the_episodes_it_uses(tmp_path: Path) -> None:
    cache = _cache(tmp_path, [_episode(10, tag=t) for t in range(3)])
    directory, line = cache.entries[0]
    damaged = read_summaries(directory, line)[2]
    path = directory / f"{line.shard}.frames.zst"
    payload = bytearray(path.read_bytes())
    assert damaged.frames is not None
    payload[damaged.frames.offset + damaged.frames.size // 2] ^= 0xFF
    path.write_bytes(bytes(payload))
    parts = window(cache, shard=0, episode=0, start=0, t_g=20, s_max=4)
    assert len(parts) == 1
    assert cache.resident == [(0, 0)]
    with pytest.raises(ValueError, match="CRC-32"):
        window(cache, shard=0, episode=2, start=0, t_g=20, s_max=4)


def test_window_starts_mid_episode_then_fills_with_following_episodes(
    tmp_path: Path,
) -> None:
    cache = _cache(
        tmp_path,
        [_episode(10, tag=1), _episode(3, tag=2), _episode(4, tag=3)],
    )
    parts = window(cache, shard=0, episode=0, start=6, t_g=20, s_max=4)
    segments = [segment for segment, _ in parts]
    assert [s.starts_episode for s in segments] == [False, True, True]
    assert [len(s.actions) for s in segments] == [4, 3, 2]
    assert [len(s.cells) for s in segments] == [4, 3, 3]
    assert segments[0].cells[0, 1, 0].item() == 6
    assert [int(s.cells[0, 0, 0]) for s in segments] == [1, 2, 3]
    packed, _ = pack([parts], t_g=20, s_max=4)
    assert packed.kind[0, -1].item() != 3


def test_window_pads_when_the_shard_runs_out(tmp_path: Path) -> None:
    cache = _cache(tmp_path, [_episode(10), _episode(4)])
    parts = window(cache, shard=0, episode=1, start=0, t_g=20, s_max=4)
    assert len(parts) == 1
    packed, _ = pack([parts], t_g=20, s_max=4)
    assert packed.kind[0, 9:].tolist() == [3] * 11


def test_window_stops_at_s_max_segments(tmp_path: Path) -> None:
    cache = _cache(tmp_path, [_episode(1) for _ in range(10)])
    assert len(window(cache, shard=0, episode=0, start=0, t_g=40, s_max=4)) == 4


def _corpus(tmp_path: Path) -> Path:
    train = tmp_path / "train" / "0" / "w0"
    val = tmp_path / "val" / "0" / "w0"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    lines = [
        (
            train,
            write_shard(
                train,
                index=0,
                episodes=[_episode(40, tag=1), _episode(9, tag=2, floor=1)],
                provenance={},
            ),
        ),
        (
            train,
            write_shard(
                train,
                index=1,
                episodes=[_episode(30, tag=3, floor=2)],
                provenance={},
            ),
        ),
        (
            val,
            write_shard(
                val,
                index=0,
                episodes=[_episode(20, tag=9, split=1)],
                provenance={},
            ),
        ),
    ]
    write_corpus(tmp_path / "corpora" / "tiny.json", entries=lines)
    return tmp_path


def _stream(root: Path, **overrides: object) -> ReplayStream:
    config = ReplayStream.Config()
    config.working_dir = root
    config.corpus = "corpora/tiny.json"
    config.windows = 2
    config.t_g = 16
    config.s_max = 4
    config.validation = StratifiedWindows.Config(batches=2)
    config.device = "cpu"
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def _take(
    stream: ReplayStream,
    count: int,
) -> list[dict[str, PackedBatch | torch.Tensor]]:
    return list(itertools.islice(stream.train_dataloader(), count))


def _packed(batch: Mapping[str, object]) -> PackedBatch:
    media = batch["media"]
    assert isinstance(media, PackedBatch)
    return media


def test_micro_batch_is_a_pure_function_of_its_key(tmp_path: Path) -> None:
    stream = _stream(_corpus(tmp_path))
    sampler, cache = stream.train_sampler, stream.train_cache
    geometry = {"windows": 2, "t_g": 16, "s_max": 4}
    first, _ = micro_batch(sampler, cache, key=(0, 0, 5, 1), **geometry)
    again, _ = micro_batch(sampler, cache, key=(0, 0, 5, 1), **geometry)
    _assert_same_batch(first, again)
    other_ranks = [
        micro_batch(sampler, cache, key=(0, rank, 5, 1), **geometry)[0]
        for rank in range(1, 5)
    ]
    assert any(not torch.equal(b.cells, first.cells) for b in other_ranks)


def test_stream_resumes_from_its_served_count(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    uninterrupted = _take(_stream(root, micro_batches_per_step=2), 4)
    first = _stream(root, micro_batches_per_step=2)
    _take(first, 3)
    state = first.state_dict()
    assert state["batches"] == 3
    resumed = _stream(root, micro_batches_per_step=2)
    resumed.load_state_dict(state)
    _assert_same_batch(_packed(_take(resumed, 1)[0]), _packed(uninterrupted[3]))


def test_train_and_eval_read_their_own_splits(tmp_path: Path) -> None:
    stream = _stream(_corpus(tmp_path))
    for batch in _take(stream, 3):
        assert 9 not in _packed(batch).cells[:, 0, 0].tolist()
    evaluation = list(stream.eval_dataloader())
    assert len(evaluation) == 2
    for batch in evaluation:
        assert set(_packed(batch).cells[:, 0, 0].tolist()) == {9}
        stratum = batch["stratum"]
        assert isinstance(stratum, torch.Tensor)
        assert stratum.shape == (2, 16)


def test_stream_caches_episodes_across_its_training_shards(tmp_path: Path) -> None:
    stream = _stream(_corpus(tmp_path), cached_decisions=40)
    cache = stream.train_cache
    assert isinstance(cache, EpisodeCache)
    cache.get(0, 1)
    cache.get(1, 0)
    assert cache.resident == [(0, 1), (1, 0)]
    cache.get(0, 0)
    assert cache.resident == [(0, 0)]


def test_counts_pad_to_their_multiple_with_inert_jobs_and_frames(
    tmp_path: Path,
) -> None:
    root = _corpus(tmp_path)
    plain = _take(_stream(root), 3)
    padded = _take(_stream(root, count_multiple=8), 3)
    for batch, reference in zip(padded, plain, strict=True):
        media, original = _packed(batch), _packed(reference)
        frames, jobs = len(original.aux), len(original.job_at)
        assert len(media.aux) % 8 == len(media.job_at) % 8 == 0
        assert len(media.aux) - frames < 8
        assert len(media.job_at) - jobs < 8
        for name in ("cells", "aux", *(f.name for f in _JOB_FIELDS)):
            limit = frames if name in {"cells", "aux"} else jobs
            head = cast("torch.Tensor", getattr(media, name))[:limit]
            assert torch.equal(head, cast("torch.Tensor", getattr(original, name))), (
                name
            )
        # A padded job is a start job with no memory and no next frame: nothing
        # of it is scored.
        assert media.job_is_start[jobs:].all()
        assert (media.job_memory[jobs:] == -1).all()
        assert (media.job_next[jobs:] == -1).all()


def test_eval_spans_weigh_every_decision_to_the_natural_mix(tmp_path: Path) -> None:
    # Two episodes with the same strata, one five times longer: drawn by span,
    # each weighted by its inverse draw probability, both count by decisions.
    lengths = (200, 1_000)
    episodes = [_episode(n, split=1) for n in lengths]
    index = build_index(episodes)
    config = EvalSpans.Config(spans=20_000, span_decisions=50)
    spans = EvalSpans(
        config,
        indexes=[index],
        cache=_cache(tmp_path, episodes),
        t_g=256,
        s_max=4,
        windows=1,
    )
    weighted = collections.defaultdict[int, float](float)
    drawn = collections.Counter[int]()
    for span in spans.drawn:
        weighted[span.episode] += span.weight * span.length
        drawn[span.episode] += span.length
    total = sum(weighted.values())
    for episode, length in enumerate(lengths):
        assert weighted[episode] / total == pytest.approx(length / 1_200, abs=0.01)
    # Entry decisions are drawn at square-root rates, so unweighted the short
    # episode is over-represented.
    assert drawn[0] / sum(drawn.values()) > 0.2
    counts = index.counts().double()
    assert total == pytest.approx(float(counts.sum()), rel=0.02)


def test_eval_spans_tile_each_episode_from_its_first_decision(tmp_path: Path) -> None:
    lengths = (30, 7, 45)
    episodes = [_episode(n, split=1) for n in lengths]
    config = EvalSpans.Config(spans=200, span_decisions=8)
    spans = EvalSpans(
        config,
        indexes=[build_index(episodes)],
        cache=_cache(tmp_path, episodes),
        t_g=64,
        s_max=4,
        windows=2,
    )
    for span in spans.drawn:
        assert span.first % 8 == 0
        assert span.length == min(8, lengths[span.episode] - span.first)
    assert {(s.episode, s.first) for s in spans.drawn} == {
        (e, first) for e, n in enumerate(lengths) for first in range(0, n, 8)
    }


def test_eval_span_targets_depend_on_neither_the_window_nor_the_ranks(
    tmp_path: Path,
) -> None:
    episodes = [_episode(n, tag=t, split=1) for t, n in enumerate((30, 7, 45, 12, 60))]
    cache = _cache(tmp_path, episodes)
    index = build_index(episodes)
    config = EvalSpans.Config(spans=12, span_decisions=8)
    one = _scored_targets(config, index, cache, t_g=64, world=1)
    assert _scored_targets(config, index, cache, t_g=128, world=1) == one
    assert _scored_targets(config, index, cache, t_g=64, world=2) == one
    assert _scored_targets(config, index, cache, t_g=64, world=3) == one


def test_eval_spans_score_only_the_span_after_its_history(tmp_path: Path) -> None:
    episodes = [_episode(n, tag=t, split=1) for t, n in enumerate((30, 60))]
    cache = _cache(tmp_path, episodes)
    config = EvalSpans.Config(spans=16, span_decisions=8)
    for t_g in (24, 64):
        spans = EvalSpans(
            config,
            indexes=[build_index(episodes)],
            cache=cache,
            t_g=t_g,
            s_max=4,
            windows=2,
        )
        for index in range(len(spans)):
            batch, _, weight = spans.micro_batch(index)
            for row, segment in itertools.product(range(len(weight)), range(4)):
                where = (batch.segment[row] == segment) & (batch.kind[row] == Kind.ACT)
                if not where.any():
                    continue
                frames = batch.frame_of[row].roll(1)[where].long()
                decisions = batch.cells[frames, 1, 0].long()
                scored = weight[row][where] > 0
                # The history before the span is unscored; the span is a
                # whole aligned block, up to t_g // 2 decisions after its start.
                first = int(decisions[scored][0])
                assert first % 8 == 0
                assert scored.tolist() == (decisions >= first).tolist()
                assert len(decisions) <= t_g // 2
                assert int(decisions[-1]) - first < 8


def test_eval_spans_score_each_episodes_first_frame_in_its_first_span(
    tmp_path: Path,
) -> None:
    episodes = [_episode(n, tag=t, split=1) for t, n in enumerate((30, 7, 45))]
    cache = _cache(tmp_path, episodes)
    config = EvalSpans.Config(spans=40, span_decisions=8)
    spans = EvalSpans(
        config,
        indexes=[build_index(episodes)],
        cache=cache,
        t_g=64,
        s_max=4,
        windows=2,
    )
    expected = collections.defaultdict[int, float](float)
    for span in spans.drawn:
        if span.first == 0:
            expected[span.episode] += span.weight
    assert expected.keys() == {0, 1, 2}
    scored = collections.defaultdict[int, float](float)
    history = 0
    for index in range(len(spans)):
        batch, _, weight = spans.micro_batch(index)
        start = batch.job_is_start
        frames = batch.job_next[start].long()
        # The start job generates the episode's first frame.
        for frame, value in zip(
            frames,
            weight.flatten()[batch.job_at[start].long()],
            strict=True,
        ):
            assert int(batch.cells[frame, 1, 0]) == 0
            scored[int(batch.cells[frame, 0, 0])] += float(value)
            history += float(value) == 0
    # A later span packed with its episode's start leaves that frame unscored.
    assert history > 0
    assert {k: v for k, v in scored.items() if v} == pytest.approx(dict(expected))


@pytest.mark.parametrize(
    ("spans", "span_decisions", "reason"),
    [(4, 16, "span_decisions"), (0, 8, "spans=0")],
    ids=("longer-than-a-window", "no-spans"),
)
def test_eval_spans_reject_a_draw_that_cannot_be_scored(
    tmp_path: Path,
    spans: int,
    span_decisions: int,
    reason: str,
) -> None:
    episodes = [_episode(40, split=1)]
    config = EvalSpans.Config(spans=spans, span_decisions=span_decisions)
    with pytest.raises(ValueError, match=reason):
        EvalSpans(
            config,
            indexes=[build_index(episodes)],
            cache=_cache(tmp_path, episodes),
            t_g=32,
            s_max=4,
            windows=1,
        )


def test_every_eval_spans_micro_batch_holds_all_its_windows(tmp_path: Path) -> None:
    # One span fills one of two windows; the other is padding, weighted 0, so
    # the last micro-batch compiles to the same shapes as the rest.
    episodes = [_episode(30, split=1)]
    spans = EvalSpans(
        EvalSpans.Config(spans=1, span_decisions=8),
        indexes=[build_index(episodes)],
        cache=_cache(tmp_path, episodes),
        t_g=64,
        s_max=4,
        windows=2,
    )
    assert len(spans) == 1
    batch, stratum, weight = spans.micro_batch(0)
    assert batch.kind.shape == stratum.shape == weight.shape == (2, 64)
    assert (batch.kind[1] == Kind.PAD).all()
    assert not weight[1].any()


def test_eval_spans_pack_branches_without_a_start(tmp_path: Path) -> None:
    branches = [
        dataclasses.replace(_episode(5, tag=t, split=1), origin=b"\x01" * 16)
        for t in range(2)
    ]
    cache = _cache(tmp_path, branches)
    config = EvalSpans.Config(spans=2, span_decisions=8)
    spans = EvalSpans(
        config,
        indexes=[build_index(branches)],
        cache=cache,
        t_g=20,
        s_max=4,
        windows=1,
    )
    # A whole 5-decision branch is 10 positions, with no start: two fill a window.
    assert len(spans) == 1
    batch, _, _ = spans.micro_batch(0)
    assert int(Kind.PAD) not in batch.kind[0].tolist()


def test_a_window_budgets_no_start_for_a_branch(tmp_path: Path) -> None:
    branch = dataclasses.replace(_episode(3, tag=1), origin=b"\x01" * 16)
    cache = _cache(tmp_path, [branch, _episode(10, tag=2)])
    parts = window(cache, shard=0, episode=0, start=0, t_g=10, s_max=4)
    # The branch's 6 positions leave 4: the next episode's start and 2 decisions.
    assert [len(s.actions) for s, _ in parts] == [3, 2]
    packed, _ = pack([parts], t_g=10, s_max=4)
    assert int(Kind.PAD) not in packed.kind[0].tolist()
    assert [cache.starts_at_reset(0, e) for e in range(2)] == [False, True]


def test_stream_scores_eval_spans_and_weights_every_batch(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    for batch in _take(_stream(root), 2) + list(_stream(root).eval_dataloader()):
        stratum, weight = batch["stratum"], batch["weight"]
        assert isinstance(stratum, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        assert torch.equal(weight, (stratum >= 0).float())
    spans = EvalSpans.Config(spans=6, span_decisions=4)
    stream = _stream(root, validation=spans, count_multiple=4)
    evaluation = list(stream.eval_dataloader())
    assert isinstance(stream.validation, EvalSpans)
    assert len(evaluation) == len(stream.validation) > 0
    for batch in evaluation:
        media, weight = _packed(batch), batch["weight"]
        assert isinstance(weight, torch.Tensor)
        assert set(media.cells[:, 0, 0].tolist()) <= {0, 9}
        assert len(media.job_at) % 4 == 0
        assert (weight > 0).any()


def test_training_split_scores_fixed_training_spans_then_validation_again(
    tmp_path: Path,
) -> None:
    spans = EvalSpans.Config(spans=6, span_decisions=4)
    stream = _stream(_corpus(tmp_path), validation=spans, train_spans=spans)
    with stream.training_split():
        fitted = list(stream.eval_dataloader())
        again = list(stream.eval_dataloader())
    assert stream.train_spans is not None
    assert len(fitted) == len(stream.train_spans) > 0
    for batch, repeat in zip(fitted, again, strict=True):
        _assert_same_batch(_packed(batch), _packed(repeat))
        assert set(_packed(batch).cells[:, 0, 0].tolist()) <= {0, 1, 2, 3}
    tags = {tag for b in fitted for tag in _packed(b).cells[:, 0, 0].tolist()}
    assert tags & {1, 2, 3}
    for batch in stream.eval_dataloader():
        assert set(_packed(batch).cells[:, 0, 0].tolist()) <= {0, 9}


def test_prefetch_thread_pins_on_the_consumers_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CUDA's current device is per thread, and a new thread's is GPU 0: pinning
    # there opened a 522 MiB context on GPU 0 for every other rank of 8.
    stream = _stream(_corpus(tmp_path))
    current: list[torch.device] = []

    def unpinned(self: torch.Tensor) -> torch.Tensor:
        return self

    # Scoped, not test-wide: priml's ``cleanup_cuda`` teardown synchronizes the
    # current device, and a GPU 3 faked past the test is an invalid ordinal on
    # a host with fewer GPUs whose worker already holds a CUDA context.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "current_device", lambda: 3)
        patch.setattr(torch.cuda, "set_device", current.append)
        patch.setattr(torch.Tensor, "pin_memory", unpinned)
        prefetch = _Stream(
            stream._train_batch,
            first=0,
            count=1,
            device=torch.device("cuda"),
        )
        assert prefetch.device == torch.device("cuda", 3)
        ready: queue.Queue[_Item | BaseException | None] = queue.Queue()
        prefetch._fill(ready, threading.Event(), 0)
    assert current == [torch.device("cuda", 3)]
    assert ready.qsize() == 2


def test_training_split_needs_its_spans(tmp_path: Path) -> None:
    stream = _stream(_corpus(tmp_path))
    assert stream.train_spans is None
    with pytest.raises(ValueError, match="train_spans"), stream.training_split():
        pass


_JOB_FIELDS = [f for f in dataclasses.fields(PackedBatch) if f.name.startswith("job_")]


def _scored_targets(
    config: EvalSpans.Config,
    index: ShardIndex,
    cache: EpisodeCache,
    *,
    t_g: int,
    world: int,
) -> dict[tuple[int, int], float]:
    """Return every scored act job's (episode tag, decision) and summed weight."""
    scored = collections.defaultdict[tuple[int, int], float](float)
    for rank in range(world):
        spans = EvalSpans(
            config,
            indexes=[index],
            cache=cache,
            t_g=t_g,
            s_max=4,
            windows=2,
            rank=rank,
            world=world,
        )
        for batch_index in range(len(spans)):
            batch, _, weight = spans.micro_batch(batch_index)
            act = (batch.kind == Kind.ACT) & (weight > 0)
            frames = batch.frame_of.roll(1, dims=-1)[act].long()
            for frame, value in zip(frames, weight[act], strict=True):
                key = (int(batch.cells[frame, 0, 0]), int(batch.cells[frame, 1, 0]))
                scored[key] += float(value)
    return {key: round(value, 3) for key, value in scored.items()}


def test_prefetch_thread_stops_when_the_consumer_stops(tmp_path: Path) -> None:
    stream = _stream(_corpus(tmp_path))
    before = threading.active_count()
    batches = iter(stream.train_dataloader())
    next(batches)
    assert threading.active_count() == before + 1
    del batches
    for thread in threading.enumerate():
        if thread.name == "craftax-replay-stream":
            thread.join(timeout=5)
    assert threading.active_count() == before


def test_the_prefetch_thread_hands_a_producers_error_to_the_consumer() -> None:
    stream = _Stream(_damaged, first=0, device=torch.device("cpu"))
    errors: list[BaseException] = []
    consumer = threading.Thread(target=_first_error, args=(stream, errors), daemon=True)
    consumer.start()
    # Bounded: were the error lost, the consumer would wait on the worker forever.
    consumer.join(timeout=10)
    assert not consumer.is_alive()
    assert [(type(e), str(e)) for e in errors] == [
        (ValueError, "damaged episode at batch 0"),
    ]


def _damaged(index: int) -> _Item:
    """Fail to produce micro-batch ``index``, as a damaged episode does."""
    raise ValueError(f"damaged episode at batch {index}")


def _first_error(stream: _Stream, errors: list[BaseException]) -> None:
    """Take ``stream``'s first micro-batch, keeping the error it raises."""
    try:
        next(iter(stream))
    except ValueError as error:
        errors.append(error)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
