"""Check the per-arm archive statistics: lengths, floors, and idle frames."""

from pathlib import Path

import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    ManifestLine,
    Receipt,
    Span,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.baselines.craftax.world_model.scripts import data_stats
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.lib.codec import from_plain, loads


def test_summaries_count_lengths_timeouts_floors_and_repeated_starts() -> None:
    summaries = [
        _summary(_episode(5, floors=[500], death=True, start=7), 500),
        _summary(_episode(5, floors=[1_000, 2_000], death=True, start=7), 3_000),
        _summary(
            _episode(5, floors=[10_000, 0, 60_000], timeout=True, start=8),
            70_000,
        ),
    ]
    stats = data_stats.summary_stats(summaries)
    assert stats["episodes"] == 3
    assert stats["decisions"] == 73_500
    buckets = from_plain(stats["buckets"], dict[str, object])
    assert from_plain(buckets["<1k"], dict[str, object])["episodes"] == 1
    assert from_plain(buckets["1k-4k"], dict[str, object])["decisions"] == 3_000
    assert from_plain(buckets[">=64k"], dict[str, object])["share"] == pytest.approx(
        70 / 73.5,
    )
    timeouts = from_plain(stats["timeouts"], dict[str, object])
    assert timeouts["episodes"] == 1
    assert timeouts["share_decisions"] == pytest.approx(70 / 73.5)
    assert from_plain(stats["deaths"], dict[str, object])["episodes"] == 2
    floors = from_plain(stats["floors"], dict[str, object])
    floor2 = from_plain(floors["2"], dict[str, object])
    assert floor2["episodes"] == 1
    assert floor2["decisions"] == 60_000
    assert from_plain(floors["0"], dict[str, object])["reach"] == 1
    assert from_plain(floors["1"], dict[str, object])["reach"] == pytest.approx(1 / 3)
    assert stats["repeated_starts"] == 2
    # The median decision sits in the 70,000-decision episode.
    assert from_plain(stats["length"], dict[str, object])["decision_p50"] == 70_000


def test_idle_counts_find_unchanged_repeated_and_long_idle_frames() -> None:
    # 10 distinct frames, a 200-decision two-frame loop, then 10 distinct frames.
    codes = [*range(10), *[100, 101] * 100, *range(20, 30)]
    episode = _episode(len(codes), codes=codes)
    counts = data_stats.idle_counts(episode)
    assert counts.decisions == 220
    # Frames 12..209 each repeat the frame two decisions earlier.
    assert counts.repeated == 198
    assert counts.idle == 198
    assert counts.unchanged == 0
    assert counts.noop == 220


def test_a_frame_differing_only_in_light_is_unchanged() -> None:
    episode = _episode(4, codes=[5, 5, 5, 6])
    episode.aux[:, 43] = torch.arange(4, dtype=torch.int16)
    counts = data_stats.idle_counts(episode)
    assert counts.unchanged == 2
    assert counts.repeated == 2
    # A two-decision repeat is not a long idle stretch.
    assert counts.idle == 0


def test_main_reports_each_split_and_arm_of_a_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _publish(tmp_path, [_episode(300, codes=[1] * 300)])
    val = _publish(
        tmp_path,
        [_episode(40, codes=list(range(40)))],
        split=VALIDATION,
        arm=2,
    )
    corpus = tmp_path / "corpora" / "c.json"
    write_corpus(corpus, entries=[train, val])
    output = tmp_path / "out" / "stats.json"
    argv = ["data_stats.py", str(corpus), "--frame-shards", "1", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output)])
    assert data_stats.main() == 0
    result = from_plain(loads(output.read_text()), dict[str, object])
    groups = from_plain(result["groups"], dict[str, object])
    assert sorted(groups) == ["train/arm1", "val/arm2"]
    frames = from_plain(
        from_plain(groups["train/arm1"], dict[str, object])["frames"],
        dict[str, object],
    )
    assert from_plain(frames["decisions"], int) == 300
    # Every frame after the first repeats it, and the run exceeds 128 decisions.
    assert from_plain(frames["idle"], float) == pytest.approx(299 / 300)
    val_frames = from_plain(
        from_plain(groups["val/arm2"], dict[str, object])["frames"],
        dict[str, object],
    )
    assert from_plain(val_frames["repeated"], float) == 0


def test_an_archive_root_names_its_shards_alike_however_it_is_spelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A directory seeds its shard's frame draw, so a relative spelling of a
    # root would draw other shards than an absolute one.
    _publish(tmp_path / "archive", [_episode(5)])
    monkeypatch.chdir(tmp_path)
    (spelled,) = data_stats.discover([Path("archive")])["train/arm1"]
    (absolute,) = data_stats.discover([tmp_path / "archive"])["train/arm1"]
    assert spelled == absolute


def test_a_replay_corpus_reports_as_its_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tmp_path / "frames"
    train = _publish(frames, [_episode(300, codes=[1] * 300)])
    val = _publish(
        frames,
        [_episode(40, codes=list(range(40)))],
        split=VALIDATION,
        arm=2,
    )
    write_corpus(frames / "corpora" / "c.json", entries=[train, val])
    replay_twin(frames, tmp_path / "replay", monkeypatch)
    groups: list[object] = []
    for root in (frames, tmp_path / "replay"):
        output = tmp_path / "out" / f"{root.name}.json"
        argv = ["data_stats.py", str(root / "corpora" / "c.json"), "--workers", "1"]
        argv += ["--frame-shards", "1"]
        monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output)])
        assert data_stats.main() == 0
        groups.append(
            from_plain(loads(output.read_text()), dict[str, object])["groups"],
        )
    assert "frames" in from_plain(
        from_plain(groups[0], dict[str, object])["train/arm1"],
        dict[str, object],
    )
    assert groups[0] == groups[1]


def _episode(
    decisions: int,
    *,
    codes: list[int] | None = None,
    floors: list[int] | None = None,
    death: bool = False,
    timeout: bool = False,
    start: int = 0,
    arm: int = 1,
    split: int = TRAIN,
) -> Episode:
    """Build an episode whose frame ``t`` is filled with ``codes[t]``."""
    codes = codes if codes is not None else [0] * decisions
    cells = torch.tensor(codes, dtype=torch.uint8)[:, None, None].expand(-1, 99, 8)
    aux = torch.zeros(decisions, 51, dtype=torch.int16)
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    floors = floors if floors is not None else [decisions]
    records = [
        {"reached": int(k < len(floors) and floors[k] > 0), "decisions": d}
        for k, d in enumerate([*floors, *[0] * (9 - len(floors))])
    ]
    return Episode(
        receipt=Receipt(
            world_seed=decisions,
            sampling_seed=0,
            initial_state_hash=start,
            arm=arm,
            split=split,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=cells.contiguous(),
        aux=aux,
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={
            "death": int(death),
            "timeout": int(timeout),
            "return": 1,
            "floors": records,
        },
    )


def _summary(episode: Episode, decisions: int) -> EpisodeSummary:
    """Return ``episode``'s summary line as if it held ``decisions`` decisions."""
    return EpisodeSummary(
        receipt=episode.receipt,
        decisions=decisions,
        bin=Span(offset=0, size=0, crc32=0),
        frames=Span(offset=0, size=0, crc32=0),
        summary=episode.summary,
    )


def _publish(
    root: Path,
    episodes: list[Episode],
    *,
    split: int = TRAIN,
    arm: int = 1,
) -> tuple[Path, ManifestLine]:
    """Publish ``episodes`` as one shard of worker 0; return its corpus entry."""
    directory = shard_directory(root, split=split, arm=arm, worker=0)
    directory.mkdir(parents=True)
    line = write_shard(directory, index=0, episodes=episodes, provenance={})
    return directory, line


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
