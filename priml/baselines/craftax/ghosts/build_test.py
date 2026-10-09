"""Check the site's files: sets in nested groups, timelines, the manifest, and captures.

The ghosts are built by hand, so each set's rule meets known lengths and
outcomes, and the world is the tiny one ``eager`` generates; extraction and
world generation have their own tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import dataclasses
import gzip
import hashlib
import sys
import zlib

import numpy as np
import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import (
    NUM_LEVELS,
    env_state,
)
from priml.baselines.craftax.ghosts import build
from priml.baselines.craftax.ghosts.build import (
    TierGhosts,
    read_pool,
    write_site,
)
from priml.baselines.craftax.ghosts.extract import (
    TIMELINE,
    Ghost,
)
from priml.baselines.craftax.ghosts.layout import (
    EpisodeEntry,
    GroupFile,
    Manifest,
    QuietRule,
    TimeRule,
    displayed,
    read_events,
    read_keeps,
    read_sleep,
    read_timeline,
    read_window,
    read_world,
)
from priml.baselines.craftax.ghosts.sets import (
    Pool,
    activity,
    keep_runs,
    quiet_segments,
)
from priml.baselines.craftax.ghosts.testing import made_ghost
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    Receipt,
    write_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from priml.baselines.craftax.world_model.archive import Record


_WORLD: Final = 1
_SHORT: Final = 180
_RULE: Final = TimeRule(steps=60, levels=((1, 2), (0, 1), (0, 0)))


# Its creatures stand in row ``ordinal``, so groups and windows that mix ghosts up show;
# an odd ordinal has an escape row and reaches floor 1. Each sleep of ``k`` ticks has
# ``(k - 1) // 4`` samples of one creature, and its first sample a ripe plant.
_GHOSTS: Final = (
    made_ghost(0, decisions=150, outcome="death"),
    made_ghost(1, decisions=300, outcome="timeout"),
    made_ghost(2, decisions=120, outcome="death"),
)
"""The main capture: two deaths within ``_SHORT`` decisions and a timeout past it."""

_WON: Final = (
    made_ghost(0, decisions=90, outcome="win", sleeps=((43, 13), (77, 3))),
    made_ghost(1, decisions=50, outcome="win"),
    made_ghost(2, decisions=100, outcome="death"),
)
"""A tier that wins: a win with a sleep of 3 samples, a win short enough to pin, a death."""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    out = tmp_path_factory.mktemp("site") / "data"
    pool = Pool(root="r", capped=False, provenance={"k": "v"}, ghosts=_GHOSTS)
    with eager(world=tiny_world):
        manifest = write_site(
            out,
            tiers=[TierGhosts(name="a", arm=0, pools=(pool,))],
            counts=(1, 3),
            short_decisions=_SHORT,
            window=64,
        )
    return out, manifest


@pytest.fixture
def tiny() -> Generator[None]:
    with eager(world=tiny_world):
        yield


def test_each_set_holds_its_episodes_in_nested_groups(
    site: tuple[Path, Manifest],
) -> None:
    out, manifest = site
    (tier,) = manifest.tiers
    every, short = tier.sets
    assert (every.name, every.counts) == ("all", (1, 3))
    assert (short.name, short.counts) == ("short", (1, 2))
    assert [g.path for g in every.groups] == ["a/all/g0", "a/all/g1"]
    assert [g.windows for g in every.groups] == [3, 5]
    for episode_set, members in ((every, _GHOSTS), (short, _GHOSTS[::2])):
        indices = [
            entry.ordinal
            for group in episode_set.groups
            for entry in _check_group(out / group.path, group.first)
        ]
        assert indices == [g.ordinal for g in members]


@pytest.mark.parametrize("pin", [False, True])
def test_a_tier_that_wins_gets_a_wins_set_of_time_mapped_wins(
    tmp_path: Path,
    tiny: None,
    pin: bool,
) -> None:
    del tiny
    winners = [g for g in _WON if g.outcome == "win"]
    pinned = winners[0] if pin else None
    rule = dataclasses.replace(_RULE, steps=pinned.decisions) if pinned else _RULE
    pool = Pool(root="r", capped=False, provenance={"k": "v"}, ghosts=_WON)
    manifest = write_site(
        tmp_path / "data",
        tiers=[
            TierGhosts(
                name="w",
                arm=0,
                pools=(pool,),
                unbroken=pinned.sampling_seed if pinned else None,
            ),
        ],
        counts=(1, 3),
        short_decisions=_SHORT,
        window=64,
        time_rule=rule,
    )
    every, short, wins = manifest.tiers[0].sets
    assert (every.time_map, short.time_map) == (None, None)
    assert wins.name == "wins"
    assert wins.counts == (1, 2)
    assert wins.timelines == ()
    lengths: list[int] = []
    for group in wins.groups:
        keeps = read_keeps(
            gzip.decompress(
                (tmp_path / "data" / group.path / "keeps.bin.gz").read_bytes(),
            ),
        )
        members = winners[group.first : group.first + group.count]
        assert len(keeps) == len(members)
        for runs, ghost in zip(keeps, members, strict=True):
            if ghost is pinned:
                np.testing.assert_array_equal(runs, [[0, ghost.decisions]])
            else:
                np.testing.assert_array_equal(
                    runs,
                    keep_runs(ghost.active, rule=rule)[0],
                )
            shown = displayed(runs)
            assert shown[0] == 0
            assert shown[-1] == ghost.decisions - 1
            lengths.append(len(shown))
    time_map = wins.time_map
    assert time_map is not None
    assert max(lengths) <= rule.steps
    assert (time_map.steps, time_map.kept, time_map.shortest) == (
        max(lengths),
        sum(lengths),
        min(lengths),
    )
    assert time_map.unbroken == (0 if pinned else None)
    assert sum(time_map.levels) == len(winners) - bool(pinned)


def test_an_unbroken_set_plays_every_decision_of_each_win_that_fits(
    tmp_path: Path,
    tiny: None,
) -> None:
    del tiny
    winners = [g for g in _WON if g.outcome == "win"]
    pinned = winners[-1]
    steps = max(g.decisions for g in winners)
    pool = Pool(root="r", capped=False, provenance={"k": "v"}, ghosts=_WON)
    manifest = write_site(
        tmp_path / "data",
        tiers=[
            TierGhosts(name="w", arm=0, pools=(pool,), unbroken=pinned.sampling_seed),
        ],
        counts=(1, 3),
        short_decisions=_SHORT,
        window=64,
        time_rule=_RULE,
        whole_wins=(3, steps),
    )
    *_, wins, unbroken = manifest.tiers[0].sets
    assert (wins.name, unbroken.name) == ("wins", "unbroken")
    time_map = unbroken.time_map
    assert time_map is not None
    assert unbroken.episodes == len(winners)
    assert (time_map.unbroken, time_map.whole, time_map.rule.steps) == (
        0,
        unbroken.episodes,
        steps,
    )
    data = tmp_path / "data"
    keeps = [
        runs
        for group in unbroken.groups
        for runs in read_keeps(
            gzip.decompress((data / group.path / "keeps.bin.gz").read_bytes()),
        )
    ]
    entries = [
        entry
        for group in unbroken.groups
        for entry in from_plain(
            loads((data / group.path / "episodes.json").read_text()),
            GroupFile,
        ).episodes
    ]
    assert entries[0].sampling_seed == str(pinned.sampling_seed)
    for runs, entry in zip(keeps, entries, strict=True):
        np.testing.assert_array_equal(runs, [[0, entry.decisions]])


def test_a_time_mapped_set_also_maps_the_view_that_shows_sleep(
    tmp_path: Path,
    tiny: None,
) -> None:
    del tiny
    winners = [g for g in _WON if g.outcome == "win"]
    pool = Pool(root="r", capped=False, provenance={"k": "v"}, ghosts=_WON)
    manifest = write_site(
        tmp_path / "data",
        tiers=[TierGhosts(name="w", arm=0, pools=(pool,))],
        counts=(1, 3),
        short_decisions=_SHORT,
        window=64,
        time_rule=_RULE,
    )
    assert manifest.sleep_stride == 4
    *_, wins = manifest.tiers[0].sets
    sleep_map = wins.sleep_map
    assert sleep_map is not None
    data, lengths = tmp_path / "data", list[int]()
    for group in wins.groups:
        keeps = read_keeps(
            gzip.decompress((data / group.path / "keeps-sleep.bin.gz").read_bytes()),
        )
        sleeps = read_sleep(
            gzip.decompress((data / group.path / "sleep.bin.gz").read_bytes()),
            stride=manifest.sleep_stride,
        )
        for runs, slept, ghost in zip(
            keeps,
            sleeps,
            winners[group.first : group.first + group.count],
            strict=True,
        ):
            np.testing.assert_array_equal(slept.sleeps, ghost.sleeps)
            np.testing.assert_array_equal(slept.samples, ghost.sleep_samples)
            np.testing.assert_array_equal(slept.changes, ghost.sleep_changes)
            assert slept.creatures == ghost.sleep_creatures
            # Every sleep is kept, and the view adds a step per sample.
            shown = set(ints(displayed(runs)))
            assert set(ints(ghost.sleeps[:, 0])) <= shown
            lengths.append(len(shown) + len(ghost.sleep_samples) - 1)
    assert (sleep_map.steps, sleep_map.kept, sleep_map.samples) == (
        max(lengths),
        sum(lengths),
        sum(len(g.sleep_samples) - 1 for g in winners),
    )
    assert sleep_map.samples == 3


def test_a_pinned_win_too_long_to_play_unbroken_fails_the_build(
    tmp_path: Path,
    tiny: None,
) -> None:
    del tiny
    pinned = _WON[0]
    pool = Pool(root="r", capped=False, provenance={"k": "v"}, ghosts=_WON)
    with pytest.raises(ValueError, match="cannot play unbroken"):
        write_site(
            tmp_path / "data",
            tiers=[
                TierGhosts(
                    name="w",
                    arm=0,
                    pools=(pool,),
                    unbroken=pinned.sampling_seed,
                ),
            ],
            counts=(1, 3),
            short_decisions=_SHORT,
            window=64,
            time_rule=TimeRule(steps=pinned.decisions - 1),
        )


def test_timelines_sum_the_shown_episodes_activity(
    site: tuple[Path, Manifest],
) -> None:
    out, manifest = site
    every = manifest.tiers[0].sets[0]
    assert [t.count for t in every.timelines] == [1, 3]
    # Quiet runs collapse: more than one kept range.
    assert min(t.segments for t in every.timelines) > 1
    for timeline in every.timelines:
        active, segments = read_timeline(
            gzip.decompress((out / timeline.path).read_bytes()),
        )
        shown = _GHOSTS[: timeline.count]
        expected, live, decisive = activity(shown)
        np.testing.assert_array_equal(active, expected)
        np.testing.assert_array_equal(
            segments,
            quiet_segments(expected, live=live, decisive=decisive, rule=QuietRule()),
        )
        assert timeline.decisions == max(g.decisions for g in shown)
        assert timeline.kept == (segments[:, 1] - segments[:, 0]).sum()
        assert active.sum() == sum(
            np.count_nonzero(np.frombuffer(g.active, np.uint8) & TIMELINE)
            for g in shown
        )


def test_the_manifest_names_every_file_with_its_size_and_digest(
    site: tuple[Path, Manifest],
) -> None:
    out, manifest = site
    written = {
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest.files) == written == set(manifest.sizes)
    for name, digest in manifest.files.items():
        data = (out / name).read_bytes()
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (
            digest,
            manifest.sizes[name],
        )
    assert from_plain(loads((out / "manifest.json").read_text()), Manifest) == manifest
    assert manifest.short_decisions == _SHORT
    assert manifest.tiers[0].sources[0].episodes == 3


def test_the_world_file_is_the_reset_world_and_the_start_its_player(
    site: tuple[Path, Manifest],
    tiny: None,
) -> None:
    del tiny
    out, manifest = site
    world = read_world(gzip.decompress((out / "world.bin.gz").read_bytes()))
    state = env_state(replay.reset_world(_WORLD)[0], 0)
    np.testing.assert_array_equal(world.block, state.map)
    np.testing.assert_array_equal(world.item, state.item_map)
    np.testing.assert_array_equal(world.light, state.light_map)
    np.testing.assert_array_equal(world.up_ladders, state.up_ladders)
    np.testing.assert_array_equal(world.down_ladders, state.down_ladders)
    assert manifest.start == (0, 24, 24, 3)
    assert manifest.world_seed == _WORLD


def test_a_file_gzips_to_one_header_on_every_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = build._gzip(b"ghosts")
    # Magic, deflate, no flags, no timestamp, level 9's XFL, then OS 255.
    assert packed[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    assert gzip.decompress(packed) == b"ghosts"
    monkeypatch.setattr(gzip, "compress", _gzip_compress_312)
    assert build._gzip(b"ghosts") == packed


def test_stats_and_composition_count_each_set(site: tuple[Path, Manifest]) -> None:
    _, manifest = site
    every, short = manifest.tiers[0].sets
    first, both = every.stats
    assert (first.count, both.count) == (1, 3)
    assert both.mean_return == np.mean([g.achievement_return for g in _GHOSTS])
    assert both.mean_decisions == np.mean([g.decisions for g in _GHOSTS])
    assert both.reached == (3, 1, *[0] * (NUM_LEVELS - 2))
    assert both.deaths == (2, *[0] * (NUM_LEVELS - 1))
    assert first.deaths[0] == 1
    assert (both.timeouts, both.wins, both.escapes) == (1, 0, 1)
    assert (every.composition.run, every.composition.qualified) == (3, 3)
    assert short.composition.qualified == 2
    assert short.composition.death_decisions == (120, 150)


def test_a_site_needs_one_world_enough_ghosts_and_a_new_directory(
    tmp_path: Path,
    site: tuple[Path, Manifest],
    tiny: None,
) -> None:
    del tiny
    other = dataclasses.replace(_GHOSTS[0], world_seed=_WORLD + 1)
    for tier_ghosts, counts, capped in (
        ((*_GHOSTS, other), (1, 4), False),
        (_GHOSTS, (1, 4), False),
        (_GHOSTS, (1, 3), True),
    ):
        pool = Pool(root="r", capped=capped, provenance={}, ghosts=tier_ghosts)
        with pytest.raises(ValueError, match="one world"):
            write_site(
                tmp_path / "x",
                tiers=[TierGhosts(name="a", arm=0, pools=(pool,))],
                counts=counts,
            )
    pool = Pool(root="r", capped=False, provenance={}, ghosts=_GHOSTS)
    with pytest.raises(FileExistsError):
        write_site(
            site[0],
            tiers=[TierGhosts(name="a", arm=0, pools=(pool,))],
            counts=(1, 3),
            window=64,
        )


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a capture of the main ghosts' records, extraction looking each one up.

    The train split holds ordinals 2 and 0, the validation split 1. Each
    record's actions are its ordinal plus one, so a ghost read back shows
    which record it came from.
    """
    monkeypatch.setattr(build, "extract", _extracted)
    root = tmp_path / "capture"
    for split, chosen in ((TRAIN, (2, 0)), (VALIDATION, (1,))):
        directory = shard_directory(root, split=split, arm=1, worker=0)
        directory.mkdir(parents=True)
        write_shard(
            directory,
            index=0,
            episodes=[_episode(_GHOSTS[i], split=split) for i in chosen],
            provenance={"checkpoint": "c"},
        )
    return root


def test_a_capture_merges_both_splits_by_ordinal(archive: Path) -> None:
    pool = read_pool(archive, arm=1, capped=True, workers=2)
    assert (pool.provenance, pool.capped) == ({"checkpoint": "c"}, True)
    assert [g.ordinal for g in pool.ghosts] == [0, 1, 2]
    assert [g.players for g in pool.ghosts] == [
        bytes([g.ordinal + 1]) * g.decisions for g in _GHOSTS
    ]
    kept = read_pool(archive, arm=1, capped=False, workers=2, worlds=(_WORLD,))
    assert [g.players for g in kept.ghosts] == [g.players for g in pool.ghosts]
    with pytest.raises(ValueError, match="holds 0 episodes"):
        read_pool(archive, arm=1, capped=False, workers=2, worlds=(_WORLD + 1,))


def test_a_summary_that_disagrees_with_the_replay_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build, "extract", _extracted)
    directory = shard_directory(tmp_path, split=TRAIN, arm=0, worker=0)
    directory.mkdir(parents=True)
    claimed = dataclasses.replace(_GHOSTS[0], outcome="timeout")
    write_shard(
        directory,
        index=0,
        episodes=[_episode(claimed, split=TRAIN)],
        provenance={},
    )
    with pytest.raises(ValueError, match="its summary says"):
        read_pool(tmp_path, arm=0, capped=False, workers=1)


def test_the_command_line_builds_a_site(
    archive: Path,
    tmp_path: Path,
    tiny: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del tiny
    out = tmp_path / "site"
    argv = [
        *("build.py", str(archive), str(out), "--tier", "t=1", "--counts", "2,3"),
        *("--extra", str(archive), "--capped", str(archive), "--unbroken-set", "3:100"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert build.main() == 0
    manifest = from_plain(loads((out / "manifest.json").read_text()), Manifest)
    (tier,) = manifest.tiers
    assert [g.count for g in tier.sets[0].groups] == [2, 1]
    assert [s.capped for s in tier.sources] == [False, False, True]
    assert tier.sets[1].composition.run == 9
    assert "world.bin.gz" in capsys.readouterr().out


def test_the_command_line_refuses_an_unbroken_win_of_a_tier_not_built(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        *("build.py", str(archive), str(tmp_path / "site"), "--tier", "t=1"),
        *("--counts", "2,3", "--unbroken", "boss=5"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        build.main()
    assert not (tmp_path / "site").exists()


def _extracted(record: Record, *, ordinal: int, stride: int) -> Ghost:
    """Stand in for ``extract``: the main ghost of ``ordinal``, its players the record's."""
    assert stride == 4
    return dataclasses.replace(
        _GHOSTS[ordinal],
        players=record.actions.numpy().tobytes(),
    )


def _episode(ghost: Ghost, *, split: int) -> Episode:
    """Return a captured episode of ``ghost``'s length, its summary saying how it ended."""
    decisions = ghost.decisions
    return Episode(
        receipt=Receipt(
            world_seed=ghost.world_seed,
            sampling_seed=ghost.sampling_seed,
            initial_state_hash=0,
            arm=1,
            split=split,
        ),
        actions=torch.full((decisions,), ghost.ordinal + 1, dtype=torch.uint8),
        hashes=torch.zeros(-(-decisions // 256) + 1, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=torch.zeros(decisions, 51, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=torch.zeros(decisions, dtype=torch.bool),
        summary={
            "episode": ghost.ordinal,
            "death": int(ghost.outcome == "death"),
            "timeout": int(ghost.outcome == "timeout"),
            "return": float(ghost.achievement_return),
        },
    )


def _check_group(directory: Path, first: int) -> tuple[EpisodeEntry, ...]:
    """Check one group's files against its ghosts; return its entries."""
    entries = from_plain(
        loads((directory / "episodes.json").read_text()),
        GroupFile,
    ).episodes
    players = gzip.decompress((directory / "players.bin.gz").read_bytes())
    events = read_events(gzip.decompress((directory / "events.bin.gz").read_bytes()))
    windows = [
        read_window(gzip.decompress(path.read_bytes()))
        for path in sorted(
            directory.glob("creatures-w*.bin.gz"),
            key=lambda path: int(path.name.split(".")[0].removeprefix("creatures-w")),
        )
    ]
    for i, entry in enumerate(entries):
        ghost = _GHOSTS[entry.ordinal]
        assert (entry.index, entry.source) == (first + i, 0)
        assert players[entry.players : entry.players + entry.decisions] == ghost.players
        for table, span, own in (
            (events.map, entry.map, ghost.events.map),
            (events.achievements, entry.achievements, ghost.events.achievements),
            (events.escapes, entry.escapes, ghost.events.escapes),
        ):
            np.testing.assert_array_equal(table[span[0] : span[0] + span[1]], own)
        assert b"".join(window[i] for window in windows) == ghost.creatures
        assert (entry.outcome, entry.end, entry.decisions) == (
            ghost.outcome,
            ghost.end,
            ghost.decisions,
        )
        assert (entry.sampling_seed, entry.floor_first) == (
            str(ghost.sampling_seed),
            ghost.floor_first,
        )
    return entries


def _gzip_compress_312(data: bytes, *, compresslevel: int, mtime: int) -> bytes:
    """Return ``gzip.compress`` as Python 3.12 runs it: zlib's header, the host's OS byte."""
    assert mtime == 0
    return zlib.compress(data, level=compresslevel, wbits=31)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
