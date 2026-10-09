"""Check the committed fixture, the encoders it was written by, and how it is minted.

The fixture is static data the viewer's tests read: its files must be what
today's encoders write of their own contents, and decode to its expected
values. The minting is tested in pieces on tiny inputs: the scripted ladder
trip on the eager game's tiny world, and the site and expected values of
hand-built ghosts.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import dataclasses
import gzip
import hashlib
import json

import numpy as np
import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game import step
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    DEFAULT_MAX_TIMESTEPS,
    MAP_SIZE,
    Achievement,
    Action,
    BlockType,
    env_state,
    env_stats,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules
from priml.baselines.craftax.ghosts import build, fixture
from priml.baselines.craftax.ghosts.fixture import (
    FIXTURE_TIME_RULE,
    FIXTURE_UNBROKEN,
    FIXTURE_WIN,
    expected,
    fixture_records,
    ladder_trip,
    resolved,
    shared,
    write_fixture,
)
from priml.baselines.craftax.ghosts.layout import (
    FORMAT,
    STEPS,
    GroupFile,
    Manifest,
    QuietRule,
    encode_events,
    encode_keeps,
    encode_sleep,
    encode_timeline,
    encode_window,
    read_events,
    read_keeps,
    read_sleep,
    read_timeline,
    read_window,
    read_world,
    world_bytes,
)
from priml.baselines.craftax.ghosts.testing import made_ghost
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import Episode, Receipt, Record
from priml.baselines.craftax.world_model.capture.seeds import splitmix64
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from priml.baselines.craftax.ghosts.extract import Ghost


_CWD: Final = Path(__file__).resolve().parent

_FIXTURE: Final = _CWD / "testdata" / "fixture"

_CENTRE: Final = MAP_SIZE // 2


def test_every_file_matches_its_manifest_digest() -> None:
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    for name, digest in manifest.files.items():
        data = (_FIXTURE / "site" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        assert len(data) == manifest.sizes[name], name


def test_each_fixture_file_is_what_the_encoders_write_of_its_contents() -> None:
    site = _FIXTURE / "site"
    text = (site / "manifest.json").read_text()
    manifest = from_plain(loads(text), Manifest)
    assert manifest.format == FORMAT
    assert text == json.dumps(dataclasses.asdict(manifest), indent=1) + "\n"
    assert {path.relative_to(site).as_posix() for path in site.rglob("*.*")} == {
        "manifest.json",
        *manifest.files,
    }
    for name in manifest.files:
        data = (site / name).read_bytes()
        if name.endswith(".json"):
            group = from_plain(loads(data.decode()), GroupFile)
            line = json.dumps(dataclasses.asdict(group), separators=(",", ":"))
            assert data.decode() == line + "\n", name
            continue
        raw = gzip.decompress(data)
        assert build._gzip(raw) == data, name
        assert _reencoded(name, raw, stride=manifest.sleep_stride) == raw, name


def test_the_expected_values_are_what_the_site_decodes_to() -> None:
    decoded = loads(json.dumps(expected(_FIXTURE / "site")))
    assert decoded == loads((_FIXTURE / "expected.json").read_text())


def test_the_fixture_crosses_floors_both_ways() -> None:
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    tier, _ = manifest.tiers
    every, short = tier.sets
    assert [group.count for group in every.groups] == [3]
    assert short.counts == (2,)
    assert short.composition.qualified == 2
    trip = _expected_episodes()[0]
    floors = [step[0] for step in from_plain(trip["path"], list[list[int]])]
    assert floors[0] == 0
    assert 1 in floors
    assert floors[floors.index(1) :].count(0) > 0


def test_the_fixture_plays_its_pinned_win_unbroken_and_compresses_the_other() -> None:
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    _, tier = manifest.tiers
    every, _, wins = tier.sets
    assert (tier.name, wins.name, wins.counts) == ("fixture-wins", "wins", (2,))
    assert wins.time_map is not None
    assert wins.time_map.unbroken == 0
    (group,) = wins.groups
    entries = from_plain(
        loads((_FIXTURE / "site" / group.path / "episodes.json").read_text()),
        GroupFile,
    ).episodes
    assert [e.sampling_seed for e in entries] == [str(FIXTURE_UNBROKEN), "0"]
    shown = [e for e in _expected_episodes() if e["set"] == "wins"]
    lengths: list[int] = []
    for episode in shown:
        steps = from_plain(episode["displayed"], list[int])
        decisions = from_plain(episode["decisions"], int)
        assert steps[0] == 0
        assert steps[-1] == decisions - 1
        assert steps == sorted(set(steps))
        lengths.append(len(steps) - decisions)
    unbroken, compressed = lengths
    assert unbroken == 0
    assert compressed < 0
    assert wins.time_map.steps <= wins.time_map.rule.steps
    assert every.time_map is None


def test_shared_entries_resolve_to_what_the_site_decodes_each_to() -> None:
    episodes = _expected_episodes()
    stored = from_plain(
        from_plain(
            loads((_FIXTURE / "expected.json").read_text()),
            dict[str, object],
        )["episodes"],
        list[dict[str, object]],
    )
    # The repeats are references, and each expands to a full entry.
    assert sum("same_as" in entry for entry in stored) >= 6
    assert all("path" in entry and "same_as" not in entry for entry in episodes)
    assert shared(episodes) == stored
    # A ghost in two sets decodes alike in both: the all and short sets of the
    # first tier share their dying episodes.
    every, short = (
        [e for e in episodes if (e["tier"], e["set"]) == ("fixture", name)]
        for name in ("all", "short")
    )
    assert short[0]["path"] == every[0]["path"]


def test_the_walk_to_a_ladder_steps_along_a_shortest_path_around_a_wall() -> None:
    states = new_states(1)
    state = env_state(states, 0)
    tiny_world(state, np.array([0], np.uint32))
    ladder = np.array([_CENTRE, _CENTRE + 6], np.int64)
    for row in range(_CENTRE - 4, _CENTRE + 5):
        state.map[0, row, _CENTRE + 3] = BlockType.STONE
    distance = fixture._distances(state.map[0], target=(_CENTRE, _CENTRE + 6))
    # Six columns, around the wall's end four rows up and back down.
    assert distance[_CENTRE, _CENTRE] == 6 + 2 * 5
    assert distance[_CENTRE, _CENTRE + 3] == _UNREACHED
    position = (_CENTRE, _CENTRE)
    for _ in range(16):
        action = fixture._toward(state, ladder=ladder, take=Action.DESCEND)
        dr, dc = STEPS[Action(action)]
        state.player_position[0] += dr
        state.player_position[1] += dc
        state.player_direction = action
        moved = (int(state.player_position[0]), int(state.player_position[1]))
        assert distance[moved] == distance[position] - 1
        position = moved
    assert position == (_CENTRE, _CENTRE + 6)
    assert fixture._toward(state, ladder=ladder, take=Action.DESCEND) == Action.DESCEND


def test_the_walk_strikes_a_creature_it_faces_on_its_path() -> None:
    states = new_states(1)
    state = env_state(states, 0)
    tiny_world(state, np.array([0], np.uint32))
    ladder = np.array([_CENTRE, _CENTRE + 2], np.int64)
    state.mob_bits[0, _CENTRE] = np.uint64(1) << np.uint64(_CENTRE + 1)
    # Facing up, the walk turns toward the creature first, then strikes it.
    assert fixture._toward(state, ladder=ladder, take=Action.DESCEND) == Action.RIGHT
    state.player_direction = Action.RIGHT
    assert fixture._toward(state, ladder=ladder, take=Action.DESCEND) == Action.DO


def test_a_ladder_out_of_reach_is_refused() -> None:
    states = new_states(1)
    state = env_state(states, 0)
    tiny_world(state, np.array([0], np.uint32))
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        state.map[0, 10 + dr, 10 + dc] = BlockType.WATER
    with pytest.raises(ValueError, match="out of reach"):
        fixture._toward(
            state,
            ladder=np.array([10, 10], np.int64),
            take=Action.DESCEND,
        )


def test_a_random_decision_is_a_legal_action_drawn_by_splitmix64() -> None:
    trip = fixture._Trip(stream=5, wander=0, phase=3)
    mask = np.zeros(ATN_DIM, np.uint8)
    mask[[1, 4, 7]] = 1
    state = env_state(new_states(1), 0)
    stream, draw = splitmix64(5)
    assert trip.act(state, mask) == [1, 4, 7][draw % 3]
    assert trip.stream == stream


def test_the_trip_descends_wanders_climbs_back_and_plays_on() -> None:
    # The tiny world's clock runs out at the 8th decision.
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 8)):
        record = ladder_trip(world_seed=1, sampling_seed=0, wander=1)
        floors = _floors(record)
        status = replay.verify(record)
    actions = record.actions.tolist()
    assert actions[:3] == [Action.RIGHT, Action.RIGHT, Action.DESCEND]
    climbed = actions.index(Action.ASCEND)
    assert floors[: climbed + 2] == [0, 0, 0, *[1] * (climbed - 2), 0]
    assert len(actions) == 8
    assert status == replay.MATCHED
    assert (record.receipt.world_seed, record.receipt.sampling_seed) == (1, 0)


def test_a_trip_that_does_not_end_is_refused() -> None:
    with (
        eager(world=tiny_world),
        pytest.raises(ValueError, match="did not end within 4"),
    ):
        ladder_trip(world_seed=1, sampling_seed=0, wander=1, max_decisions=4)


def test_the_fixture_records_are_the_trip_then_two_random_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trip = _record(0, decisions=3)
    monkeypatch.setattr(fixture, "ladder_trip", partial(_trip, record=trip))
    monkeypatch.setattr(replay, "record", _random_episode)
    records = fixture_records(world_seed=9)
    assert records[0] is trip
    assert [r.receipt.sampling_seed for r in records] == [0, 1, 2]
    assert [r.receipt.world_seed for r in records[1:]] == [9, 9]
    assert all(type(r) is Record for r in records)
    assert [len(r.actions) for r in records[1:]] == [4, 5]


def test_the_minted_site_has_both_tiers_by_the_fixtures_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [_record(seed, decisions=3) for seed in (0, 1, 2)]
    wins: list[int] = []
    monkeypatch.setattr(fixture, "fixture_records", partial(_records, records=records))
    monkeypatch.setattr(fixture, "extract", partial(_extracted, wins=wins))
    with eager(world=tiny_world):
        write_fixture(tmp_path, world_seed=1)
    site = tmp_path / "site"
    manifest = from_plain(loads((site / "manifest.json").read_text()), Manifest)
    assert sorted(set(wins)) == sorted({Achievement.DEFEAT_NECROMANCER, FIXTURE_WIN})
    assert [t.name for t in manifest.tiers] == ["fixture", "fixture-wins"]
    assert (manifest.counts, manifest.short_decisions) == ((3,), 160)
    assert manifest.quiet == QuietRule(per_live=1, min_run=16, keep=4)
    every, short = manifest.tiers[0].sets
    assert (every.counts, short.counts) == ((3,), (2,))
    *_, won = manifest.tiers[1].sets
    assert won.time_map is not None
    assert (won.time_map.rule, won.time_map.unbroken) == (FIXTURE_TIME_RULE, 0)
    entries = from_plain(
        loads((site / won.groups[0].path / "episodes.json").read_text()),
        GroupFile,
    ).episodes
    assert [e.sampling_seed for e in entries] == [str(FIXTURE_UNBROKEN), "0"]
    assert (tmp_path / "expected.json").read_text() == (
        json.dumps(expected(site), separators=(",", ":")) + "\n"
    )


_UNREACHED: Final = MAP_SIZE * MAP_SIZE


def _reencoded(name: str, raw: bytes, *, stride: int) -> bytes:
    """Return a site file's contents written again by the encoder of its kind."""
    stem = Path(name).name.split(".")[0]
    if stem == "world":
        return world_bytes(read_world(raw))
    if stem == "events":
        return encode_events(read_events(raw))
    if stem.startswith("creatures-w"):
        return encode_window(read_window(raw))
    if stem.startswith("timeline-n"):
        active, segments = read_timeline(raw)
        return encode_timeline(active, segments=segments)
    if stem in {"keeps", "keeps-sleep"}:
        return encode_keeps(read_keeps(raw))
    if stem == "sleep":
        return encode_sleep(read_sleep(raw, stride=stride))
    assert stem == "players", name
    return raw


def _floors(record: Record) -> list[int]:
    """Return the floor before each decision of ``record`` and after its last."""
    states, rng = replay.reset_world(record.receipt.world_seed)
    stats, state = new_stats(1), env_state(states, 0)
    floors = [int(state.player_level)]
    for action in ints(record.actions.numpy()):
        step.play_numba(state, rng, env_stats(stats, 0), action, Rules())
        floors.append(int(state.player_level))
    return floors


def _record(seed: int, *, decisions: int) -> Record:
    """Return a record of ``decisions`` NOOPs on world 1; only its receipt is read."""
    return Record(
        receipt=Receipt(
            world_seed=1,
            sampling_seed=seed,
            initial_state_hash=0,
            arm=0,
            split=0,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(2, dtype=torch.int64),
    )


def _records(*, world_seed: int, records: list[Record]) -> list[Record]:
    """Stand in for ``fixture.fixture_records``: return ``records``."""
    assert world_seed == 1
    return records


def _trip(*, world_seed: int, sampling_seed: int, record: Record) -> Record:
    """Stand in for ``fixture.ladder_trip``: return ``record``."""
    assert (world_seed, sampling_seed) == (9, 0)
    return record


def _random_episode(
    *,
    world_seed: int,
    sampling_seed: int,
    max_decisions: int,
) -> Episode:
    """Stand in for ``replay.record``: an episode of ``3 + sampling_seed`` decisions."""
    assert max_decisions == 20_000
    decisions = 3 + sampling_seed
    return Episode(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=sampling_seed,
            initial_state_hash=0,
            arm=0,
            split=0,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(2, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=torch.zeros(decisions, 51, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=torch.zeros(decisions, dtype=torch.bool),
        summary={},
    )


# Played to its death, episode 0 outlasts the short set's 160 decisions; won, it is
# longer than the fixture's 120-step rule, and episode 2 fits it.
def _extracted(record: Record, *, ordinal: int, win: int, wins: list[int]) -> Ghost:
    """Stand in for ``extract``: episodes 0 and 2 win at ``FIXTURE_WIN``, 2 the shorter."""
    wins.append(win)
    seed = record.receipt.sampling_seed
    if win == FIXTURE_WIN and ordinal in {0, 2}:
        return made_ghost(
            ordinal,
            decisions=122 if ordinal == 0 else 118,
            outcome="win",
            sampling_seed=seed,
            sleeps=((60, 9),),
        )
    return made_ghost(
        ordinal,
        decisions=(220, 154, 150)[ordinal],
        outcome="death",
        sampling_seed=seed,
    )


def _expected_episodes() -> list[dict[str, object]]:
    data = from_plain(
        loads((_FIXTURE / "expected.json").read_text()),
        dict[str, object],
    )
    return resolved(from_plain(data["episodes"], list[dict[str, object]]))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
