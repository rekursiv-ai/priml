"""Check that a ghost decodes to the game its record replays, and refuses what is not one.

The fixture's three episodes (a ladder trip down to floor 1 and back, and two
random ones) are replayed here step by step on the game, independently of the
extraction kernel; the decoded player bytes, map events, creature samples and
end must reproduce what that replay shows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataclasses
import itertools

from numpy.typing import NDArray

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game.rules import (
    damage_mob_at_numba,
    set_mob_bit_numba,
)
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    NUM_LEVELS,
    Achievement,
    Action,
    env_state,
    env_stats,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, play_numba
from priml.baselines.craftax.ghosts import extract as extract_module
from priml.baselines.craftax.ghosts.extract import Ghost, extract
from priml.baselines.craftax.ghosts.fixture import fixture_records
from priml.baselines.craftax.ghosts.layout import (
    World,
    decode_path,
    decode_samples,
    maps_before,
)
from priml.baselines.craftax.lib.arrays import int_rows, typed
from priml.baselines.craftax.world_model import replay


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.baselines.craftax.game.state import Array3, EnvState, Mobs
    from priml.baselines.craftax.world_model.archive import Record


_WORLD = 1
_CLASSES = (
    "melee_mobs",
    "passive_mobs",
    "ranged_mobs",
    "mob_projectiles",
    "player_projectiles",
)
# The move action that steps each projectile direction (LEFT, RIGHT, UP, DOWN),
# written out here apart from the kernel's.
_FACINGS = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Replayed:
    """What the game shows along one episode, read from its States."""

    path: NDArray[np.int64]
    creatures: list[list[tuple[int, int, int, int, int]]]
    maps: dict[int, tuple[NDArray[np.uint8], NDArray[np.uint8]]]
    final: NDArray[np.void]
    active: list[int]


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return fixture_records(world_seed=_WORLD)


@pytest.fixture(scope="module")
def ghosts(records: list[Record]) -> list[Ghost]:
    return [extract(record, ordinal=i) for i, record in enumerate(records)]


@pytest.fixture(scope="module")
def replays(records: list[Record], ghosts: list[Ghost]) -> list[_Replayed]:
    return [
        _replay(record, stop=ghost.decisions)
        for record, ghost in zip(records, ghosts, strict=True)
    ]


@pytest.fixture(scope="module")
def world() -> World:
    states, _ = replay.reset_world(_WORLD)
    return World(
        block=typed(states["map"], np.uint8)[0, ...],
        item=typed(states["item_map"], np.uint8)[0, ...],
        light=typed(states["light_map"], np.uint8)[0, ...],
        down_ladders=typed(states["down_ladders"], np.int32)[0, ...].astype(np.int64),
        up_ladders=typed(states["up_ladders"], np.int32)[0, ...].astype(np.int64),
    )


def test_the_ladder_trip_goes_down_and_comes_back(ghosts: list[Ghost]) -> None:
    trip = ghosts[0]
    assert trip.floor_first[1] > 0
    assert any(byte & 0x80 for byte in trip.players[trip.floor_first[1] :])


def test_decoded_players_are_the_replayed_places_and_facings(
    ghosts: list[Ghost],
    replays: list[_Replayed],
    world: World,
) -> None:
    for ghost, replayed in zip(ghosts, replays, strict=True):
        path = decode_path(
            ghost.players,
            escapes=ghost.events.escapes,
            world=world,
            start=(0, 24, 24, 3),
        )
        np.testing.assert_array_equal(path, replayed.path)


def test_map_events_rebuild_the_replayed_maps(
    ghosts: list[Ghost],
    replays: list[_Replayed],
    world: World,
) -> None:
    for ghost, replayed in zip(ghosts, replays, strict=True):
        for decision, (block, item) in replayed.maps.items():
            rebuilt = maps_before(world, map_events=ghost.events.map, decision=decision)
            np.testing.assert_array_equal(rebuilt[0], block)
            np.testing.assert_array_equal(rebuilt[1], item)


def test_creature_samples_are_the_replayed_creatures(
    ghosts: list[Ghost],
    replays: list[_Replayed],
) -> None:
    for ghost, replayed in zip(ghosts, replays, strict=True):
        assert decode_samples(ghost.creatures) == replayed.creatures


def test_the_end_is_the_last_state_and_floors_are_first_reached_there(
    records: list[Record],
    ghosts: list[Ghost],
    replays: list[_Replayed],
) -> None:
    for record, ghost, replayed in zip(records, ghosts, replays, strict=True):
        path = replayed.path
        assert ghost.end == tuple(path.item(-1, j) for j in range(4))
        assert ghost.outcome == (
            "death" if env_state(replayed.final, 0).player_health <= 0 else "timeout"
        )
        assert ghost.decisions == len(record.actions)
        assert ghost.floor_first == tuple(
            int(np.argmax(np.equal(path[:, 0], floor)))
            if np.equal(path[:, 0], floor).any()
            else -1
            for floor in range(NUM_LEVELS)
        )


def test_achievements_are_the_unlocked_ones_and_the_return_their_reward(
    ghosts: list[Ghost],
    replays: list[_Replayed],
) -> None:
    for ghost, replayed in zip(ghosts, replays, strict=True):
        unlocked = np.flatnonzero(env_state(replayed.final, 0).achievements)
        np.testing.assert_array_equal(
            np.sort(ghost.events.achievements[:, 1]),
            unlocked,
        )
        assert ghost.achievement_return == ACHIEVEMENT_REWARD_MAP[unlocked].sum()


def test_a_win_ends_the_ghost_at_its_first_unlock(
    records: list[Record],
    ghosts: list[Ghost],
) -> None:
    decision, achievement = (ghosts[1].events.achievements.item(0, j) for j in range(2))
    won = extract(records[1], ordinal=1, win=achievement)
    assert won.outcome == "win"
    assert won.decisions == decision + 1
    assert won.players == ghosts[1].players[: decision + 1]
    np.testing.assert_array_equal(
        won.events.map,
        ghosts[1].events.map[ghosts[1].events.map[:, 0] <= decision],
    )
    assert (
        int(Achievement.DEFEAT_NECROMANCER) not in ghosts[1].events.achievements[:, 1]
    )


@pytest.mark.parametrize("damage", ["action", "hash"])
def test_a_record_off_its_hashes_is_refused(
    records: list[Record],
    ghosts: list[Ghost],
    damage: str,
) -> None:
    record = records[1]
    if damage == "action":
        # A move turns the player even when blocked, so a last move the other
        # way leaves a final State the last hash does not match.
        actions = record.actions.clone()
        actions[-1] = next(move for move in range(1, 5) if move != ghosts[1].end[3])
        damaged = dataclasses.replace(record, actions=actions)
    else:
        hashes = record.hashes.clone()
        hashes[-1] ^= 1
        damaged = dataclasses.replace(record, hashes=hashes)
    with pytest.raises(ValueError, match="replay status"):
        extract(damaged, ordinal=1)


def test_a_branch_is_refused(records: list[Record]) -> None:
    branch = dataclasses.replace(records[1], origin=bytes(replay.SNAPSHOT_BYTES))
    with pytest.raises(ValueError, match="origin"):
        extract(branch, ordinal=1)


def test_a_truncated_record_stops_where_it_was_cut(
    records: list[Record],
    ghosts: list[Ghost],
) -> None:
    record, cut = records[1], 100
    states, rng = replay.reset_world(record.receipt.world_seed)
    stats = new_stats(1)
    for action in record.actions[:cut]:
        play_numba(env_state(states, 0), rng, env_stats(stats, 0), int(action), Rules())
    last = np.array([replay.fnv1a_numba(states.view(np.uint8))], np.uint64).view(
        np.int64,
    )
    truncated = dataclasses.replace(
        record,
        actions=record.actions[:cut],
        hashes=torch.cat([record.hashes[:1], torch.from_numpy(last)]),
        truncated=True,
    )
    ghost = extract(truncated, ordinal=1)
    assert (ghost.outcome, ghost.decisions) == ("truncated", cut)
    assert ghost.players == ghosts[1].players[:cut]
    with pytest.raises(ValueError, match="replay status"):
        extract(dataclasses.replace(truncated, truncated=False), ordinal=1)


def test_activity_flags_are_the_replayed_events(
    ghosts: list[Ghost],
    replays: list[_Replayed],
) -> None:
    for ghost, replayed in zip(ghosts, replays, strict=True):
        assert list(ghost.active) == replayed.active
    # Every flag but a fight occurs in the fixture's play; the next test fights.
    seen = 0
    for ghost in ghosts:
        for flags in ghost.active:
            seen |= flags
    assert seen == 127 - extract_module.FOUGHT


def test_a_hit_and_a_kill_are_fights_and_a_despawn_is_not() -> None:
    states, _ = replay.reset_world(_WORLD)
    state = env_state(states, 0)
    level = int(state.player_level)
    row, col = (int(v) for v in state.player_position)
    row -= 1  # The player faces up at the reset.
    cows = state.passive_mobs[level]
    cows.mask[0], cows.health[0], cows.type_id[0] = 1, 3.0, 0
    cows.position[0, 0], cows.position[0, 1] = row, col
    set_mob_bit_numba(state, level, row, col, True)
    trace = extract_module._new_trace(states, decisions=1, stride=4)
    hits: list[bool] = []
    while cows.mask[0]:
        score = extract_module._snapshot_fight_numba(state, trace, level)
        damage_mob_at_numba(state, level, row, col, np.float32(1.0), True, True)
        hits.append(extract_module._fought_numba(state, trace, level, score))
    assert hits == [True, True, True]
    cows.mask[0], cows.health[0] = 1, 3.0
    score = extract_module._snapshot_fight_numba(state, trace, level)
    cows.mask[0] = 0
    assert not extract_module._fought_numba(state, trace, level, score)


def test_a_sample_gives_each_projectile_its_facing_and_refuses_a_diagonal() -> None:
    states, _ = replay.reset_world(_WORLD)
    state = env_state(states, 0)
    level = int(state.player_level)
    arrows, fireballs = state.player_projectiles[level], state.mob_projectiles[level]
    directions = state.player_projectile_directions
    for slot, (direction, (row, col)) in enumerate(
        zip(((0, -1), (1, 0), (0, 1)), ((3, 4), (5, 6), (7, 8)), strict=True),
    ):
        arrows.mask[slot], arrows.type_id[slot] = 1, 0
        arrows.position[slot, 0], arrows.position[slot, 1] = row, col
        directions[level, slot, 0], directions[level, slot, 1] = direction
    fireballs.mask[1], fireballs.type_id[1] = 1, 2
    fireballs.position[1, 0], fireballs.position[1, 1] = 9, 10
    state.mob_projectile_dirs[level, 1, 0], state.mob_projectile_dirs[level, 1, 1] = (
        -1,
        0,
    )
    trace = extract_module._new_trace(states, decisions=1, stride=4)
    assert extract_module._sample_numba(state, trace, 0)
    run = trace.creatures[: trace.samples[1]].tobytes()
    projectiles = [c for c in decode_samples(run)[0] if c[0] >= 3]
    assert projectiles == [
        (3, 2, 9, 10, Action.UP.value),
        (4, 0, 3, 4, Action.LEFT.value),
        (4, 0, 5, 6, Action.DOWN.value),
        (4, 0, 7, 8, Action.RIGHT.value),
    ]
    directions[level, 2, 0], directions[level, 2, 1] = 1, 1
    assert not extract_module._sample_numba(state, trace, 0)


def test_a_change_on_an_untracked_floor_fails_the_extraction() -> None:
    states, rng = replay.reset_world(_WORLD)
    actions = np.zeros(300, np.uint8)
    trace = extract_module._new_trace(states, decisions=len(actions), stride=4)
    trace.block[5, 0, 0] ^= 1
    stats = new_stats(1)
    copy = extract_module._Copy(
        states=states.copy(),
        rng=rng.copy(),
        stats=stats.copy(),
    )
    status = extract_module._trace_numba(
        states, states.view(np.uint8), rng, stats, actions, (Rules(), Rules(collapse_sleep=False)),
        trace, copy, 0, 4, 49,
    )  # fmt: skip
    assert status == extract_module._UNTRACKED


def _ticked(
    states: NDArray[np.void],
    rng: NDArray[np.uint32],
    stats: NDArray[np.void],
    action: int,
) -> tuple[NDArray[np.void], NDArray[np.uint32], int, list[NDArray[np.void]]]:
    """Play a sleep a tick at a time on copies; return them, its ticks and its samples."""
    states, rng, stats = states.copy(), rng.copy(), stats.copy()
    state, stat, ticked = (
        env_state(states, 0),
        env_stats(stats, 0),
        Rules(collapse_sleep=False),
    )
    _, done = play_numba(state, rng, stat, action, ticked)
    ticks, samples = 1, list[NDArray[np.void]]()
    while not done and (state.is_sleeping or state.is_resting):
        if ticks % extract_module.SLEEP_STRIDE == 0:
            samples.append(states.copy())
        _, done = play_numba(state, rng, stat, Action.NOOP.value, ticked)
        ticks += 1
    return states, rng, ticks, samples


def test_each_sleep_played_a_tick_at_a_time_ends_on_the_collapsed_steps_state(
    records: list[Record],
    ghosts: list[Ghost],
) -> None:
    compared = 0
    for record, ghost in zip(records, ghosts, strict=True):
        states, rng = replay.reset_world(record.receipt.world_seed)
        stats = new_stats(1)
        state, stat = env_state(states, 0), env_stats(stats, 0)
        sleeps: list[tuple[int, int]] = []
        creatures: list[list[tuple[int, int, int, int, int]]] = []
        changes: list[tuple[int, int, int, int, int, int]] = []
        samples = 0
        for t in range(ghost.decisions):
            action = int(record.actions[t])
            if action != Action.SLEEP.value:
                play_numba(state, rng, stat, action, Rules())
                continue
            before = states.copy()
            ticked, ticked_rng, ticks, ticked_samples = _ticked(
                states,
                rng,
                stats,
                action,
            )
            play_numba(state, rng, stat, action, Rules())
            if int(stat.last_ticks) <= 1:
                continue
            # The proof: the same bytes, the same stream, the same ticks.
            assert replay.fnv1a_numba(ticked.view(np.uint8)) == replay.fnv1a_numba(
                states.view(np.uint8),
            )
            assert ticked.tobytes() == states.tobytes()
            assert np.array_equal(ticked_rng, rng)
            assert ticks == int(stat.last_ticks)
            compared += 1
            sleeps.append((t, ticks))
            level = int(env_state(before, 0).player_level)
            for sample in ticked_samples:
                now = env_state(sample, 0)
                creatures.append(
                    [
                        _creature(now, klass, level, slot)
                        for klass in range(len(_CLASSES))
                        for slot in range(3)
                        if _mobs(now, klass)[level].mask[slot]
                    ],
                )
                for floor in sorted({0, level}):
                    differs = np.logical_or(
                        np.not_equal(
                            typed(sample["map"], np.uint8)[0, floor, ...],
                            typed(before["map"], np.uint8)[0, floor, ...],
                        ),
                        np.not_equal(
                            typed(sample["item_map"], np.uint8)[0, floor, ...],
                            typed(before["item_map"], np.uint8)[0, floor, ...],
                        ),
                    )
                    rows, cols = np.nonzero(differs)
                    changes += [
                        (
                            samples,
                            floor,
                            rows.item(k),
                            cols.item(k),
                            int(now.map[floor, rows.item(k), cols.item(k)]),
                            int(now.item_map[floor, rows.item(k), cols.item(k)]),
                        )
                        for k in range(len(rows))
                    ]
                samples += 1
        assert int_rows(ghost.sleeps) == sleeps
        assert (
            len(ghost.sleep_samples) - 1
            == samples
            == sum((ticks - 1) // 4 for _, ticks in sleeps)
        )
        bounds = [ghost.sleep_samples.item(i) for i in range(len(ghost.sleep_samples))]
        decoded = [
            sorted(decode_samples(ghost.sleep_creatures[a:b])[0]) if b > a else []
            for a, b in itertools.pairwise(bounds)
        ]
        assert decoded == [sorted(listed) for listed in creatures]
        assert sorted(int_rows(ghost.sleep_changes)) == sorted(changes)
    assert compared >= 3


def test_a_sleep_whose_ticks_end_elsewhere_fails_the_extraction(
    records: list[Record],
) -> None:
    record = records[0]
    states, rng = replay.reset_world(record.receipt.world_seed)
    actions = np.ascontiguousarray(record.actions.numpy(), dtype=np.uint8)
    trace = extract_module._new_trace(
        states,
        decisions=len(actions),
        stride=4,
        sleeps=int(np.count_nonzero(np.equal(actions, Action.SLEEP.value))),
    )
    stats = new_stats(1)
    copy = extract_module._Copy(
        states=states.copy(),
        rng=rng.copy(),
        stats=stats.copy(),
    )
    # Ticks played by the capture's own rules collapse each sleep into one.
    status = extract_module._trace_numba(
        states, states.view(np.uint8), rng, stats, actions, (Rules(), Rules()), trace, copy,
        0, 4, 49,
    )  # fmt: skip
    assert status == extract_module._SLEPT_ELSEWHERE


def _replay(record: Record, *, stop: int) -> _Replayed:
    """Play ``record``'s first ``stop`` decisions on the game, reading each State."""
    states, rng = replay.reset_world(record.receipt.world_seed)
    state, stats, rules = env_state(states, 0), new_stats(1), Rules()
    path = np.zeros((stop + 1, 4), np.int64)
    creatures: list[list[tuple[int, int, int, int, int]]] = []
    maps: dict[int, tuple[NDArray[np.uint8], NDArray[np.uint8]]] = {}
    active: list[int] = []
    for t in range(stop + 1):
        level = int(state.player_level)
        path[t] = level, *state.player_position, state.player_direction
        if t % 4 == 0 and t < stop:
            creatures.append(
                [
                    _creature(state, klass, level, slot)
                    for klass in range(len(_CLASSES))
                    for slot in range(3)
                    if _mobs(state, klass)[level].mask[slot]
                ],
            )
        if t % 64 == 0 or t == stop:
            maps[t] = (
                np.array(state.map, dtype=np.uint8),
                np.array(state.item_map, dtype=np.uint8),
            )
        if t < stop:
            before = states.copy()
            play_numba(state, rng, env_stats(stats, 0), int(record.actions[t]), rules)
            active.append(
                _flags(before, states, path=path[: t + 1], last=t == stop - 1),
            )
    return _Replayed(
        path=path,
        creatures=creatures,
        maps=maps,
        final=states.copy(),
        active=active,
    )


def _flags(
    before: NDArray[np.void],
    after: NDArray[np.void],
    *,
    path: NDArray[np.int64],
    last: bool,
) -> int:
    """Return a decision's activity flags, read from the States around it."""
    old, new = env_state(before, 0), env_state(after, 0)
    place = (
        int(new.player_level),
        int(new.player_position[0]),
        int(new.player_position[1]),
    )
    visited = {
        (path.item(t, 0), path.item(t, 1), path.item(t, 2)) for t in range(len(path))
    }
    return (
        (1 if place not in visited else 0)
        | (0 if np.array_equal(old.map, new.map) else 2)
        | (0 if np.array_equal(old.item_map, new.item_map) else 2)
        | (0 if np.array_equal(old.achievements, new.achievements) else 4)
        | (8 if old.player_level != new.player_level else 0)
        | (16 if last else 0)
        | (32 if before["inventory"].tobytes() != after["inventory"].tobytes() else 0)
        | (64 if _fought(old, new) else 0)
    )


def _fought(before: EnvState, after: EnvState) -> bool:
    """Whether a creature on the player's floor lost health, or kills or the boss advanced."""
    if (
        after.monsters_killed.sum() + after.boss_progress
        > before.monsters_killed.sum() + before.boss_progress
    ):
        return True
    level = int(before.player_level)
    return any(
        _mobs(before, klass)[level].mask[slot]
        and _mobs(after, klass)[level].health[slot]
        < _mobs(before, klass)[level].health[slot]
        for klass in range(3)
        for slot in range(3)
    )


def _creature(
    state: EnvState,
    klass: int,
    level: int,
    slot: int,
) -> tuple[int, int, int, int, int]:
    """Return one live slot as (class, species, row, column, facing)."""
    mobs = _mobs(state, klass)[level]
    row, col = mobs.position[slot, 0], mobs.position[slot, 1]
    directions = _directions(state, klass)
    if directions is None:
        return klass, mobs.type_id[slot], row, col, 0
    facing = _FACINGS[directions[level, slot, 0], directions[level, slot, 1]]
    return klass, mobs.type_id[slot], row, col, facing


def _mobs(state: EnvState, klass: int) -> Sequence[Mobs]:
    """Return class ``klass``'s slots on every floor, ``_CLASSES[klass]``."""
    return (
        state.melee_mobs,
        state.passive_mobs,
        state.ranged_mobs,
        state.mob_projectiles,
        state.player_projectiles,
    )[klass]


def _directions(state: EnvState, klass: int) -> Array3[int] | None:
    """Return a projectile class's direction field, or None for a creature class."""
    return {3: state.mob_projectile_dirs, 4: state.player_projectile_directions}.get(
        klass,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
