"""Check that a ghost decodes to the game its record replays, and refuses what is not one.

Every test plays the game's kernels as Python (``eager``) on a tiny world of
grass. The scripted episode cuts the tree above the start, turns against a
penned cow and hits it, walks two tiles to the down ladder, descends, ascends
and places a stone as the clock runs out: every activity flag, both floor
changes, map changes and a creature, in eight decisions whose ghost is known
in advance.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import dataclasses

import numpy as np
import pytest

from priml.baselines.craftax.eager import eager, scripted, tiny_world
from priml.baselines.craftax.game import rules
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    DEFAULT_MAX_TIMESTEPS,
    MAP_SIZE,
    NUM_LEVELS,
    Achievement,
    Action,
    BlockType,
    env_state,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules
from priml.baselines.craftax.ghosts import extract as extract_module
from priml.baselines.craftax.ghosts.extract import (
    ACHIEVED,
    ENDED,
    FLOOR_CHANGED,
    FOUGHT,
    HELD,
    MAP_CHANGE,
    NEW_TILE,
    Ghost,
    extract,
)
from priml.baselines.craftax.ghosts.layout import (
    FLOOR_CHANGE,
    MOVED,
    World,
    decode_path,
    decode_samples,
    maps_before,
)
from priml.baselines.craftax.lib.arrays import int_rows, ints, typed
from priml.baselines.craftax.world_model import replay


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import Array1, EnvState
    from priml.baselines.craftax.world_model.archive import Record


_WORLD: Final = 3
_CENTRE: Final = MAP_SIZE // 2
_COW: Final = (_CENTRE, _CENTRE - 1)
"""Where the penned cow stands: left of the start, stones on its other three sides."""

_PLANT: Final = (20, 20)
"""Where the sleep's world grows a plant that ripens during the sleep."""

_SCRIPT: Final = (
    Action.DO,  # Cut the tree above the start.
    Action.LEFT,  # The cow blocks the way: a turn.
    Action.DO,  # Hit the cow.
    Action.RIGHT,
    Action.RIGHT,  # Onto the down ladder.
    Action.DESCEND,
    Action.ASCEND,
    Action.PLACE_STONE,  # The last decision: the clock runs out.
)


def _world(state: EnvState, rng: Array1[np.uint32], *, timestep: int) -> None:
    """Fill the tiny world, a penned cow and a stone in the bag; no creature spawns."""
    tiny_world(state, rng, timestep=timestep)
    typed(np.asarray(state.spawn_land), np.uint64)[...] = 0
    row, col = _COW
    for dr, dc in ((-1, 0), (1, 0), (0, -1)):
        rules.set_block_numba(state, 0, row + dr, col + dc, BlockType.STONE)
    cows = state.passive_mobs[0]
    cows.mask[0], cows.health[0], cows.type_id[0] = 1, np.float32(3.0), 0
    cows.position[0, 0], cows.position[0, 1] = row, col
    rules.set_mob_bit_numba(state, 0, row, col, True)
    state.inventory.stone = 1


def _sleepy(state: EnvState, rng: Array1[np.uint32]) -> None:
    """Fill the scripted world, one energy short, with a plant 7 ticks from ripe."""
    _world(state, rng, timestep=DEFAULT_MAX_TIMESTEPS - 13)
    state.player_energy = 8
    rules.set_block_numba(state, 0, *_PLANT, BlockType.PLANT)
    state.growing_plants_mask[0] = 1
    state.growing_plants_pos[0, 0], state.growing_plants_pos[0, 1] = _PLANT
    state.growing_plants_age[0] = 593


_SCRIPTED: Final = partial(_world, timestep=DEFAULT_MAX_TIMESTEPS - len(_SCRIPT))
"""The scripted episode's world: its clock runs out at its last decision."""


@pytest.fixture(scope="module")
def record() -> Record:
    with eager(world=_SCRIPTED):
        return scripted([int(a) for a in _SCRIPT], world_seed=_WORLD)


@pytest.fixture(scope="module")
def ghost(record: Record) -> Ghost:
    with eager(world=_SCRIPTED):
        return extract(record, ordinal=7)


def test_each_player_byte_is_the_action_and_whether_it_moved_or_took_a_ladder(
    ghost: Ghost,
) -> None:
    assert list(ghost.players) == [
        Action.DO,
        Action.LEFT,
        Action.DO,
        Action.RIGHT | MOVED,
        Action.RIGHT | MOVED,
        Action.DESCEND | FLOOR_CHANGE,
        Action.ASCEND | FLOOR_CHANGE,
        Action.PLACE_STONE,
    ]
    assert len(ghost.events.escapes) == 0


def test_decoded_players_are_the_places_and_facings_played(ghost: Ghost) -> None:
    with eager(world=_SCRIPTED):
        states, _ = replay.reset_world(_WORLD)
    path = decode_path(
        ghost.players,
        escapes=ghost.events.escapes,
        world=_layout_world(states),
        start=(0, _CENTRE, _CENTRE, Action.UP),
    )
    up, left, right = Action.UP, Action.LEFT, Action.RIGHT
    assert int_rows(path) == [
        (0, _CENTRE, _CENTRE, up),
        (0, _CENTRE, _CENTRE, up),
        (0, _CENTRE, _CENTRE, left),
        (0, _CENTRE, _CENTRE, left),
        (0, _CENTRE, _CENTRE + 1, right),
        (0, _CENTRE, _CENTRE + 2, right),
        (1, _CENTRE, _CENTRE - 2, right),
        (0, _CENTRE, _CENTRE + 2, right),
        (0, _CENTRE, _CENTRE + 2, right),
    ]
    assert int_rows(path[-1:]) == [ghost.end]


def test_map_events_rebuild_the_cut_tree_and_the_placed_stone(ghost: Ghost) -> None:
    tree, stone = (0, _CENTRE - 1, _CENTRE), (0, _CENTRE, _CENTRE + 3)
    assert int_rows(ghost.events.map) == [
        (0, *tree, BlockType.GRASS, 0),
        (7, *stone, BlockType.STONE, 0),
    ]
    with eager(world=_SCRIPTED):
        states, _ = replay.reset_world(_WORLD)
    world = _layout_world(states)
    block, item = maps_before(world, map_events=ghost.events.map, decision=7)
    changed = np.argwhere(np.not_equal(block, world.block))
    assert int_rows(changed) == [tree]
    assert block[tree] == BlockType.GRASS
    np.testing.assert_array_equal(item, world.item)
    block, _ = maps_before(world, map_events=ghost.events.map, decision=8)
    assert int_rows(np.argwhere(np.not_equal(block, world.block))) == [tree, stone]


def test_creature_samples_are_every_fourth_decisions_creatures(ghost: Ghost) -> None:
    cow = (1, 0, *_COW, 0)
    assert decode_samples(ghost.creatures) == [[cow], [cow]]
    assert ints(ghost.samples) == [0, 5, 10]


def test_the_end_is_the_last_state_and_floors_are_first_reached_there(
    ghost: Ghost,
) -> None:
    assert (ghost.outcome, ghost.decisions) == ("timeout", len(_SCRIPT))
    assert ghost.end == (0, _CENTRE, _CENTRE + 2, Action.RIGHT)
    assert ghost.floor_first == (0, 6, *[-1] * (NUM_LEVELS - 2))
    assert (ghost.ordinal, ghost.world_seed) == (7, _WORLD)


def test_achievements_are_the_unlocked_ones_and_the_return_their_reward(
    ghost: Ghost,
) -> None:
    unlocked = [
        (0, Achievement.COLLECT_WOOD),
        # The bag's stone counts as collected on the first tick.
        (0, Achievement.COLLECT_STONE),
        (5, Achievement.ENTER_DUNGEON),
        (7, Achievement.PLACE_STONE),
    ]
    assert int_rows(ghost.events.achievements) == unlocked
    rewards = int(ACHIEVEMENT_REWARD_MAP[[a for _, a in unlocked]].sum())
    assert ghost.achievement_return == rewards > 0


def test_activity_flags_are_each_decisions_events(ghost: Ghost) -> None:
    assert list(ghost.active) == [
        MAP_CHANGE | ACHIEVED | HELD,
        0,
        FOUGHT,
        NEW_TILE,
        NEW_TILE,
        NEW_TILE | ACHIEVED | FLOOR_CHANGED,
        FLOOR_CHANGED,
        MAP_CHANGE | ACHIEVED | ENDED | HELD,
    ]


def test_a_win_ends_the_ghost_at_its_first_unlock(
    record: Record,
    ghost: Ghost,
) -> None:
    with eager(world=_SCRIPTED):
        won = extract(record, ordinal=7, win=Achievement.ENTER_DUNGEON)
    assert (won.outcome, won.decisions) == ("win", 6)
    assert won.players == ghost.players[:6]
    np.testing.assert_array_equal(won.events.map, ghost.events.map[:1])
    assert ints(won.events.achievements[:, 1])[-1] == Achievement.ENTER_DUNGEON
    assert won.end == (1, _CENTRE, _CENTRE - 2, Action.RIGHT)


@pytest.mark.parametrize("damage", ["action", "hash"])
def test_a_record_off_its_hashes_is_refused(record: Record, damage: str) -> None:
    if damage == "action":
        # A move turns the player even when blocked, so a last move the other
        # way leaves a final State the last hash does not match.
        actions = record.actions.clone()
        actions[-1] = Action.LEFT
        damaged = dataclasses.replace(record, actions=actions)
    else:
        hashes = record.hashes.clone()
        hashes[-1] ^= 1
        damaged = dataclasses.replace(record, hashes=hashes)
    with eager(world=_SCRIPTED), pytest.raises(ValueError, match="replay status"):
        extract(damaged, ordinal=1)


def test_a_branch_is_refused(record: Record) -> None:
    branch = dataclasses.replace(record, origin=bytes(replay.SNAPSHOT_BYTES))
    with pytest.raises(ValueError, match="origin"):
        extract(branch, ordinal=1)


def test_a_truncated_record_stops_where_it_was_cut(ghost: Ghost) -> None:
    cut = 4
    with eager(world=_SCRIPTED):
        truncated = scripted(
            [int(a) for a in _SCRIPT[:cut]],
            world_seed=_WORLD,
            truncated=True,
        )
        stopped = extract(truncated, ordinal=1)
        with pytest.raises(ValueError, match="replay status"):
            extract(dataclasses.replace(truncated, truncated=False), ordinal=1)
    assert (stopped.outcome, stopped.decisions) == ("truncated", cut)
    assert stopped.players == ghost.players[:cut]


def test_a_hit_and_a_kill_are_fights_and_a_despawn_is_not() -> None:
    with eager(world=tiny_world):
        states, _ = replay.reset_world(_WORLD)
        state = env_state(states, 0)
        row, col = _CENTRE - 1, _CENTRE
        cows = state.passive_mobs[0]
        cows.mask[0], cows.health[0], cows.type_id[0] = 1, np.float32(3.0), 0
        cows.position[0, 0], cows.position[0, 1] = row, col
        rules.set_mob_bit_numba(state, 0, row, col, True)
        trace = extract_module._new_trace(states, decisions=1, stride=4)
        hits: list[bool] = []
        while cows.mask[0]:
            score = extract_module._snapshot_fight_numba(state, trace, 0)
            rules.damage_mob_at_numba(state, 0, row, col, np.float32(1.0), True, True)
            hits.append(extract_module._fought_numba(state, trace, 0, score))
        cows.mask[0], cows.health[0] = 1, np.float32(3.0)
        score = extract_module._snapshot_fight_numba(state, trace, 0)
        cows.mask[0] = 0
        despawned = extract_module._fought_numba(state, trace, 0, score)
        state.monsters_killed[3] += 1
        counted = extract_module._fought_numba(state, trace, 0, score)
    assert hits == [True, True, True]
    assert (despawned, counted) == (False, True)


def test_a_sample_gives_each_projectile_its_facing_and_refuses_a_diagonal() -> None:
    with eager(world=tiny_world):
        states, _ = replay.reset_world(_WORLD)
        state = env_state(states, 0)
        arrows, fireballs = state.player_projectiles[0], state.mob_projectiles[0]
        directions = state.player_projectile_directions
        for slot, (direction, (row, col)) in enumerate(
            zip(((0, -1), (1, 0), (0, 1)), ((3, 4), (5, 6), (7, 8)), strict=True),
        ):
            arrows.mask[slot], arrows.type_id[slot] = 1, 0
            arrows.position[slot, 0], arrows.position[slot, 1] = row, col
            directions[0, slot, 0], directions[0, slot, 1] = direction
        fireballs.mask[1], fireballs.type_id[1] = 1, 2
        fireballs.position[1, 0], fireballs.position[1, 1] = 9, 10
        state.mob_projectile_dirs[0, 1, 0], state.mob_projectile_dirs[0, 1, 1] = -1, 0
        trace = extract_module._new_trace(states, decisions=1, stride=4)
        sampled = extract_module._sample_numba(state, trace, 0)
        directions[0, 2, 0], directions[0, 2, 1] = 1, 1
        diagonal = extract_module._sample_numba(state, trace, 0)
    assert sampled
    assert not diagonal
    run = trace.creatures[: trace.samples[1]].tobytes()
    assert decode_samples(run) == [
        [
            (3, 2, 9, 10, Action.UP.value),
            (4, 0, 3, 4, Action.LEFT.value),
            (4, 0, 5, 6, Action.DOWN.value),
            (4, 0, 7, 8, Action.RIGHT.value),
        ],
    ]


def test_a_move_the_byte_cannot_say_is_an_escape_row() -> None:
    with eager(world=tiny_world):
        states, _ = replay.reset_world(_WORLD)
        state = env_state(states, 0)
        trace = extract_module._new_trace(states, decisions=2, stride=4)
        # A step right that lands two tiles over: no byte decodes to it.
        state.player_position[1] += 2
        state.player_direction = Action.RIGHT
        extract_module._move_numba(
            state,
            trace,
            1,
            Action.RIGHT,
            0,
            _CENTRE,
            _CENTRE,
            Action.UP,
        )
    assert int_rows(trace.escapes[: trace.counts[2]]) == [
        (1, 0, _CENTRE, _CENTRE + 2, Action.RIGHT),
    ]
    assert trace.players[1] == Action.RIGHT


def test_a_change_on_an_untracked_floor_fails_the_extraction() -> None:
    with eager(world=tiny_world):
        states, rng = replay.reset_world(_WORLD)
        actions = np.zeros(3, np.uint8)
        trace = extract_module._new_trace(states, decisions=len(actions), stride=4)
        trace.block[5, 0, 0] ^= 1
        stats = new_stats(1)
        copy = extract_module._Copy(
            states=states.copy(),
            rng=rng.copy(),
            stats=stats.copy(),
        )
        status = extract_module._trace_numba(
            states, states.view(np.uint8), rng, stats, actions,
            (Rules(), Rules(collapse_sleep=False)), trace, copy, 0, 4, 49,
        )  # fmt: skip
    assert status == extract_module._UNTRACKED


def test_a_sleep_played_a_tick_at_a_time_is_sampled_every_fourth_tick() -> None:
    # One energy short, the player sleeps until its fatigue passes -10, 11
    # ticks, and wakes on the 12th; the clock then runs out at the NOOP.
    with eager(world=_sleepy):
        record = scripted([Action.SLEEP, Action.NOOP], world_seed=_WORLD)
        ghost = extract(record, ordinal=0)
    assert ghost.outcome == "timeout"
    assert int_rows(ghost.sleeps) == [(0, 12)]
    assert ints(ghost.sleep_samples) == [0, 5, 10]
    cow = (1, 0, *_COW, 0)
    assert decode_samples(ghost.sleep_creatures) == [[cow], [cow]]
    # The plant ripens at the 7th tick: the 8th tick's sample shows it.
    assert int_rows(ghost.sleep_changes) == [(1, 0, *_PLANT, BlockType.RIPE_PLANT, 0)]


def test_a_sleep_whose_ticks_end_elsewhere_fails_the_extraction() -> None:
    with eager(world=_sleepy):
        record = scripted([Action.SLEEP, Action.NOOP], world_seed=_WORLD)
        states, rng = replay.reset_world(_WORLD)
        actions = np.ascontiguousarray(record.actions.numpy(), dtype=np.uint8)
        trace = extract_module._new_trace(
            states,
            decisions=len(actions),
            stride=4,
            sleeps=1,
        )
        stats = new_stats(1)
        copy = extract_module._Copy(
            states=states.copy(),
            rng=rng.copy(),
            stats=stats.copy(),
        )
        # Ticks played by the capture's own rules collapse the sleep into one.
        status = extract_module._trace_numba(
            states, states.view(np.uint8), rng, stats, actions, (Rules(), Rules()),
            trace, copy, 0, 4, 49,
        )  # fmt: skip
    assert status == extract_module._SLEPT_ELSEWHERE


def _layout_world(states: NDArray[np.void]) -> World:
    """Return the decoder's view of a reset world."""
    return World(
        block=typed(states["map"], np.uint8)[0, ...],
        item=typed(states["item_map"], np.uint8)[0, ...],
        light=typed(states["light_map"], np.uint8)[0, ...],
        down_ladders=typed(states["down_ladders"], np.int32)[0, ...].astype(np.int64),
        up_ladders=typed(states["up_ladders"], np.int32)[0, ...].astype(np.int64),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
