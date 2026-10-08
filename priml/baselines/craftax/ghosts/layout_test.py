"""Check the site's binary layouts round-trip and the reference decoder's rules."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from priml.baselines.craftax.game.state import (
    MAP_SIZE,
    NUM_LEVELS,
    Action,
)
from priml.baselines.craftax.ghosts.layout import (
    ACHIEVEMENT_COLUMNS,
    ESCAPE_COLUMNS,
    FLOOR_CHANGE,
    KEEP_COLUMNS,
    KEEP_COUNT_COLUMNS,
    MAP_COLUMNS,
    MOVED,
    Events,
    Sleeps,
    World,
    decode_path,
    decode_samples,
    displayed_sleep,
    encode_events,
    encode_keeps,
    encode_sleep,
    encode_table,
    encode_window,
    interaction_targets,
    maps_before,
    read_events,
    read_keeps,
    read_sleep,
    read_table,
    read_window,
    read_world,
    world_bytes,
)


def _world() -> World:
    grids = np.arange(3 * NUM_LEVELS * MAP_SIZE * MAP_SIZE, dtype=np.int64) % 37
    grids = grids.astype(np.uint8).reshape(3, NUM_LEVELS, MAP_SIZE, MAP_SIZE)
    # ``World``'s down and up ladders: per floor, a (row, column).
    ladders = np.arange(2 * NUM_LEVELS * 2, dtype=np.int64).reshape(2, NUM_LEVELS, 2)
    return World(
        block=grids[0, ...],
        item=grids[1, ...],
        light=grids[2, ...],
        down_ladders=ladders[0, ...],
        up_ladders=ladders[1, ...],
    )


def test_a_table_pads_each_column_to_four_bytes_and_reads_back() -> None:
    rows = np.array([[70_000, 1, 2, 3, 4, 5], [9, 8, 7, 6, 5, 4], [1, 0, 0, 0, 0, 0]])
    table = encode_table(rows, columns=MAP_COLUMNS)
    assert len(table) == 4 + 12 + 5 * 4
    back, end = read_table(table + b"tail", offset=0, columns=MAP_COLUMNS)
    np.testing.assert_array_equal(back, rows)
    assert end == len(table)


def test_events_hold_three_tables_and_nothing_after() -> None:
    events = Events(
        map=np.array([[3, 0, 1, 2, 7, 0]]),
        achievements=np.array([[3, 9], [5, 10]]),
        escapes=np.zeros((0, len(ESCAPE_COLUMNS)), np.int64),
    )
    data = encode_events(events)
    back = read_events(data)
    np.testing.assert_array_equal(back.map, events.map)
    np.testing.assert_array_equal(back.achievements, events.achievements)
    assert back.escapes.shape == (0, len(ESCAPE_COLUMNS))
    assert len(ACHIEVEMENT_COLUMNS) == back.achievements.shape[1]
    with pytest.raises(
        ValueError,
        match=r"^events\.bin has 4 bytes after its tables\.$",
    ):
        read_events(data + bytes(4))


def test_a_time_map_keeps_each_episodes_runs_and_refuses_a_mismatch() -> None:
    runs = [np.array([[0, 3], [5, 9]]), np.zeros((0, 2), np.int64), np.array([[2, 4]])]
    data = encode_keeps(runs)
    back = read_keeps(data)
    assert len(back) == len(runs)
    for got, want in zip(back, runs, strict=True):
        np.testing.assert_array_equal(got, want)
    mismatch = r"^A time map's run counts do not match its runs\.$"
    with pytest.raises(ValueError, match=mismatch):
        read_keeps(data + bytes(4))
    # Counts of 2, 0 and 2 for three runs.
    miscounted = encode_table(
        np.array([[2], [0], [2]]),
        columns=KEEP_COUNT_COLUMNS,
    ) + encode_table(np.concatenate(runs), columns=KEEP_COLUMNS)
    with pytest.raises(ValueError, match=mismatch):
        read_keeps(miscounted)


def test_a_window_keeps_each_run_and_its_empty_ones() -> None:
    runs = [b"\x01\x23\x04\x05", b"", b"\x00\x00"]
    data = encode_window(runs)
    assert read_window(data) == runs
    with pytest.raises(ValueError, match="offsets"):
        read_window(data + b"\x00")


def test_sleeps_read_back_per_episode_with_their_samples_and_changes() -> None:
    # Episode 0 sleeps twice: 9 ticks (2 samples) and 3 (none); episode 1 never;
    # episode 2 once for 6 ticks (1 sample, one changed tile).
    episodes = [
        Sleeps(
            sleeps=np.array([[5, 9], [20, 3]]),
            samples=np.array([0, 4, 5]),
            creatures=bytes([1, 0x10, 3, 4, 0]),
            changes=np.zeros((0, 6), np.int64),
        ),
        Sleeps(
            sleeps=np.zeros((0, 2), np.int64),
            samples=np.zeros(1, np.int64),
            creatures=b"",
            changes=np.zeros((0, 6), np.int64),
        ),
        Sleeps(
            sleeps=np.array([[7, 6]]),
            samples=np.array([0, 1]),
            creatures=bytes([0]),
            changes=np.array([[0, 0, 4, 5, 6, 0]]),
        ),
    ]
    back = read_sleep(encode_sleep(episodes), stride=4)
    for got, want in zip(back, episodes, strict=True):
        np.testing.assert_array_equal(got.sleeps, want.sleeps)
        np.testing.assert_array_equal(got.samples, want.samples)
        assert got.creatures == want.creatures
        np.testing.assert_array_equal(got.changes, want.changes)
    with pytest.raises(ValueError, match="do not add up"):
        read_sleep(encode_sleep(episodes), stride=2)


def test_the_sleep_view_shows_each_sleeps_samples_after_its_decision() -> None:
    runs, sleeps = np.array([[0, 3], [5, 7]]), np.array([[1, 9], [6, 4]])
    steps = displayed_sleep(runs, sleeps, stride=4)
    assert steps.tolist() == [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0], [5, 0], [6, 0]]


def test_samples_decode_class_species_tile_and_facing() -> None:
    run = bytes([2, 0x13, 4, 5, 0, 0x41, 6, 7, 3, 0])
    assert decode_samples(run) == [[(1, 3, 4, 5, 0), (4, 1, 6, 7, 3)], []]


def test_the_world_file_reads_back_and_refuses_another_length() -> None:
    world = _world()
    back = read_world(world_bytes(world))
    for name in ("block", "item", "light", "down_ladders", "up_ladders"):
        np.testing.assert_array_equal(
            cast("np.ndarray", getattr(back, name)),
            cast("np.ndarray", getattr(world, name)),
        )
    with pytest.raises(ValueError, match="bytes"):
        read_world(world_bytes(world)[1:])


def test_the_path_follows_moves_turns_ladders_and_escapes() -> None:
    world = _world()
    players = bytes(
        [
            Action.LEFT | MOVED,
            Action.UP,
            Action.DO,
            Action.DESCEND | FLOOR_CHANGE,
            Action.RIGHT | MOVED,
            Action.ASCEND | FLOOR_CHANGE,
            Action.NOOP,
        ],
    )
    escapes = np.array([[6, 2, 10, 11, 2]])
    path = decode_path(players, escapes=escapes, world=world, start=(0, 24, 24, 3))
    up, down = world.up_ladders, world.down_ladders
    assert path.tolist() == [
        [0, 24, 24, 3],
        [0, 24, 23, 1],
        [0, 24, 23, 3],
        [0, 24, 23, 3],
        [1, *up[1], 3],
        [1, up[1][0], up[1][1] + 1, 2],
        [0, *down[0], 2],
        [2, 10, 11, 2],
    ]
    assert interaction_targets(players, path=path) == [(2, 0, 23, 23)]


def test_maps_before_a_decision_apply_only_earlier_events() -> None:
    world = _world()
    events = np.array([[0, 1, 2, 3, 30, 1], [4, 1, 2, 3, 31, 0]])
    block, item = maps_before(world, map_events=events, decision=4)
    assert (block[1, 2, 3], item[1, 2, 3]) == (30, 1)
    block, item = maps_before(world, map_events=events, decision=5)
    assert (block[1, 2, 3], item[1, 2, 3]) == (31, 0)
    assert np.count_nonzero(np.not_equal(block, world.block)) == 1


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
