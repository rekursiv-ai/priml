"""Tests for the eager kernels: their stand-ins, the record views, and the tiny world."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, cast

import ctypes

from numba.core.dispatcher import Dispatcher

import numpy as np
import pytest

from priml.baselines.craftax import eager as eager_module
from priml.baselines.craftax.eager import (
    LADDER_DOWN,
    LADDER_UP,
    eager,
    scripted,
    tiny_world,
)
from priml.baselines.craftax.game import jit, rules, step
from priml.baselines.craftax.game.state import (
    DEFAULT_MAX_TIMESTEPS,
    MAP_SIZE,
    Action,
    BlockType,
    ItemType,
    env_state,
    env_stats,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, play_numba
from priml.baselines.craftax.lib.arrays import typed
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.capture.seeds import splitmix64


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import EnvState


def test_kernels_run_as_python_inside_and_compiled_again_after() -> None:
    kernel = replay.fnv1a_numba
    assert isinstance(rules.daylight_numba, Dispatcher)
    with eager():
        assert not isinstance(rules.daylight_numba, Dispatcher)
        assert not isinstance(replay.fnv1a_numba, Dispatcher)
        # A kernel imported by name into another module goes eager there too.
        assert not isinstance(play_numba, Dispatcher)
    assert replay.fnv1a_numba is kernel


def test_a_kernel_wraps_its_integers_on_any_thread_as_compiled_code_does() -> None:
    # numpy's error state is the thread's own: a rollout's buffer thread starts
    # with none set, where an overflow warning would raise.
    # A thread's later calls, after its first returned, wrap alike.
    top = np.uint64(2**64 - 1)
    with eager(), ThreadPoolExecutor(1) as pool:
        draws = [pool.submit(replay._splitmix64_numba, top).result() for _ in "ab"]
        draws.append(replay._splitmix64_numba(top))
    assert [(int(s), int(d)) for s, d in draws] == [splitmix64(2**64 - 1)] * 3


@pytest.mark.parametrize(
    "data",
    [b"", b"\x00", b"ghost", b"\x00\x00gh\x00\x00\x00ost\x00", bytes(range(256))],
)
def test_the_python_hash_is_the_kernels(data: bytes) -> None:
    array = np.frombuffer(data, np.uint8)
    with np.errstate(over="ignore"):
        expected = replay.fnv1a_numba.py_func(array)
    with eager():
        assert replay.fnv1a_numba(array) == expected
    assert eager_module._fnv1a(array) == expected


def test_the_bit_counts_are_the_builtins() -> None:
    with eager():
        assert [jit.popcount(np.uint64(v)) for v in (0, 1, 0b1011, 2**64 - 1)] == [
            0,
            1,
            3,
            64,
        ]
        assert [jit.trailing_zeros(np.uint64(v)) for v in (1, 8, 2**63, 0)] == [
            0,
            3,
            63,
            64,
        ]
        assert jit.prefetch(object(), 7) is None


def test_cosf_sinf_and_powf_are_the_platform_libms_float32_calls() -> None:
    libm = ctypes.CDLL(None)
    for name in ("cosf", "sinf", "powf"):
        getattr(libm, name).restype = ctypes.c_float
    x, y = ctypes.c_float(0.3), ctypes.c_float(3.0)
    expected = [
        np.float32(cast("float", libm.cosf(x))),
        np.float32(cast("float", libm.sinf(x))),
        np.float32(cast("float", libm.powf(x, y))),
    ]
    with eager():
        assert jit.cosf is not np.cos
        at, power = np.float32(x.value), np.float32(y.value)
        got = [jit.cosf(at), jit.sinf(at), jit.powf(at, power)]
    assert got == expected
    assert all(type(value) is np.float32 for value in got)
    assert jit.cosf is np.cos


def test_the_daylight_table_reads_the_light_of_each_timestep() -> None:
    with eager():
        table = rules.daylight_table()
        assert table.shape == (rules.DAYLIGHT_TIMESTEPS,)
        for timestep in (0, 1, 299, DEFAULT_MAX_TIMESTEPS):
            assert table[np.int64(timestep)] == rules.daylight_numba(timestep)


def test_a_structured_argument_reaches_the_kernel_as_records() -> None:
    with eager(world=tiny_world):
        states, rng = replay.reset_world(5)
        # ``_run_numba`` reads ``states[0]`` as a record; a plain structured
        # array would give it a ``numpy.void`` without fields as attributes.
        actions = np.array([Action.LEFT.value], np.uint8)
        hashes = np.array([replay.fnv1a_numba(states.view(np.uint8)), 0], np.uint64)
        status, _ = replay._run_numba(
            states, states.view(np.uint8), rng, new_stats(1), actions, hashes,
            0, 1, True, False, *replay._arrays(replay._empty_frames(0)), Rules(),
        )  # fmt: skip
    assert status == len(hashes)
    assert env_state(states, 0).player_direction == Action.LEFT.value


def test_the_tiny_world_is_a_reset_world_of_grass() -> None:
    states = new_states(1)
    state = env_state(states, 0)
    tiny_world(state, np.array([50], np.uint32), timestep=7)
    centre = MAP_SIZE // 2
    assert (state.player_level, *state.player_position) == (0, centre, centre)
    assert (state.player_direction, state.timestep) == (Action.UP.value, 7)
    assert state.player_health == np.float32(9.0)
    assert state.map[0, centre - 1, centre] == BlockType.TREE
    assert state.map[0, 0, 50 % MAP_SIZE] == BlockType.STONE
    assert np.equal(typed(states["map"], np.uint8)[0, 1:2], BlockType.GRASS).all()
    assert not typed(states["map"], np.uint8)[0, 2:].any()
    for floor in range(len(state.map)):
        assert tuple(state.down_ladders[floor]) == LADDER_DOWN
        assert tuple(state.up_ladders[floor]) == LADDER_UP
        assert state.item_map[(floor, *LADDER_DOWN)] == ItemType.LADDER_DOWN
        assert state.item_map[(floor, *LADDER_UP)] == ItemType.LADDER_UP


@pytest.mark.parametrize("cell", [(0, 0, 2), (0, 0, 3), (0, 23, 24), (4, 47, 47)])
def test_the_tiny_worlds_spawn_bits_are_what_a_block_write_leaves(
    cell: tuple[int, int, int],
) -> None:
    states = new_states(1)
    state = env_state(states, 0)
    tiny_world(state, np.array([2], np.uint32))
    bitsets = states[["spawn_land", "spawn_grave", "spawn_water"]].copy()
    with eager():
        rules.refresh_spawn_cell_numba(state, *cell)
    assert states[["spawn_land", "spawn_grave", "spawn_water"]] == bitsets


def test_a_world_stand_in_replaces_world_generation_and_its_seed_shows() -> None:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 2)):
        first, rng = replay.reset_world(1)
        second, _ = replay.reset_world(2)
        episode = replay.record(world_seed=1, sampling_seed=3, max_decisions=4)
        assert replay.verify(episode) == replay.MATCHED
    assert rng.tolist() == [1]
    assert first.tobytes() != second.tobytes()
    assert env_state(first, 0).timestep == DEFAULT_MAX_TIMESTEPS - 2
    assert len(episode.actions) == 2


class _Pair(NamedTuple):
    states: object
    size: int


def test_a_records_fields_read_and_write_through_to_its_array() -> None:
    states = new_states(1)
    viewed = eager_module._viewed(env_state(states, 0))
    assert isinstance(viewed, eager_module._Fields)
    # The record view reads as the State a kernel reads.
    fields = cast("EnvState", viewed)
    fields.player_level = 3
    blocks = typed(np.asarray(fields.map), np.uint8)
    blocks[1, 2, 3] = BlockType.WATER
    assert np.asarray(fields.map) is blocks
    assert (fields.player_level, env_state(states, 0).player_level) == (3, 3)
    assert typed(states["map"], np.uint8)[0, 1, 2, 3] == BlockType.WATER
    plain = _Pair(states=1, size=2)
    assert eager_module._viewed(plain) is plain


def test_a_structured_arrays_records_read_once_and_write_back() -> None:
    states = new_states(2)
    pair = eager_module._viewed(_Pair(states=states, size=2))
    assert isinstance(pair, _Pair)
    records = pair.states
    assert isinstance(records, eager_module._Records)
    first = records[0]
    assert isinstance(first, eager_module._Fields)
    assert records[np.int64(0)] is first
    cast("EnvState", first).timestep = 5
    records[1] = first
    # A field of records reads as records too: a floor's creatures.
    cows = cast("EnvState", first).passive_mobs[3]
    cows.health[1] = np.float32(2.0)
    assert typed(states["passive_mobs"]["health"], np.float32)[0, 3, 1] == 2.0
    assert typed(states["timestep"], np.int32).tolist() == [5, 5]
    assert len(records) == 2
    assert records.dtype == states.dtype
    assert len(cast("np.ndarray", records[0:1])) == 1


def test_kernels_handed_one_array_in_a_block_share_its_records_till_it_ends() -> None:
    states = new_states(1)
    with eager():
        records = eager_module._viewed(states)
        assert eager_module._viewed(states) is records
    assert eager_module._viewed(states) is not records


def test_a_scripted_record_replays_to_its_hashes_and_stats_are_their_record() -> None:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 2)):
        record = scripted([Action.RIGHT, Action.NOOP], world_seed=4, sampling_seed=9)
        assert replay.verify(record) == replay.MATCHED
        stats = env_stats(new_stats(1), 0)
        assert step._training(stats) is stats
    assert (record.receipt.world_seed, record.receipt.sampling_seed) == (4, 9)
    assert record.actions.tolist() == [Action.RIGHT, Action.NOOP]
    assert len(record.hashes) == 2
    assert not record.truncated


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
