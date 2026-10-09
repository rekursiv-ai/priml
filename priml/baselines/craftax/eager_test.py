"""Tests for the eager kernels: their stand-ins, the record views, and the tiny world."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast

import sys

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
from priml.baselines.craftax.lib.arrays import ints, typed
from priml.baselines.craftax.scripts import mint_goldens
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.capture import step as capture_step
from priml.baselines.craftax.world_model.capture.seeds import splitmix64


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import EnvState, Records


class _Cell(Protocol):
    """A record of :data:`_CELL`."""

    x: int


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
        assert replay.fnv1a_numba is capture_step.fnv1a_numba is eager_module._fnv1a
        assert replay.fnv1a_numba(array) == expected
    assert eager_module._fnv1a(array) == expected


def test_the_python_hash_refuses_elements_wider_than_a_byte() -> None:
    # The kernel hashes one byte an element; the bytes of wider ones hash apart.
    with pytest.raises(TypeError, match="uint8"):
        eager_module._fnv1a(np.zeros(2, np.uint16))


class _Counted(NamedTuple):
    """A named tuple of arrays, as a ``Batch`` is."""

    states: np.ndarray
    counts: np.ndarray


@jit.jit
def _uncompiled(x: int) -> int:
    return x + 1


@jit.jit
def _calls_uncompiled(x: int) -> int:
    return _uncompiled(x) * 2


@jit.jit
def _tick(pair: _Counted, row: int) -> int:
    """Advance one record's clock through its attribute; count the call."""
    states = cast("Records[EnvState]", pair.states)
    states[row].timestep += 1
    pair.counts[0] += 1
    return int(states[row].timestep)


@jit.jit
def _first_timestep(states: Records[EnvState]) -> int:
    return _timestep_of(states[0])


@jit.jit
def _timestep_of(state: EnvState) -> int:
    return int(state.timestep)


def _stand_in_target() -> str:
    return "compiled"


def test_a_kernel_and_its_callees_run_as_python_then_compiled_again() -> None:
    kernel = _calls_uncompiled
    with eager():
        assert _calls_uncompiled(3) == 8
        assert _calls_uncompiled is not kernel
    assert _calls_uncompiled is kernel
    # Neither compiled: the callee ran as Python too.
    assert not kernel.signatures
    assert not _uncompiled.signatures


def test_a_kernel_a_test_replaced_comes_back_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test's replacement is undone first, then the Python kernel it replaced."""
    kernel = _uncompiled
    with monkeypatch.context() as patch, eager(monkeypatch=patch):
        patch.setattr(sys.modules[__name__], "_uncompiled", _stand_in_target)
        assert _uncompiled is _stand_in_target
    assert _uncompiled is kernel


def test_a_kernel_reads_plain_arrays_as_records_and_writes_through() -> None:
    pair = _Counted(new_states(2), np.zeros(1, dtype=np.int64))
    with eager():
        assert _tick(pair, 1) == 1
        assert _tick(pair, 1) == 2
    assert typed(pair.states["timestep"], np.int32).tolist() == [0, 2]
    assert pair.counts.tolist() == [2]
    assert not _tick.signatures


def test_a_stand_in_by_name_is_handed_records_not_their_views() -> None:
    held: list[type] = []

    def timestep_of(state: object) -> int:
        held.append(type(state))
        return 7

    with eager(_timestep_of=timestep_of):
        assert _first_timestep(new_states(1)) == 7
    assert held == [np.record]


def test_the_intrinsics_and_the_libm_stand_in_and_a_name_adds_one() -> None:
    # jit_test's anchor 64, where glibc's and macOS's libms agree.
    angle = np.float32(
        np.array([0x3FC9_D9B5], dtype=np.uint32).view(np.float32).item(0),
    )
    expected = mint_goldens.light_levels_libm([7, 100_000])
    stats = env_stats(new_stats(1), 0)
    with eager(_stand_in_target=lambda: "eager"):
        assert _stand_in_target() == "eager"
        assert jit.popcount(np.uint64(0b1011)) == 3
        assert jit.popcount(np.uint64(2**64 - 1)) == 64
        assert jit.trailing_zeros(np.uint64(0b1000)) == 3
        assert jit.trailing_zeros(np.uint64(0)) == 64
        assert jit.prefetch(object(), 7) is None
        assert step._training(stats) is stats
        table = rules.daylight_table()
        assert table.shape == (rules.DAYLIGHT_TIMESTEPS,)
        light = np.array([table[7], table[100_000]])
        # numpy's float32 calls may round apart from the libm's, but not at the anchor.
        assert {jit.cosf, jit.sinf, jit.powf}.isdisjoint({np.cos, np.sin, np.power})
        libm = np.array(
            [
                jit.cosf(angle),
                jit.sinf(angle),
                jit.powf(np.float32(1.25), np.float32(3.0)),
            ],
        )
    assert _stand_in_target() == "compiled"
    assert jit.cosf is np.cos
    assert light.dtype == libm.dtype == np.float32
    assert light.view(np.uint32).tolist() == expected.view(np.uint32).tolist()
    assert libm.view(np.uint32).tolist() == [
        0xBBC9_DA0A,
        0x3F7F_FEC2,
        np.float32(1.953125).view(np.uint32),
    ]


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


def test_a_block_inside_another_plays_its_own_world_then_the_outer_ones() -> None:
    late = DEFAULT_MAX_TIMESTEPS - 2
    clocks: list[int] = []
    with eager(world=tiny_world):
        for _ in range(2):
            with eager(world=partial(tiny_world, timestep=late)):
                # The hash's stand-in, not the outer stand-in's kernel run as Python.
                assert replay.fnv1a_numba is eager_module._fnv1a
                clocks.append(_clock())
            assert not isinstance(rules.daylight_numba, Dispatcher)
            clocks.append(_clock())
    assert isinstance(rules.daylight_numba, Dispatcher)
    assert clocks == [late, 0, late, 0]


def _clock() -> int:
    """Return the clock of world 1's State at its reset."""
    states, _ = replay.reset_world(1)
    return int(env_state(states, 0).timestep)


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
    # Python code a kernel hands its record reads and writes fields by name.
    viewed["timestep"] = 9
    assert (viewed["timestep"], env_state(states, 0).timestep) == (9, 9)
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
    # Any other index is the array's own: a record of a grid of them.
    grid = eager_module._viewed(new_states(4).reshape(2, 2))
    assert isinstance(grid, eager_module._Records)
    assert type(grid[1, 0]) is np.record
    # A plain tuple's plain arrays read as recarrays, its records as records.
    counts = np.zeros(2)
    held = eager_module._viewed((states, counts))
    assert type(held) is tuple
    records, plain = cast("tuple[object, np.ndarray]", held)
    assert isinstance(records, eager_module._Records)
    assert type(plain) is np.recarray
    assert np.shares_memory(plain, counts)


def test_kernels_handed_one_array_in_a_block_share_its_records_till_it_ends() -> None:
    states = new_states(1)
    with eager():
        records = eager_module._viewed(states)
        assert eager_module._viewed(states) is records
    assert eager_module._viewed(states) is not records


_CELL: Final = np.dtype([("x", np.int64)])
"""A record of one field, for views made and freed call by call."""


class _Two(NamedTuple):
    """Two arrays of bytes, each viewed as records in turn, as a ``Batch``'s are."""

    first: np.ndarray
    second: np.ndarray


@jit.jit
def _stamp(cells: Records[_Cell], value: int) -> None:
    cells[0].x = value


@jit.jit
def _stamp_both(two: _Two) -> None:
    _stamp(cast("Records[_Cell]", two.first.view(_CELL)), 1)
    _stamp(cast("Records[_Cell]", two.second.view(_CELL)), 2)


def test_a_temporary_view_handed_to_a_kernel_is_never_taken_for_another() -> None:
    # The first view is freed once its call returns, and the second may take
    # its ``id``, as the step's views of a batch's observations do.
    two = _Two(np.zeros(8, np.uint8), np.zeros(8, np.uint8))
    with eager():
        for _ in range(3):
            _stamp_both(two)
    assert [ints(t.view(np.int64))[0] for t in (two.first, two.second)] == [1, 2]


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
