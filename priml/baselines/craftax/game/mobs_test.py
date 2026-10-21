"""Tests for the creature phases' spawn-tile selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba.core.dispatcher import Dispatcher

import numpy as np
import pytest

from priml.baselines.craftax.eager import eager
from priml.baselines.craftax.game.mobs import (
    BOSS_RING,
    HOSTILE_RING,
    PASSIVE_RING,
    column_span_numba,
    count_spawn_cells_numba,
    select_spawn_cell_numba,
    spawn_row_cells_numba,
)
from priml.baselines.craftax.game.state import (
    MAP_SIZE,
    MOB_DESPAWN_DISTANCE,
    env_state,
    new_states,
)
from priml.baselines.craftax.lib.arrays import typed


if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


_RINGS = {
    PASSIVE_RING: (9, 196),
    HOSTILE_RING: (81, 196),
    BOSS_RING: (-1, 37),
}
"""Each ring's ``(min_exclusive, max_exclusive)``, as ``spawn_mobs`` passes them in C."""


def _c_list(
    states: NDArray[np.void],
    level: int,
    band: tuple[int, int],
    terrain: str,
) -> list[tuple[int, int]]:
    """``collect_spawn_cells`` as ``craftax.h`` writes it: a scan of every candidate."""
    state = env_state(states, 0)
    pr, pc = state.player_position
    limit = MOB_DESPAWN_DISTANCE - 1
    cells: list[tuple[int, int]] = []
    for row in range(max(pr - limit, 0), min(pr + limit, MAP_SIZE - 1) + 1):
        bits = typed(states[terrain], np.uint64).item(0, level, row) & ~int(
            state.mob_bits[level, row],
        )
        for col in range(
            max(pc - limit, 0),
            min(pc + limit, MAP_SIZE - 1) + 1,
        ):
            distance2 = (row - pr) * (row - pr) + (col - pc) * (col - pc)
            if bits >> col & 1 and band[0] < distance2 < band[1]:
                cells.append((row, col))
    return cells


@pytest.mark.parametrize("ring", sorted(_RINGS))
@pytest.mark.parametrize(
    ("boss", "water_only", "terrain"),
    [
        (False, False, "spawn_land"),
        (False, True, "spawn_water"),
        (True, False, "spawn_grave"),
    ],
)
def test_the_count_and_every_index_select_what_the_c_list_holds(
    ring: int,
    boss: bool,
    water_only: bool,
    terrain: str,
) -> None:
    draws = np.random.default_rng(ring * 4 + 2 * boss + water_only)
    states = new_states(1)
    level = 3
    checked = 0
    for _ in range(2):
        states[0]["player_position"] = draws.integers(0, MAP_SIZE, size=2)
        for name in ("spawn_land", "spawn_water", "spawn_grave", "mob_bits"):
            states[0][name][level] = draws.integers(
                0,
                1 << MAP_SIZE,
                size=MAP_SIZE,
                dtype=np.uint64,
            )
        expected = _c_list(states, level, _RINGS[ring], terrain)
        state = env_state(states, 0)
        assert count_spawn_cells_numba(state, level, ring, boss, water_only) == len(
            expected,
        )
        for index, cell in enumerate(expected):
            assert (
                select_spawn_cell_numba(state, level, ring, boss, water_only, index)
                == cell
            )
        checked += len(expected)
    assert checked > 0


def test_interpreted_spawn_branches_keep_ring_terrain_occupancy_and_order() -> None:
    assert not isinstance(count_spawn_cells_numba, Dispatcher)
    assert not isinstance(select_spawn_cell_numba, Dispatcher)
    assert not isinstance(spawn_row_cells_numba, Dispatcher)
    assert not isinstance(column_span_numba, Dispatcher)
    assert column_span_numba(-2, 1) == np.uint64(0b11)
    assert column_span_numba(
        MAP_SIZE - 2,
        MAP_SIZE + 1,
    ) == np.uint64(3 << (MAP_SIZE - 2))
    assert column_span_numba(2, 1) == 0

    states = new_states(1)
    states[0]["player_position"] = (32, 2)
    state = env_state(states, 0)
    level = 3
    states[0]["spawn_land"][level, 32] = sum(1 << col for col in (0, 6, 15, 16))
    states[0]["mob_bits"][level, 32] = np.uint64(1 << 6)
    assert (
        count_spawn_cells_numba(
            state,
            level,
            PASSIVE_RING,
            False,
            False,
        )
        == 1
    )
    assert select_spawn_cell_numba(
        state,
        level,
        PASSIVE_RING,
        False,
        False,
        0,
    ) == (32, 15)
    assert select_spawn_cell_numba(
        state,
        level,
        PASSIVE_RING,
        False,
        False,
        1,
    ) == (-1, -1)

    states[0]["player_position"] = (32, 32)
    states[0]["spawn_water"][level, 31] = np.uint64((1 << 42) | (1 << 44) | (1 << 45))
    states[0]["spawn_water"][level, 32] = np.uint64(1 << 45)
    states[0]["mob_bits"][level, 31] = np.uint64(1 << 42)
    assert (
        count_spawn_cells_numba(
            state,
            level,
            HOSTILE_RING,
            False,
            True,
        )
        == 3
    )
    assert select_spawn_cell_numba(
        state,
        level,
        HOSTILE_RING,
        False,
        True,
        0,
    ) == (31, 44)
    assert select_spawn_cell_numba(
        state,
        level,
        HOSTILE_RING,
        False,
        True,
        1,
    ) == (31, 45)
    assert select_spawn_cell_numba(
        state,
        level,
        HOSTILE_RING,
        False,
        True,
        2,
    ) == (32, 45)

    states[0]["spawn_grave"][level, 32] = np.uint64((1 << 38) | (1 << 39))
    assert (
        count_spawn_cells_numba(
            state,
            level,
            BOSS_RING,
            True,
            False,
        )
        == 1
    )
    assert select_spawn_cell_numba(
        state,
        level,
        BOSS_RING,
        True,
        False,
        0,
    ) == (32, 38)
    assert (
        spawn_row_cells_numba(
            state,
            level,
            BOSS_RING,
            True,
            False,
            39,
        )
        == 0
    )


@pytest.fixture(autouse=True)
def _eager_kernels() -> Iterator[None]:
    """Keep ordinary spawn tests on the Python bodies, without compiling Numba."""
    with eager():
        yield


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
