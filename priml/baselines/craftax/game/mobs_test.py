"""Tests for the creature phases' spawn-tile selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.baselines.craftax.game.mobs import (
    BOSS_RING,
    HOSTILE_RING,
    PASSIVE_RING,
    count_spawn_cells_numba,
    select_spawn_cell_numba,
)
from priml.baselines.craftax.game.state import (
    MAP_SIZE,
    MOB_DESPAWN_DISTANCE,
    env_state,
    new_states,
)
from priml.baselines.craftax.lib.arrays import typed


if TYPE_CHECKING:
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


@pytest.mark.compute_large_fixture
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
    for _ in range(64):
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
