"""Tests for the observations: the symbolic layout against JAX's renderer, and its slots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.baselines.craftax.game import observation
from priml.baselines.craftax.game.observation import (
    compute_observations_numba,
    compute_symbolic_observations_numba,
)
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    INVENTORY_OBS_SIZE,
    MAP_SIZE,
    MAX_RANGED_MOBS,
    MOB_SLOTS,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_MOB_TYPES,
    OBS_COLS,
    OBS_ROWS,
    OBS_SIZE,
    SYMBOLIC_OBS_SIZE,
    SYMBOLIC_TILE_CHANNELS,
    VISIBLE_LIGHT_THRESHOLD,
    env_state,
    new_states,
)
from priml.baselines.craftax.game.testing import eager_kernels, small_worlds
from priml.baselines.craftax.lib.arrays import typed


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import Array1, EnvState, Mobs


_CREATURES = (
    "melee_mobs",
    "passive_mobs",
    "ranged_mobs",
    "mob_projectiles",
    "player_projectiles",
)


@pytest.fixture(autouse=True)
def eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the views as Python: a compile costs seconds, one view milliseconds."""
    eager_kernels(monkeypatch)


# The C's light rule (a uint8 light above 12, D10) and its creature rule (live, in view
# and lit; D9) stand in for JAX's float light and scatter.
def _reference_map(state: EnvState) -> NDArray[np.float32]:
    """``render_craftax_symbolic``'s view (``renderer.py:9-122``) on a port state."""
    rows, cols = OBS_ROWS, OBS_COLS
    view = np.zeros((rows, cols, SYMBOLIC_TILE_CHANNELS), dtype=np.float32)
    level = int(state.player_level)
    pr, pc = (int(v) for v in state.player_position)
    lit = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            row, col = pr + r - rows // 2, pc + c - cols // 2
            if not (0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE):
                continue
            if state.light_map[level, row, col] <= VISIBLE_LIGHT_THRESHOLD:
                continue
            lit[r, c] = True
            view[r, c, state.map[level, row, col]] = 1
            view[
                r,
                c,
                NUM_BLOCK_TYPES + state.item_map[level, row, col],
            ] = 1
            view[r, c, -1] = 1
    tables = (
        state.melee_mobs,
        state.passive_mobs,
        state.ranged_mobs,
        state.mob_projectiles,
        state.player_projectiles,
    )
    for mob_class, table in enumerate(tables):
        mobs = table[level]
        for slot in range(len(mobs.mask)):
            r = mobs.position[slot, 0] - pr + rows // 2
            c = mobs.position[slot, 1] - pc + cols // 2
            if mobs.mask[slot] and 0 <= r < rows and 0 <= c < cols and lit[r, c]:
                channel = (
                    NUM_BLOCK_TYPES
                    + NUM_ITEM_TYPES
                    + mob_class * NUM_MOB_TYPES
                    + mobs.type_id[slot]
                )
                view[r, c, channel] = 1
    return view.reshape(-1)


@pytest.mark.parametrize(
    ("seed", "row", "col", "level"),
    [(0, 2, 3, 0), (1, MAP_SIZE - 3, MAP_SIZE - 2, 2), (2, 24, 25, 8)],
    ids=["top-left", "bottom-right", "middle"],
)
def test_the_symbolic_view_is_jaxs_one_hot_and_the_scalars_match_the_packed_ones(
    seed: int,
    row: int,
    col: int,
    level: int,
) -> None:
    """Random blocks, items, light and creatures around a player, clipped by the map's edge."""
    states = _random_world(seed, row=row, col=col, level=level)
    state = env_state(states, 0)
    packed = np.zeros(OBS_SIZE, dtype=np.float32)
    symbolic = np.zeros(SYMBOLIC_OBS_SIZE, dtype=np.float32)
    packed_mask = np.zeros(ATN_DIM, dtype=np.uint8)
    symbolic_mask = np.zeros(ATN_DIM, dtype=np.uint8)
    compute_observations_numba(state, packed, packed_mask)
    compute_symbolic_observations_numba(state, symbolic, symbolic_mask)
    map_size = symbolic.size - INVENTORY_OBS_SIZE
    np.testing.assert_array_equal(symbolic[:map_size], _reference_map(state))
    np.testing.assert_array_equal(
        symbolic[map_size:], packed[-INVENTORY_OBS_SIZE :],
    )  # fmt: skip
    np.testing.assert_array_equal(symbolic_mask, packed_mask)
    tiles = symbolic[:map_size].reshape(OBS_ROWS * OBS_COLS, -1)
    # Lit tiles, dark ones, and live creatures in view: the cases the view tells apart.
    assert 0 < tiles[:, -1].sum() < OBS_ROWS * OBS_COLS
    assert tiles[:, NUM_BLOCK_TYPES + NUM_ITEM_TYPES : -1].sum() > 0


def test_the_observation_passes_its_slot_counts_as_plain_int64s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal slot count compiles a copy of the creature writer per value.

    LLVM then left those copies out of line in the step (measured on the
    Xeon). Numba types a module constant as a literal, so each count goes
    through ``unliteral``, an ``np.int64`` as the Python runs.
    """
    counts: list[object] = []
    write = observation._write_mob_obs_numba

    def record(
        obs: Array1[np.float32],
        state: EnvState,
        mobs: Mobs,
        slots: int,
        mob_class: int,
    ) -> None:
        counts.append(slots)
        write(obs, state, mobs, slots, mob_class)

    monkeypatch.setattr(observation, "_write_mob_obs_numba", record)
    compute_observations_numba(
        env_state(small_worlds(1), 0),
        np.zeros(OBS_SIZE, dtype=np.float32),
        np.zeros(ATN_DIM, dtype=np.uint8),
    )
    assert [type(count) for count in counts] == [np.int64] * len(_CREATURES)
    assert counts == [3, 3, 2, 3, 3]


# Every block, item and light level, inventory counts below 10, and each creature slot
# live or not, of a random species, within 7 tiles of the player: some in view, some
# past its edge, some on dark tiles.
def _random_world(seed: int, *, row: int, col: int, level: int) -> NDArray[np.void]:
    """Return a world whose floor ``level`` is random within 7 tiles of the player."""
    draws = np.random.default_rng(seed)
    states = new_states(1)
    shape = (MAP_SIZE, MAP_SIZE)
    typed(states["map"], np.uint8)[0, level] = draws.integers(0, NUM_BLOCK_TYPES, shape)
    items = typed(states["item_map"], np.uint8)
    items[0, level] = draws.integers(0, NUM_ITEM_TYPES, shape)
    light = typed(states["light_map"], np.uint8)
    light[0, level] = draws.integers(0, 256, shape)
    # Both sides of the threshold, next to the player.
    light[0, level, row, col] = VISIBLE_LIGHT_THRESHOLD
    light[0, level, row, col - 1] = VISIBLE_LIGHT_THRESHOLD + 1
    state = env_state(states, 0)
    state.player_level = level
    state.player_position[0] = row
    state.player_position[1] = col
    player = np.array([row, col])
    for name in _CREATURES:
        offsets = draws.integers(-7, 8, (MOB_SLOTS, 2))
        positions = typed(states[name]["position"], np.int32)
        positions[0, level] = np.clip(offsets + player, 0, MAP_SIZE - 1)
        typed(states[name]["mask"], np.uint8)[0, level] = draws.integers(
            0,
            2,
            MOB_SLOTS,
        )
        species = typed(states[name]["type_id"], np.int32)
        species[0, level] = draws.integers(0, NUM_MOB_TYPES, MOB_SLOTS)
    # A ranged creature has two slots; the third is never live.
    typed(states["ranged_mobs"]["mask"], np.uint8)[0, level, MAX_RANGED_MOBS:] = 0
    for name in ("wood", "stone", "coal", "iron", "diamond", "sapling", "torches"):
        typed(states["inventory"][name], np.int32)[0] = draws.integers(0, 10)
    state.player_health = np.float32(draws.integers(1, 10))
    return states


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
