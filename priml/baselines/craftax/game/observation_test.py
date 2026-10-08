"""Tests for the observations: the symbolic layout against JAX's renderer, and compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba.core.types as nbtypes
import numpy as np
import pytest

from priml.baselines.craftax.game import step
from priml.baselines.craftax.game.observation import (
    _write_mob_obs_numba,
    compute_observations_numba,
    compute_symbolic_observations_numba,
)
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    INVENTORY_OBS_SIZE,
    MAP_SIZE,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_MOB_TYPES,
    OBS_COLS,
    OBS_ROWS,
    OBS_SIZE,
    STATE_DTYPE,
    SYMBOLIC_OBS_SIZE,
    SYMBOLIC_TILE_CHANNELS,
    VISIBLE_LIGHT_THRESHOLD,
    Action,
    env_state,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.testing import kernel_llvm
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    build_pool_numba,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import EnvState


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


@pytest.mark.compute_large_fixture
def test_the_symbolic_view_is_jaxs_one_hot_and_the_scalars_match_the_packed_ones() -> (
    None
):
    pool = new_states(4)
    build_pool_numba(
        pool,
        pool.view(np.uint8).reshape(4, STATE_DTYPE.itemsize),
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
        0,
    )
    batch = step.Batch(
        pool.copy(),
        np.arange(4, dtype=np.uint32),
        new_stats(4),
        np.zeros((4, 1), dtype=np.float32),  # ``Batch.actions``: ``[num_envs, 1]``.
        np.zeros((4, OBS_SIZE), dtype=np.float32),
        np.ones((4, ATN_DIM), dtype=np.uint8),
        np.zeros(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    draws = np.random.default_rng(0)
    packed = np.zeros(OBS_SIZE, dtype=np.float32)
    symbolic = np.zeros(SYMBOLIC_OBS_SIZE, dtype=np.float32)
    packed_mask = np.zeros(ATN_DIM, dtype=np.uint8)
    symbolic_mask = np.zeros(ATN_DIM, dtype=np.uint8)
    checked_mobs = 0
    for _ in range(300):
        batch.actions[:, 0] = draws.integers(Action.LEFT, Action.DO + 1, size=4)
        step.step_range_numba(batch, pool, 0, 4)
        for i in range(4):
            state = env_state(batch.states, i)
            compute_observations_numba(state, packed, packed_mask)
            compute_symbolic_observations_numba(state, symbolic, symbolic_mask)
            map_size = symbolic.size - INVENTORY_OBS_SIZE
            np.testing.assert_array_equal(
                symbolic[:map_size], _reference_map(state),
            )  # fmt: skip
            np.testing.assert_array_equal(
                symbolic[map_size:], packed[-INVENTORY_OBS_SIZE :],
            )  # fmt: skip
            np.testing.assert_array_equal(symbolic_mask, packed_mask)
            checked_mobs += int(symbolic[:map_size].reshape(99, -1)[:, 42:82].sum())
    assert checked_mobs > 0


@pytest.mark.compute_large_fixture
def test_the_observation_passes_its_slot_counts_as_plain_int64s() -> None:
    # A literal slot count compiles a copy of _write_mob_obs per value, which
    # LLVM then leaves out of line in the step (measured on the Xeon).
    states = new_states(1)
    compute_observations_numba(
        env_state(states, 0),
        np.zeros(OBS_SIZE, dtype=np.float32),
        np.zeros(ATN_DIM, dtype=np.uint8),
    )
    kernel_llvm(compute_observations_numba)
    assert {signature[3] for signature in _write_mob_obs_numba.signatures} == {
        nbtypes.int64,
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
