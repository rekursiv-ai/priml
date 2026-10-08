"""Tests for the shared mechanics and for the daylight a tick advances.

The light level is checked bit for bit against the platform libm, live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llvmlite import ir

import numpy as np
import pytest

from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.rules import (
    _DAYLIGHT_ARRAY,
    DAYLIGHT_TIMESTEPS,
    _daylight_table_signature,
    daylight_numba,
    daylight_table,
    refresh_spawn_cell_numba,
    set_block_numba,
    tick_numba,
)
from priml.baselines.craftax.game.state import (
    Action,
    BlockType,
    env_state,
    new_states,
)
from priml.baselines.craftax.game.testing import (
    FMA_MNEMONIC,
    ir_builder,
    kernel_assembly,
    kernel_inspection,
)
from priml.baselines.craftax.scripts import mint_goldens


if TYPE_CHECKING:
    import numba.core.types as nbtypes


@pytest.mark.compute_large_fixture
def test_set_block_keeps_every_spawn_bitset_in_step_with_the_map() -> None:
    states = new_states(1)
    state = env_state(states, 0)
    set_block_numba(state, 2, 5, 7, BlockType.GRASS)
    assert states[0]["map"][2, 5, 7] == BlockType.GRASS
    assert states[0]["spawn_land"][2, 5] == 1 << 7
    set_block_numba(state, 2, 5, 7, BlockType.WATER)
    assert states[0]["spawn_land"][2, 5] == 0
    assert states[0]["spawn_water"][2, 5] == 1 << 7
    set_block_numba(state, 2, 5, 7, BlockType.GRAVE3)
    assert states[0]["spawn_water"][2, 5] == 0
    assert states[0]["spawn_grave"][2, 5] == 1 << 7
    set_block_numba(state, 2, 5, 7, BlockType.STONE)
    assert states[0]["spawn_grave"][2, 5] == 0


@pytest.mark.compute_large_fixture
def test_refresh_touches_only_its_own_bit() -> None:
    states = new_states(1)
    states[0]["spawn_land"][0, 3] = np.uint64(0xFFFF_FFFF_FFFF_FFFF)
    states[0]["map"][0, 3, 10] = BlockType.STONE
    refresh_spawn_cell_numba(env_state(states, 0), 0, 3, 10)
    assert states[0]["spawn_land"][0, 3] == np.uint64(
        0xFFFF_FFFF_FFFF_FFFF,
    ) & ~np.uint64(1 << 10)


@pytest.mark.compute_large_fixture
def test_daylight_is_periodic_with_a_bright_afternoon() -> None:
    assert daylight_numba(0) == daylight_numba(300)
    assert daylight_numba(0) < daylight_numba(60)
    assert 0.0 <= daylight_numba(150) <= 1.0
    assert isinstance(daylight_numba(0), float)


@pytest.mark.compute_large_fixture
def test_daylight_is_the_platform_libms_at_every_timestep_an_episode_reaches() -> None:
    """The C calls the platform libm, and so must the kernel, in the same order.

    Live on every host, since macOS's libm and glibc's round 401 of these
    timesteps apart; ``mint_goldens_test.py`` holds the reference to PufferLib's.
    """
    expected = mint_goldens.light_levels_libm(range(DAYLIGHT_TIMESTEPS))
    actual = np.fromiter(
        (daylight_numba(t) for t in range(DAYLIGHT_TIMESTEPS)),
        np.float32,
    )
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.compute_large_fixture
def test_daylight_emits_no_fma_and_no_float64() -> None:
    daylight_numba(1)
    assembly, body = kernel_inspection(daylight_numba)
    assert not FMA_MNEMONIC.search(assembly)
    assert "double" not in body
    assert "fpext" not in body


@pytest.mark.compute_large_fixture
def test_daylight_calls_libm_rather_than_expanding_the_cube() -> None:
    # LLVM may rewrite pow(x, 3) as x*x*x, which rounds differently from
    # the C's powf; the light level must stay a libm call.
    daylight_numba(1)
    assembly = kernel_assembly(daylight_numba)
    assert "powf" in assembly
    assert "cosf" in assembly


@jit
def _embedded_daylight_table() -> np.ndarray:
    return daylight_table().copy()


@pytest.mark.compute_large_fixture
def test_a_kernel_reads_daylight_at_every_timestep_a_default_episode_reaches() -> None:
    expected = np.fromiter(
        (daylight_numba(t) for t in range(DAYLIGHT_TIMESTEPS)),
        np.float32,
    )
    actual = _embedded_daylight_table()
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.compute_large_fixture
def test_a_tick_lights_the_world_as_daylight_does_in_and_past_the_table() -> None:
    states = new_states(1)
    for timestep in (
        1,
        300,
        301,
        54_321,
        DAYLIGHT_TIMESTEPS - 1,
        DAYLIGHT_TIMESTEPS,
        250_000,
    ):
        states["timestep"][0] = timestep - 1
        tick_numba(env_state(states, 0), int(Action.NOOP), 1 << 30)
        light = env_state(states, 0).light_level
        assert light.view(np.uint32) == np.float32(daylight_numba(timestep)).view(
            np.uint32,
        ), timestep


def test_the_daylight_table_lowers_to_a_constant_of_every_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The codegen bakes ``daylight_numba`` at each timestep into a read-only constant.

    ``daylight_numba`` stands in as the timestep itself, so the lowering is
    checked without compiling the kernel; its values are the libm test's.
    """
    monkeypatch.setattr(
        "priml.baselines.craftax.game.rules.daylight_numba",
        np.float32,
    )
    signature, emit = _daylight_table_signature(None)
    assert signature == _DAYLIGHT_ARRAY()
    builder, args = ir_builder()
    constants: list[np.ndarray] = []
    context = _RecordingConstants(constants)
    assert emit(context, builder, signature, args) is context.emitted
    (table,) = constants
    assert table.dtype == np.float32
    assert np.array_equal(table, np.arange(DAYLIGHT_TIMESTEPS, dtype=np.float32))


class _RecordingConstants:
    """A lowering context that keeps each constant array it is asked to emit."""

    def __init__(self, constants: list[np.ndarray]) -> None:
        self.constants = constants
        self.emitted = ir.Constant(ir.IntType(8), 0)

    def make_constant_array(
        self,
        builder: ir.IRBuilder,
        typ: nbtypes.Array,
        ary: np.ndarray,
    ) -> ir.Value:
        """Keep ``ary`` and stand in for the global holding it."""
        del builder
        assert typ is _DAYLIGHT_ARRAY
        self.constants.append(ary)
        return self.emitted


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
