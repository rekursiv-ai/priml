"""Tests that the port's ``rand_r`` stream is glibc's, draw for draw.

The kernels run as their Python source (``testing.eager_kernels``): their
arithmetic is uint32 masked in uint64 and single float32 roundings, which
Python's numpy scalars compute bit for bit as the compiled kernels do, and
``env_test``'s goldens hold the compiled stream. The jitted helper calls the
game's scalar ``rand_r_numba`` one draw at a time: that kernel is the unit
under test, and the game draws that way because each environment's draws are
conditional on its own state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.rng import (
    F32_PER_DRAW,
    choice_valid_numba,
    choose_weighted_numba,
    choose_weighted_pair_numba,
    rand_r_numba,
    rng_f32_at_numba,
    rng_f32_numba,
    rng_int_numba,
)
from priml.baselines.craftax.game.testing import eager_kernels
from priml.baselines.craftax.scripts import mint_goldens


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import Array1, Array2


EDGE_SEED = 30_835_466
"""The smallest state whose next draw (2147483596) is within 64 of 2^31.

Found by scanning seeds with ``mint_goldens.glibc_rand_r``; the edge test asserts
the draw so a wrong constant cannot pass silently.
"""


@pytest.fixture(autouse=True)
def eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the draws as Python: a compile costs seconds, a draw microseconds."""
    eager_kernels(monkeypatch)


@jit
def _streams(seeds: Array1[np.uint32], out: Array2[np.int32]) -> None:
    rng = np.empty(1, dtype=np.uint32)
    for row in range(seeds.shape[0]):
        rng[0] = seeds[row]
        for i in range(out.shape[1]):
            out[row, i] = rand_r_numba(rng)


def _rng(seed: int) -> NDArray[np.uint32]:
    return np.array([seed], dtype=np.uint32)


def test_rand_r_draws_pufferlibs_first_draws_from_73() -> None:
    """PufferLib's own glibc draws, so a constant the transcription shares still fails."""
    expected = mint_goldens.SEED_73_DRAWS
    actual = np.empty((1, len(expected)), dtype=np.int32)
    _streams(_rng(73), actual)
    assert actual.tolist() == [list(expected)]


def test_rand_r_reproduces_glibcs_streams_across_the_seed_range() -> None:
    """128 draws from each of 16 seeds, every one compared with glibc's.

    The transcription stands for glibc (``mint_goldens_test.py``). The seeds
    are the first environments', the edge's and the top of the uint32 range,
    where a missing mask or a narrowed product would show first.
    """
    seeds = np.array(
        [
            *range(8),
            2**16,
            2**24 + 1,
            EDGE_SEED,
            2**31 - 1,
            2**31,
            0xDEAD_BEEF,
            2**32 - 3,
            2**32 - 1,
        ],
        dtype=np.uint32,
    )
    expected = mint_goldens.glibc_rand_r(seeds, 128)
    actual = np.empty_like(expected)
    _streams(seeds, actual)
    assert np.array_equal(actual, expected)


def test_rng_f32_is_the_draw_times_two_to_the_minus_31() -> None:
    rng = _rng(73)
    draw = mint_goldens.glibc_rand_r(np.array([73], np.uint32), 1).item(0, 0)
    assert rng_f32_numba(rng) == np.float32(draw) * F32_PER_DRAW
    assert rng[0] != 73


def test_rng_f32_returns_exactly_one_on_the_rounding_edge() -> None:
    draw = mint_goldens.glibc_rand_r(np.array([EDGE_SEED], np.uint32), 1).item(0, 0)
    assert draw >= 2**31 - 64, "EDGE_SEED no longer names an edge draw"
    assert rng_f32_numba(_rng(EDGE_SEED)) == np.float32(1.0)


def test_rng_int_returns_hi_on_the_edge_and_lo_without_a_span() -> None:
    assert rng_int_numba(_rng(EDGE_SEED), 0, 6) == 6
    rng = _rng(5)
    assert rng_int_numba(rng, 4, 5) == 4
    assert rng[0] == 5, "a span of one must not consume a draw"
    rng = _rng(5)
    draw = mint_goldens.glibc_rand_r(np.array([5], np.uint32), 1).item(0, 0)
    expected = 1 + int(np.float32(draw) * F32_PER_DRAW * np.float32(4))
    assert rng_int_numba(rng, 1, 5) == expected


def test_rng_f32_at_mixes_the_cell_index_into_a_fresh_state() -> None:
    seed, index = 0xDEADBEEF, 1234
    local = (seed ^ ((index * 747_796_405) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    local = (local + 2_891_336_453) & 0xFFFF_FFFF
    draw = mint_goldens.glibc_rand_r(np.array([local], np.uint32), 1).item(0, 0)
    assert (
        rng_f32_at_numba(np.uint64(seed), np.uint64(index))
        == np.float32(draw) * F32_PER_DRAW
    )


def test_choice_valid_picks_the_kth_set_entry_and_the_last_on_the_edge() -> None:
    valid = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    rng = _rng(11)
    draw = mint_goldens.glibc_rand_r(np.array([11], np.uint32), 1).item(0, 0)
    k = int(np.float32(draw) * F32_PER_DRAW * np.float32(3))
    assert choice_valid_numba(rng, valid, 5) == [1, 3, 4][k]
    assert choice_valid_numba(_rng(EDGE_SEED), valid, 5) == 4
    rng = _rng(11)
    assert choice_valid_numba(rng, np.zeros(5, dtype=np.uint8), 5) == 0
    assert rng[0] == 11, "no valid entry must not consume a draw"


def test_choose_weighted_accumulates_in_float32_order() -> None:
    weights = np.array([0.3, 0.3, 0.15, 0.125, 0.125], dtype=np.float32)
    rng = _rng(21)
    draw = mint_goldens.glibc_rand_r(np.array([21], np.uint32), 1).item(0, 0)
    total = np.float32(0.0)
    for i in range(len(weights)):
        total = np.float32(total + np.float32(weights.item(i)))
    target = np.float32(total * (np.float32(draw) * F32_PER_DRAW))
    cumulative = np.float32(0.0)
    expected = 4
    for i in range(len(weights)):
        cumulative = np.float32(cumulative + np.float32(weights.item(i)))
        if cumulative > target:
            expected = i
            break
    assert choose_weighted_numba(rng, weights, 5) == expected
    assert choose_weighted_numba(_rng(EDGE_SEED), weights, 5) == 4
    assert choose_weighted_numba(_rng(21), np.zeros(3, dtype=np.float32), 3) == 2


@pytest.mark.parametrize(
    ("first", "second"),
    [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1 / 3, 1 / 3), (0.1, 0.9)],
)
def test_choose_weighted_pair_makes_the_choice_of_the_two_weight_array(
    first: float,
    second: float,
) -> None:
    weights = np.array([first, second], dtype=np.float32)
    for seed in (*range(1, 400), EDGE_SEED):
        rng, twin = _rng(seed), _rng(seed)
        assert choose_weighted_pair_numba(
            rng,
            np.float32(weights.item(0)),
            np.float32(weights.item(1)),
        ) == choose_weighted_numba(
            twin,
            weights,
            2,
        ), seed
        assert rng[0] == twin[0], seed


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
