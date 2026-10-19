"""Tests for the hand-built ghosts the site's tests take."""

from __future__ import annotations

import numpy as np

from priml.baselines.craftax.ghosts.extract import ENDED, NEW_TILE
from priml.baselines.craftax.ghosts.testing import made_ghost
from priml.baselines.craftax.lib.arrays import int_rows, ints


def test_an_even_ghost_reaches_new_tiles_every_70th_decision_and_ends() -> None:
    ghost = made_ghost(2, decisions=150, outcome="death")
    assert (ghost.ordinal, ghost.world_seed, ghost.sampling_seed) == (2, 1, 102)
    assert (ghost.decisions, ghost.outcome) == (150, "death")
    new = [t for t, flags in enumerate(ghost.active) if flags & NEW_TILE]
    assert new == [2, 72, 142]
    assert ghost.active[-1] & ENDED
    assert not any(flags & ENDED for flags in ghost.active[:-1])
    assert ghost.players == bytes((2 + t) % 43 for t in range(150))
    # An even ghost stays on floor 0 and escapes nothing.
    assert ghost.end == (0, 24, 26, 2)
    assert ghost.floor_first == (0, -1, *[-1] * 7)
    assert ghost.events.escapes.shape == (0, 5)


def test_an_odd_ghost_descends_halfway_and_has_an_escape_row() -> None:
    ghost = made_ghost(1, decisions=50, outcome="win", world_seed=9, sampling_seed=4)
    assert (ghost.world_seed, ghost.sampling_seed) == (9, 4)
    assert ghost.end == (1, 24, 25, 2)
    assert ghost.floor_first == (0, 25, *[-1] * 7)
    assert int_rows(ghost.events.escapes) == [(7, 0, 2, 2, 3)]


def test_creature_samples_are_every_4th_decision_in_the_ghosts_row() -> None:
    ghost = made_ghost(3, decisions=9, outcome="death")
    # ceil(9 / 4) = 3 samples of 5 bytes, each in row 3.
    assert ints(ghost.samples) == [0, 5, 10, 15]
    assert ghost.creatures == bytes(
        [1, 0x10, 3, 0, 0, 1, 0x10, 3, 1, 0, 1, 0x10, 3, 2, 0],
    )
    assert int_rows(ghost.events.map) == [(5, 0, 1, 3, 6, 0)]


def test_each_sleep_holds_its_samples_and_one_change_row_per_sampled_sleep() -> None:
    ghost = made_ghost(0, decisions=90, outcome="win", sleeps=((43, 13), (77, 3)))
    assert int_rows(ghost.sleeps) == [(43, 13), (77, 3)]
    # 13 ticks give (13 - 1) // 4 = 3 samples; 3 ticks give none.
    assert ints(ghost.sleep_samples) == [0, 5, 10, 15]
    assert ghost.sleep_creatures == b"".join(
        bytes([1, 0x10, 7, k, 0]) for k in range(3)
    )
    assert int_rows(ghost.sleep_changes) == [(0, 0, 5, 5, 16, 0)]


def test_a_sleep_of_a_whole_number_of_strides_omits_its_waking_tick() -> None:
    # 12 ticks: samples after ticks 4 and 8, none at the waking 12th.
    ghost = made_ghost(0, decisions=20, outcome="win", sleeps=((5, 12),))
    assert ints(ghost.sleep_samples) == [0, 5, 10]


def test_a_sleepless_ghost_has_empty_sleep_arrays() -> None:
    ghost = made_ghost(4, decisions=8, outcome="death")
    assert ghost.sleeps.shape == (0, 2)
    assert ghost.sleep_changes.shape == (0, 6)
    assert ints(ghost.sleep_samples) == [0]
    assert ghost.sleep_creatures == b""
    assert ghost.samples.dtype == np.int64


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
