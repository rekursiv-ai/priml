"""Tests for frontier practice: its config, its archive's shape, and the controller.

The archive's rules are ``game/archive_test.py``'s; here the controller sizes
each rollout's practice rows from the steps counted so far, and a restore
marks its rows as branches.
"""

from __future__ import annotations

from typing import (
    Final,
    cast,
)

import math

import numpy as np
import pytest
import torch

from priml.baselines.craftax.env import CraftaxEnv, WorldPool
from priml.baselines.craftax.game.state import TRAINING_STATS_DTYPE
from priml.baselines.craftax.learners.practice import FrontierPractice
from priml.baselines.craftax.lib.arrays import ints


NUM_ENVS: Final = 8
ENVS_PER_BUFFER: Final = 4


def _config(**changes: float) -> FrontierPractice.Config:
    """Return a tiny archive: 4 levels of 2, 2 donors, a quarter of practice."""
    config = FrontierPractice.Config()
    config.fraction = 0.25
    config.num_donors = 2
    config.num_levels = 4
    config.entries_per_level = 2
    for name, value in changes.items():
        setattr(config, name, value)
    return config


def _practice(config: FrontierPractice.Config) -> FrontierPractice:
    return FrontierPractice(
        config,
        num_envs=NUM_ENVS,
        envs_per_buffer=ENVS_PER_BUFFER,
        save_slots=np.full(NUM_ENVS, -1, dtype=np.int32),
        restore_slots=np.full(NUM_ENVS, -1, dtype=np.int32),
    )


def test_the_defaults_are_the_recipes() -> None:
    config = FrontierPractice.Config()
    assert config.fraction == 0.2
    assert config.num_donors == 64
    assert config.level_width == 8.0
    assert config.num_levels == 32
    assert config.entries_per_level == 32
    assert config.entries_per_world == 4
    assert config.reach_decay == 0.999
    assert config.seed == 1973


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("num_donors", 5, r"^practice's 5 donors must fit in the last buffer"),
        ("num_donors", 0, r"^practice's counts must be positive"),
        ("num_levels", 0, r"^practice's counts must be positive"),
        ("entries_per_level", 0, r"^practice's counts must be positive"),
        ("entries_per_world", 0, r"^practice's counts must be positive"),
        ("level_width", 0.0, r"^practice's counts must be positive"),
        # A width that is not a positive finite fp32 never saves, or saves every
        # return into one level, depending on the platform's float conversion.
        ("level_width", math.nan, r"^practice's counts must be positive"),
        ("level_width", math.inf, r"^practice's counts must be positive"),
        ("level_width", 1e-46, r"^practice's counts must be positive"),
        ("fraction", 0.0, r"^fraction must be in \(0, 1\]"),
        ("fraction", 1.5, r"^fraction must be in \(0, 1\]"),
        ("fraction", math.nan, r"^fraction must be in \(0, 1\]"),
        ("reach_decay", 0.0, r"^reach_decay must be in \(0, 1\]"),
        ("seed", -1, r"^seed must be in \[0, 2\*\*32\)"),
        ("seed", 2**32, r"^seed must be in \[0, 2\*\*32\)"),
    ],
)
def test_check_refuses_what_the_archive_cannot_hold(
    name: str,
    value: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FrontierPractice.check(
            _config(**{name: value}),
            num_envs=NUM_ENVS,
            envs_per_buffer=ENVS_PER_BUFFER,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("level_width", 0.5),
        ("fraction", 1.0),
        ("reach_decay", 1.0),
        ("seed", 0),
        ("seed", 2**32 - 1),
        ("num_donors", ENVS_PER_BUFFER),
    ],
)
def test_check_takes_the_edges_of_what_the_archive_can_hold(
    name: str,
    value: float,
) -> None:
    FrontierPractice.check(
        _config(**{name: value}),
        num_envs=NUM_ENVS,
        envs_per_buffer=ENVS_PER_BUFFER,
    )


def test_practice_needs_natural_rows_besides_its_donors() -> None:
    match = r"^practice needs natural rows besides its donors$"
    with pytest.raises(ValueError, match=match):
        FrontierPractice.check(_config(num_donors=4), num_envs=4, envs_per_buffer=4)
    FrontierPractice.check(_config(num_donors=3), num_envs=4, envs_per_buffer=4)


def test_the_archive_is_sized_by_the_config_and_starts_empty() -> None:
    practice = _practice(_config(entries_per_world=1, seed=7))
    kept = practice.archive
    assert practice.carry_slots == 8
    assert kept.states.shape == kept.stats.shape == kept.worlds.shape == (8,)
    assert kept.sizes.tolist() == [0, 0, 0, 0]
    assert kept.reach.shape == (4,)
    assert kept.donor_levels.shape == (2,)
    assert kept.counts.shape == (2,)
    assert kept.stream.tolist() == [7]
    assert (kept.first_donor, kept.envs_per_buffer) == (6, 4)
    assert (kept.per_level, kept.per_world) == (2, 1)
    assert kept.level_width == np.float32(8.0)
    assert kept.reach_decay == 0.999
    assert practice.metrics() == {
        "practice/fraction": 0.0,
        "practice/populated_levels": 0.0,
        "practice/archive_entries": 0.0,
        "practice/sample_level_mean": 0.0,
        "practice/selected": 0.0,
    }


def test_the_archive_holds_the_kernels_dtypes_and_the_envs_slots() -> None:
    """The step's Numba kernels are compiled for these dtypes; the slots are shared."""
    save_slots = np.full(NUM_ENVS, -1, dtype=np.int32)
    restore_slots = np.full(NUM_ENVS, -1, dtype=np.int32)
    practice = FrontierPractice(
        _config(),
        num_envs=NUM_ENVS,
        envs_per_buffer=ENVS_PER_BUFFER,
        save_slots=save_slots,
        restore_slots=restore_slots,
    )
    kept = practice.archive
    assert kept.save_slots is save_slots
    assert kept.restore_slots is restore_slots
    dtypes = {
        "rngs": np.uint32,
        "actions": np.float32,
        "rewards": np.float32,
        "worlds": np.uint64,
        "sizes": np.int64,
        "reach": np.float64,
        "weights": np.float64,
        "stream": np.uint32,
        "donor_levels": np.int64,
        "donor_worlds": np.uint64,
        "donor_steps": np.int64,
        "counts": np.int64,
    }
    assert {name: cast("np.ndarray", getattr(kept, name)).dtype for name in dtypes} == {
        name: np.dtype(dtype) for name, dtype in dtypes.items()
    }
    # The entries keep a training row's stats: its clocks and branch count.
    assert kept.stats.dtype == TRAINING_STATS_DTYPE


@pytest.mark.parametrize(
    ("fraction", "controller", "counted", "selected"),
    [
        # (horizon, restores) so far, (steps, branch_steps) now.
        # Before any restore, a branch lasts the horizon: ceil(10 / 4).
        (0.5, (4, 0), (16, 14), 3),
        # Then the branch steps over the restores: ceil(6 / 2).
        (0.5, (4, 5), (0, 10), 3),
        # At most the horizon, 16, not 100: ceil(28 / 16).
        (1.0, (16, 1), (0, 100), 2),
        # At least one step, not 0.75: ceil(2 / 1).
        (1.0, (4, 40), (0, 30), 2),
        # Ahead of the target, none.
        (0.5, (4, 5), (0, 20), 0),
        # Capped at ceil(num_envs * fraction) = 4 ...
        (0.5, (4, 0), (64, 0), 4),
        # ... and at the rows besides the two donors, 6.
        (1.0, (4, 0), (64, 0), 6),
        # No rollout counted yet, so no horizon: none.
        (0.5, (0, 0), (16, 0), 0),
    ],
)
def test_the_controller_restores_the_deficit_over_a_branchs_mean_length(
    fraction: float,
    controller: tuple[int, int],
    counted: tuple[int, int],
    selected: int,
) -> None:
    """``ceil(deficit / duration)`` rows, the deficit against the target this rollout."""
    practice = _practice(_config(fraction=fraction))
    practice.archive.sizes[0] = 1
    practice.controller["horizon"], practice.controller["restores"] = controller
    assert practice._selected(*counted) == selected


def test_the_metrics_read_the_last_prepares_counters() -> None:
    practice = _practice(_config())
    practice.archive.sizes[:] = [1, 0, 2, 0]
    for name, value in (("steps", 96), ("branch_steps", 40), ("selected", 3)):
        practice.controller[name] = value
    practice.controller["levels"] = 5
    assert practice.metrics() == {
        "practice/fraction": 40 / 96,
        "practice/populated_levels": 2.0,
        "practice/archive_entries": 3.0,
        "practice/sample_level_mean": 5 / 3,
        "practice/selected": 3.0,
    }


def test_an_empty_archive_restores_nothing() -> None:
    practice = _practice(_config(fraction=1.0))
    practice.controller["horizon"] = 4
    assert practice._selected(0, 0) == 0


def test_the_state_dict_is_every_array_as_live_bytes() -> None:
    practice = _practice(_config())
    state = practice.state_dict()
    assert set(state) == {
        "states",
        "rngs",
        "stats",
        "actions",
        "rewards",
        "worlds",
        "sizes",
        "reach",
        "weights",
        "stream",
        "donor_levels",
        "donor_worlds",
        "donor_steps",
        "counts",
        "controller",
    }
    assert all(value.dtype == torch.uint8 for value in state.values())
    state["sizes"].view(torch.int64)[2] = 5
    state["controller"].view(torch.int64)[0] = 3
    assert practice.archive.sizes.tolist() == [0, 0, 5, 0]
    assert practice.controller["restores"] == 3


def _env() -> CraftaxEnv:
    """Return two buffers of four, the last two rows donors, reset."""
    cfg = CraftaxEnv.Config()
    cfg.num_envs = NUM_ENVS
    cfg.num_buffers = NUM_ENVS // ENVS_PER_BUFFER
    cfg.threads_per_buffer = 1
    pool = cfg.restart = WorldPool.Config()
    pool.num_worlds = 2
    cfg.practice = _config()
    env = cfg.make()
    env.reset()
    return env


@pytest.mark.compute_large_fixture
def test_the_controller_sizes_each_rollout_toward_the_fraction() -> None:
    """The deficit over a branch's mean length, capped at ``ceil(N * fraction)``."""
    env = _env()
    practice = env.practice
    assert practice is not None
    kept = practice.archive
    try:
        # Before any step the horizon is unknown, and the archive is empty.
        env.prepare_rollout()
        assert practice.metrics()["practice/selected"] == 0.0
        kept.states[2:5] = env.states[:3]
        kept.stats[2:5] = env.stats[:3]
        kept.sizes[:] = [0, 2, 1, 0]
        kept.save_slots[6] = 3
        # One rollout of 4 steps: duration 4, deficit (32 + 32) / 4 = 16, so
        # 16 / 4 = 4 rows, capped at min(8 - 2, ceil(8 / 4)) = 2.
        kept.counts[:] = [16, 16]
        env.prepare_rollout()
        assert practice.metrics()["practice/selected"] == 2.0
        assert kept.save_slots.tolist() == [-1] * NUM_ENVS
        restored = ints(kept.restore_slots)
        assert all(slot in {2, 3, 4} for slot in restored[:2])
        assert restored[2:] == [-1] * 6
        assert env.stats["branch"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
        assert practice.metrics()["practice/sample_level_mean"] in {1.0, 1.5, 2.0}
        # 20 branch steps over 2 restores last 10, clamped to the horizon 4;
        # the deficit 96 / 4 - 20 = 4 takes one row.
        kept.counts[:] = [32, 32]
        env.stats["branch_steps"][:2] = 10
        env.prepare_rollout()
        assert practice.metrics()["practice/selected"] == 1.0
        # Past the target, the deficit is negative: no row.
        kept.counts[:] = [48, 48]
        env.stats["branch_steps"][:2] = 20
        env.prepare_rollout()
        metrics = practice.metrics()
        assert metrics["practice/selected"] == 0.0
        assert metrics["practice/fraction"] == 40 / 96
        assert metrics["practice/populated_levels"] == 2.0
        assert metrics["practice/archive_entries"] == 3.0
        assert metrics["practice/sample_level_mean"] == 0.0
        assert kept.restore_slots.tolist() == [-1] * NUM_ENVS
        assert practice.controller.tolist()[:4] == (3, 96, 40, 4)
        # All practice: the cap is every row but the donors, min(8 - 2, 8) = 6.
        practice.fraction = 1.0
        kept.counts[:] = [64, 64]
        env.prepare_rollout()
        assert practice.metrics()["practice/selected"] == 6.0
        assert kept.restore_slots.tolist()[6:] == [-1, -1]
    finally:
        env.close()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
