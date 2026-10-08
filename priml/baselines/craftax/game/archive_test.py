"""Tests for practice's archive: snapshots, the donors' saves, and the restores.

The snapshot pair is checked the way the recipe's own native test checks it:
restored into another row, a saved environment steps exactly as the original
went on to. The archive's rules -- return levels, reach, the per-level and
per-world caps, the level weights and the crossing-only saves -- run on tiny
archives with every draw replayed from the stream by hand.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest

from priml.baselines.craftax.game import archive, step
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.state import (
    ACTION_OBS_SIZE,
    ATN_DIM,
    OBS_SIZE,
    STATE_DTYPE,
    TRAINING_STATS_DTYPE,
    Action,
    Archive,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    build_pool_numba,
)


RULES: Final = step.Rules(previous_action=True, stall_limit=50, practice=True)
"""A practice run's rules: the previous action, a stall cap, branch flags."""

STREAM: Final = 1_973
"""The archive stream's seed in these tests, the recipe's."""

CONTINUED: Final = (
    "episode_return_accum",
    "episode_length_accum",
    "max_floor_accum",
    "steps",
    "last_ticks",
    "undefined_spawns",
    "first_undefined_step",
    "stall_limit",
    "last_gain",
)
"""The stats a restore continues; the log, the escape stream and the flag are the row's."""


def _pool(count: int) -> np.ndarray:
    pool = new_states(count)
    build_pool_numba(
        pool,
        pool.view(np.uint8).reshape(count, STATE_DTYPE.itemsize),
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
        0,
    )
    return pool


# The last ``donors`` environments are its donors, all in one buffer.
def _batch(
    num_envs: int,
    *,
    levels: int = 4,
    per_level: int = 3,
    per_world: int = 2,
    donors: int = 2,
) -> tuple[step.Batch, Archive]:
    """Return ``num_envs`` zeroed environments under :data:`RULES`, and their archive."""
    rngs = np.arange(num_envs, dtype=np.uint32)
    stats = new_stats(num_envs, TRAINING_STATS_DTYPE)
    stats["escape"][:, 0] = rngs ^ np.uint32(0x9E37_79B9)
    slots = levels * per_level
    kept = Archive(
        states=new_states(slots),
        rngs=np.zeros(slots, dtype=np.uint32),
        stats=new_stats(slots, TRAINING_STATS_DTYPE),
        actions=np.zeros(slots, dtype=np.float32),
        rewards=np.zeros(slots, dtype=np.float32),
        worlds=np.zeros(slots, dtype=np.uint64),
        sizes=np.zeros(levels, dtype=np.int64),
        reach=np.zeros(levels, dtype=np.float64),
        weights=np.zeros(levels, dtype=np.float64),
        stream=np.array([STREAM], dtype=np.uint32),
        donor_levels=np.zeros(donors, dtype=np.int64),
        donor_worlds=np.zeros(donors, dtype=np.uint64),
        donor_steps=np.zeros(donors, dtype=np.int64),
        counts=np.zeros(1, dtype=np.int64),
        save_slots=np.full(num_envs, -1, dtype=np.int32),
        restore_slots=np.full(num_envs, -1, dtype=np.int32),
        first_donor=num_envs - donors,
        envs_per_buffer=num_envs,
        per_level=per_level,
        per_world=per_world,
        level_width=np.float32(8.0),
        reach_decay=0.5,
    )
    batch = step.Batch(
        new_states(num_envs),
        rngs,
        stats,
        np.zeros((num_envs, 1), dtype=np.float32),
        np.zeros((num_envs, ACTION_OBS_SIZE), dtype=np.float32),
        np.ones((num_envs, ATN_DIM), dtype=np.uint8),
        np.zeros(num_envs, dtype=np.float32),
        np.zeros(num_envs, dtype=np.float32),
        RULES,
    )
    return batch, kept


def _row(batch: step.Batch, row: int) -> list[object]:
    """Return everything a step of ``row`` writes, as bytes and lists."""
    stats = batch.stats[row : row + 1]
    return [
        batch.states[row : row + 1].tobytes(),
        batch.rngs.item(row),
        batch.observations[row, :].tobytes(),
        batch.masks[row, :].tobytes(),
        batch.rewards.item(row),
        batch.terminals.item(row),
        *(stats[name].item() for name in CONTINUED),
    ]


@pytest.mark.compute_large_fixture
def test_restore_then_step_replays_the_uninterrupted_trajectory() -> None:
    """Row 0 saved, then played on; row 2 restored from it plays the same bytes.

    The stall clock rides along: a last gain set before the save continues.
    """
    pool = _pool(3)
    batch, kept = _batch(3)
    step.reset_range_numba(batch, pool, 0, 0, 3)
    for t in range(5):
        batch.actions[:, 0] = np.float32(t % 6)
        step.step_range_numba(batch, pool, 0, 3)
    batch.stats[0]["last_gain"] = 3
    archive.save_numba(batch, 0, kept, 4)
    saved = _row(batch, 0)
    expected: list[list[object]] = []
    for t in range(32):
        batch.actions[:, 0] = np.float32(t % 6)
        step.step_range_numba(batch, pool, 0, 3)
        expected.append(_row(batch, 0))
    archive.restore_numba(batch, 2, kept, 4)
    assert _row(batch, 2) == saved
    assert batch.stats[2]["branch"] == 1
    assert batch.observations[2, OBS_SIZE] == np.float32(4.0)
    for t in range(32):
        batch.actions[:, 0] = np.float32(t % 6)
        step.step_range_numba(batch, pool, 0, 3)
        assert _row(batch, 2) == expected[t], t
    assert batch.stats[2]["escape"] != batch.stats[0]["escape"]


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(
    ("text", "shape", "expected"),
    [
        # FNV's published test vectors.
        (b"a", (1, 1, 1), 0xAF63_DC4C_8601_EC8C),
        (b"foobar", (1, 2, 3), 0x8594_4171_F739_67E8),
    ],
)
def test_the_world_hash_is_64_bit_fnv_1a(
    text: bytes,
    shape: tuple[int, int, int],
    expected: int,
) -> None:
    grid = np.frombuffer(text, dtype=np.uint8).reshape(shape).copy()
    assert int(archive.world_hash_numba(grid)) == expected


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(
    ("episode_return", "level"),
    [(0.0, 0), (7.9, 0), (8.0, 1), (25.0, 3), (1000.0, 3)],
)
def test_a_return_level_truncates_and_clamps_to_the_last(
    episode_return: float,
    level: int,
) -> None:
    _, kept = _batch(3)
    assert archive.return_level_numba(np.float32(episode_return), kept) == level


@pytest.mark.compute_large_fixture
def test_reach_counts_every_start_and_each_first_crossing() -> None:
    """Four starts decay by half each; only the first episode reaches level 3."""
    _, kept = _batch(3)
    for donor in (0, 1, 0, 1):
        assert not archive.record_level_numba(kept, donor, 0, True)
    assert kept.reach.tolist() == [1.875, 0.0, 0.0, 0.0]
    assert archive.record_level_numba(kept, 0, 3, False)
    assert not archive.record_level_numba(kept, 0, 2, False)
    assert not archive.record_level_numba(kept, 1, 0, False)
    assert kept.reach.tolist() == [1.875, 1.0, 1.0, 1.0]
    assert kept.donor_levels.tolist() == [3, 0]


@pytest.mark.compute_large_fixture
def test_a_level_appends_caps_a_world_and_replaces_from_the_stream() -> None:
    _, kept = _batch(3, per_level=3, per_world=2)
    base = 3
    stream = np.array([STREAM], dtype=np.uint32)
    assert archive.insert_numba(kept, 1, np.uint64(7)) == base
    assert archive.insert_numba(kept, 1, np.uint64(7)) == base + 1
    # A third save of world 7 replaces one of its two, drawn from the stream.
    assert (
        archive.insert_numba(kept, 1, np.uint64(7)) == base + rand_r_numba(stream) % 2
    )
    assert archive.insert_numba(kept, 1, np.uint64(9)) == base + 2
    # The level is full: any of its three entries, drawn likewise.
    assert (
        archive.insert_numba(kept, 1, np.uint64(11)) == base + rand_r_numba(stream) % 3
    )
    assert kept.stream[0] == stream[0]
    assert kept.sizes.tolist() == [0, 3, 0, 0]


@pytest.mark.compute_large_fixture
def test_the_level_draw_weights_populated_levels_by_inverse_reach() -> None:
    _, kept = _batch(3)
    kept.sizes[:] = [1, 0, 2, 1]
    kept.reach[:] = [100.0, 1.0, 0.5, 10.0]
    archive.level_weights_numba(kept)
    total = 1 / 100 + 1 / 1 + 1 / 10
    assert kept.weights.tolist() == [1 / 100 / total, 0.0, 1 / total, 1 / 10 / total]
    stream = np.array([STREAM], dtype=np.uint32)
    cumulative = np.cumsum(kept.weights)
    draws = [archive.draw_level_numba(kept) for _ in range(300)]
    expected = [
        int(np.searchsorted(cumulative, rand_r_numba(stream) / 2**31, side="right"))
        for _ in range(300)
    ]
    assert draws == expected
    assert draws.count(2) > draws.count(3) > draws.count(0)
    assert draws.count(1) == 0


@pytest.mark.compute_large_fixture
def test_a_donor_saves_only_on_the_step_it_crosses_a_new_level() -> None:
    batch, kept = _batch(3)
    batch.states["map"][1] = 3
    archive.restart_donors_numba(batch, kept)
    # Two starts, the first decayed by half before the second.
    assert kept.reach.tolist() == [1.5, 0.0, 0.0, 0.0]
    assert kept.donor_worlds[0] != kept.donor_worlds[1]
    batch.stats[1]["episode_return_accum"] = np.float32(9.0)
    batch.actions[1, 0] = np.float32(Action.DO)
    batch.rewards[1] = np.float32(1.0)
    archive.observe_donors_numba(batch, kept, 0, 3)
    assert kept.save_slots.tolist() == [-1, 3, -1]
    assert kept.sizes.tolist() == [0, 1, 0, 0]
    assert kept.states[3:4].tobytes() == batch.states[1:2].tobytes()
    assert kept.actions[3] == np.float32(Action.DO)
    assert kept.rewards[3] == np.float32(1.0)
    assert kept.worlds[3] == kept.donor_worlds[0]
    # No new level, no save; a start never saves, whatever its return.
    archive.observe_donors_numba(batch, kept, 0, 3)
    assert kept.save_slots.tolist() == [-1, -1, -1]
    batch.terminals[1] = np.float32(1.0)
    batch.stats[1]["episode_return_accum"] = np.float32(17.0)
    archive.observe_donors_numba(batch, kept, 0, 3)
    assert kept.save_slots.tolist() == [-1, -1, -1]
    assert kept.sizes.tolist() == [0, 1, 0, 0]
    assert kept.donor_levels.tolist() == [2, 0]


@pytest.mark.compute_large_fixture
def test_a_save_replaced_in_its_own_step_leaves_no_slot_to_fill() -> None:
    """Two donors of one world fill a one-entry level: the first's carry must not land."""
    batch, kept = _batch(3, per_level=1, per_world=2)
    archive.restart_donors_numba(batch, kept)
    batch.stats["episode_return_accum"][1:] = np.float32(8.0)
    archive.observe_donors_numba(batch, kept, 0, 3)
    assert kept.save_slots.tolist() == [-1, -1, 1]
    assert kept.sizes.tolist() == [0, 1, 0, 0]


@pytest.mark.compute_large_fixture
def test_after_step_counts_the_steps_clears_restores_and_observes_the_donors() -> None:
    batch, kept = _batch(3)
    archive.restart_donors_numba(batch, kept)
    kept.restore_slots[:] = [2, 5, -1]
    batch.stats["episode_return_accum"][2] = np.float32(8.0)
    archive.after_step_numba(batch, kept, 0, 3)
    archive.after_step_numba(batch, kept, 0, 3)
    assert kept.counts.tolist() == [6]
    assert kept.restore_slots.tolist() == [-1, -1, -1]
    assert kept.save_slots.tolist() == [-1, -1, -1]
    assert kept.sizes.tolist() == [0, 1, 0, 0]
    # Without an archive it does nothing at all.
    archive.after_step_numba(batch, None, 0, 3)
    assert kept.counts.tolist() == [6]


@pytest.mark.compute_large_fixture
def test_restore_rows_restores_the_first_rows_from_drawn_entries() -> None:
    batch, kept = _batch(4)
    for slot in (3, 4, 9):
        kept.states[slot]["timestep"] = slot
        kept.stats[slot]["last_gain"] = slot
        kept.actions[slot] = np.float32(slot)
        kept.rewards[slot] = np.float32(slot / 2)
    kept.sizes[:] = [0, 2, 0, 1]
    kept.reach[:] = [4.0, 2.0, 1.0, 1.0]
    batch.terminals[:] = 1.0
    stream = np.array([STREAM], dtype=np.uint32)
    cumulative = np.cumsum([0.0, 1 / 2 / 1.5, 0.0, 1 / 1.5])
    slots: list[int] = []
    for _ in range(2):
        level = int(
            np.searchsorted(cumulative, rand_r_numba(stream) / 2**31, side="right"),
        )
        slots.append(level * 3 + int(rand_r_numba(stream)) % kept.sizes.item(level))
    levels = archive.restore_rows_numba(batch, kept, 2)
    assert levels == sum(slot // 3 for slot in slots)
    assert kept.restore_slots.tolist() == [*slots, -1, -1]
    for row, slot in enumerate(slots):
        assert batch.states[row]["timestep"] == slot
        assert batch.stats[row]["last_gain"] == slot
        assert batch.stats[row]["branch"] == 1
        assert batch.observations[row, OBS_SIZE] == np.float32(slot)
        assert batch.rewards[row] == np.float32(slot / 2)
    assert batch.terminals.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert batch.stats["branch"][2:].tolist() == [0, 0]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
