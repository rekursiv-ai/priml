"""Tests for capture's buffer step, its kernels run eagerly on one tiny world."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pytest

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    DEFAULT_MAX_TIMESTEPS,
    OBS_SIZE,
    Action,
    env_state,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules
from priml.baselines.craftax.world_model.capture import step
from priml.baselines.craftax.world_model.capture.seeds import splitmix64
from priml.baselines.craftax.world_model.capture.step import (
    BAD_ACTION,
    ENDED,
    FULL,
    HASH_STRIDE,
    RECORDED_END,
    TOO_LONG,
    Rows,
    RowsView,
    fnv1a_numba,
    record_rows_numba,
)


if TYPE_CHECKING:
    from collections.abc import Generator


_CHUNK: Final = 2 * HASH_STRIDE
"""Decisions a row's chunk holds: two state hashes' worth."""

_FAR: Final = 1_000
"""Decisions left on the clock: no episode times out within a test."""


@pytest.fixture
def world() -> Generator[None]:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - _FAR)):
        yield


def _rows(*, recording: bool = True, chunk: int = _CHUNK) -> Rows:
    """Return one tiny-world row, its observation and mask written, recording or not."""
    states = new_states(1)
    tiny_world(env_state(states, 0), np.zeros(1, np.uint32), timestep=0)
    env_state(states, 0).timestep = DEFAULT_MAX_TIMESTEPS - _FAR
    return Rows(
        states=states,
        state_bytes=states.view(np.uint8).reshape(1, -1),
        rngs=np.ones(1, np.uint32),
        stats=new_stats(1),
        # One row, and ``step``'s ``Rows.actions`` is ``[N, 1]`` by contract.
        actions=np.zeros((1, 1), np.float32),
        masks=np.ones((1, ATN_DIM), np.uint8),
        observations=np.zeros((1, OBS_SIZE), np.float32),
        rewards=np.zeros(1, np.float32),
        terminals=np.zeros(1, np.float32),
        recording=np.array([1 if recording else 0], np.uint8),
        decisions=np.zeros(1, np.int64),
        fill=np.zeros(1, np.int64),
        chunk_actions=np.zeros((1, chunk), np.uint8),
        chunk_rewards=np.zeros((1, chunk), np.float32),
        chunk_done=np.zeros((1, chunk), np.uint8),
        chunk_floors=np.zeros((1, chunk), np.uint8),
        chunk_hashes=np.zeros((1, max(chunk // HASH_STRIDE, 1)), np.uint64),
        hash_fill=np.zeros(1, np.int64),
        final_hash=np.zeros(1, np.uint64),
        epsilon=np.zeros(1, np.float32),
        epsilon_rng=np.zeros(1, np.uint64),
        last_gain=np.zeros(1, np.int64),
        health_snaps=np.zeros(1, np.int64),
        returned=np.zeros(1, np.float32),
        split=np.zeros(1, np.uint8),
        flags=np.zeros(1, np.uint8),
    )


def _step(
    rows: Rows,
    action: float,
    *,
    stall_limit: int = -1,
    max_decisions: int = 10**6,
) -> int:
    """Play ``action`` on row 0 and return its flags."""
    rows.actions[0, 0] = action
    record_rows_numba(
        rows,
        0,
        1,
        stall_limit,
        max_decisions,
        DEFAULT_MAX_TIMESTEPS,
        Rules(),
        Rules(),
    )
    return rows.flags.item(0)


def _hash_row(rows: Rows) -> np.uint64:
    """Return the archive hash of the single row's state bytes."""
    return fnv1a_numba(rows.states.view(np.uint8))


def test_fnv1a_xors_then_multiplies_each_byte_from_the_archive_basis() -> None:
    # The kernel's own source, which ``eager`` swaps for a Python-int stand-in;
    # its uint64 product wraps, as Numba's does.
    basis, prime, mask = 1_469_598_103_934_665_603, 1_099_511_628_211, 2**64 - 1
    expected = basis
    for byte in b"ab":
        expected = (expected ^ byte) * prime & mask
    with np.errstate(over="ignore"):
        empty = fnv1a_numba.py_func(np.zeros(0, np.uint8))
        ab = fnv1a_numba.py_func(np.frombuffer(b"ab", np.uint8))
    assert empty == np.uint64(basis)
    assert ab == np.uint64(expected)


def test_a_recorded_decision_writes_its_chunk_and_first_hash(world: None) -> None:
    del world
    rows = _rows()
    before = _hash_row(rows)
    assert _step(rows, float(Action.UP)) == 0
    assert rows.fill.tolist() == [1]
    assert rows.decisions.tolist() == [1]
    assert rows.chunk_actions[0, 0] == Action.UP
    assert rows.chunk_floors[0, 0] == 0
    assert rows.hash_fill.tolist() == [1]
    assert rows.chunk_hashes[0, 0] == before
    assert rows.terminals.tolist() == [0.0]
    assert rows.health_snaps.tolist() == [0]


def test_an_unrecorded_row_plays_without_writing_a_chunk(world: None) -> None:
    del world
    rows = _rows(recording=False)
    assert _step(rows, float(Action.UP)) == 0
    assert rows.fill.tolist() == [0]
    assert rows.decisions.tolist() == [0]
    assert rows.hash_fill.tolist() == [0]
    assert rows.recording.tolist() == [0]


def test_a_positive_reward_records_its_gain_and_return(world: None) -> None:
    # The tiny world's tree stands above the start: facing it, DO collects wood.
    del world
    rows = _rows()
    _step(rows, float(Action.UP))
    _step(rows, float(Action.DO))
    assert rows.chunk_rewards[0, 1] > 0
    assert rows.last_gain.tolist() == [2]
    assert rows.returned.tolist() == [1.0]


def test_a_recorded_episode_too_long_is_dropped(world: None) -> None:
    del world
    rows = _rows()
    rows.decisions[0] = 5
    assert _step(rows, float(Action.UP), max_decisions=5) == TOO_LONG
    assert rows.recording.tolist() == [0]
    assert rows.fill.tolist() == [0]


@pytest.mark.parametrize("action", [-1.0, float(ATN_DIM), float("nan")])
def test_an_action_off_the_space_is_dropped_and_played_as_noop(
    world: None,
    action: float,
) -> None:
    del world
    rows = _rows()
    assert _step(rows, action) == BAD_ACTION
    assert rows.recording.tolist() == [0]
    assert rows.fill.tolist() == [0]


def test_a_stalled_training_episode_is_truncated_at_the_timeout(world: None) -> None:
    del world
    rows = _rows()
    flags = _step(rows, float(Action.NOOP), stall_limit=1)
    assert flags == RECORDED_END
    assert rows.recording.tolist() == [0]
    state = env_state(rows.states, 0)
    assert state.timestep == DEFAULT_MAX_TIMESTEPS - 1
    # The hash is of the State the decision left, before the clock was set.
    state.timestep = DEFAULT_MAX_TIMESTEPS - _FAR + 1
    assert rows.final_hash[0] == _hash_row(rows)
    assert rows.returned.tolist() == [0.0]


def test_a_validation_episode_never_stalls(world: None) -> None:
    del world
    rows = _rows()
    rows.split[0] = 1
    assert _step(rows, float(Action.NOOP), stall_limit=1) == 0
    assert rows.recording.tolist() == [1]


def test_an_episode_ending_hands_over_its_last_hash(world: None) -> None:
    del world
    rows = _rows()
    env_state(rows.states, 0).timestep = DEFAULT_MAX_TIMESTEPS - 1
    assert _step(rows, float(Action.NOOP)) == RECORDED_END | ENDED
    assert rows.final_hash[0] == _hash_row(rows)
    assert rows.chunk_done[0, 0] == 1
    assert rows.terminals.tolist() == [1.0]


def test_a_filled_chunk_is_flagged(world: None) -> None:
    del world
    rows = _rows(chunk=1)
    assert _step(rows, float(Action.NOOP)) == FULL


def test_health_off_its_grid_counts_a_snap(world: None) -> None:
    del world
    rows = _rows()
    env_state(rows.states, 0).player_health = np.float32(8.98)
    _step(rows, float(Action.NOOP))
    assert rows.health_snaps.tolist() == [1]


def _override_draws(seed: int) -> tuple[int, int]:
    """Return the epsilon override's two SplitMix64 draws from ``seed``."""
    seed, first = splitmix64(seed)
    _, second = splitmix64(seed)
    return first, second


def test_epsilon_one_plays_a_uniform_legal_action(world: None) -> None:
    del world
    rows = _rows()
    rows.epsilon[0] = np.float32(1.0)
    # No legal action is NOOP (0), so a pick that ignored the mask shows.
    legal = [Action.LEFT, Action.UP, Action.DO]
    rows.masks[0] = 0
    rows.masks[0, legal] = 1
    _step(rows, float(Action.NOOP))
    _, second = _override_draws(0)
    assert rows.chunk_actions[0, 0] == legal[second % 3]


def test_epsilon_with_no_legal_action_keeps_the_policys(world: None) -> None:
    del world
    rows = _rows()
    rows.epsilon[0] = np.float32(1.0)
    rows.masks[0] = 0
    _step(rows, float(Action.UP))
    assert rows.chunk_actions[0, 0] == Action.UP
    # One draw, the epsilon test's: with nothing legal, no pick is drawn.
    assert rows.epsilon_rng[0] == np.uint64(splitmix64(0)[0])


def test_a_draw_above_epsilon_keeps_the_policys_action(world: None) -> None:
    del world
    rows = _rows()
    first, _ = _override_draws(0)
    # Just below the first draw's uniform: the override declines.
    rows.epsilon[0] = np.float32((first >> 11) * 2.0**-53 * 0.5)
    _step(rows, float(Action.UP))
    assert rows.chunk_actions[0, 0] == Action.UP
    assert rows.epsilon_rng[0] == np.uint64(0x9E37_79B9_7F4A_7C15)


def test_explore_falls_back_when_the_pick_finds_no_set_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A mask that reads set while counting and clear while picking: the last
    # line of the pick returns the policy's action.
    class _Flicker:
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, index: tuple[int, int]) -> int:
            del index
            self.reads += 1
            return 1 if self.reads <= ATN_DIM else 0

    class _View:
        masks = _Flicker()
        epsilon = np.ones(1, np.float32)
        epsilon_rng = np.zeros(1, np.uint64)

    monkeypatch.setattr(step, "_splitmix64_numba", step._splitmix64_numba.py_func)
    explore = step._explore_numba.py_func
    with np.errstate(over="ignore"):
        picked = explore(cast("RowsView", _View()), 0, np.float32(Action.UP))
    assert picked == np.float32(Action.UP)


@pytest.mark.parametrize(
    ("value", "rounded"),
    [(2.5, 3.0), (-2.5, -3.0), (2.4, 2.0), (-2.4, -2.0), (0.0, 0.0)],
)
def test_lround_rounds_halves_away_from_zero(value: float, rounded: float) -> None:
    assert step._lround_numba.py_func(np.float64(value)) == rounded


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
