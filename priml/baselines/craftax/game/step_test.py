"""Tests for the step: its order, its ticks, its reward and its auto-reset.

Bit-exactness against PufferLib is the parity tests' job; these check the step's
contract from a pool world -- what a step returns, how sleep makes it tick,
and how an ended episode is logged and restarted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import re

from llvmlite import ir
from numba.core.registry import cpu_target
from numba.np.numpy_support import from_dtype

import numpy as np
import pytest

from priml.baselines.craftax.game import step
from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    ACTION_OBS_SIZE,
    ATN_DIM,
    NO_ACTION,
    NUM_ACHIEVEMENTS,
    NUM_LEVELS,
    OBS_SIZE,
    STATE_DTYPE,
    STATS_DTYPE,
    SYMBOLIC_OBS_SIZE,
    TRAINING_STATS_DTYPE,
    Achievement,
    Action,
    BlockType,
    env_state,
    env_stats,
    new_states,
    new_stats,
    training_stats,
)
from priml.baselines.craftax.game.testing import (
    FMA_MNEMONIC,
    ir_builder,
    kernel_inspection,
    kernel_llvm,
)
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    build_pool_numba,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numba.core.dispatcher import Dispatcher

    from priml.baselines.craftax.game.state import EnvState


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


def _batch(
    num_envs: int,
    pool: np.ndarray,
    rules: step.Rules | None = None,
) -> step.Batch:
    """Return reset environments; with a training option on, its stats and streams."""
    rules = step.Rules() if rules is None else rules
    rngs = np.arange(num_envs, dtype=np.uint32)
    training = rules.stall_limit is not None or rules.practice is not None
    stats = new_stats(num_envs, TRAINING_STATS_DTYPE if training else STATS_DTYPE)
    if training:
        stats["escape"][:, 0] = rngs ^ np.uint32(0x9E37_79B9)
    width = ACTION_OBS_SIZE if rules.previous_action is True else OBS_SIZE
    batch = step.Batch(
        new_states(num_envs),
        rngs,
        stats,
        np.zeros((num_envs, 1), dtype=np.float32),
        np.zeros((num_envs, width), dtype=np.float32),
        np.ones((num_envs, ATN_DIM), dtype=np.uint8),
        np.zeros(num_envs, dtype=np.float32),
        np.zeros(num_envs, dtype=np.float32),
        rules,
    )
    step.reset_range_numba(batch, pool, 0, 0, num_envs)
    return batch


def _step(batch: step.Batch, pool: np.ndarray, action: int) -> None:
    batch.actions[:, 0] = float(action)
    step.step_range_numba(batch, pool, 0, len(batch.states))


@pytest.mark.compute_large_fixture
def test_reset_draws_the_pool_world_from_the_seeded_stream() -> None:
    pool = _pool(4)
    batch = _batch(2, pool)
    # Env i seeds rand_r with i and draws once for its world.
    for i in range(2):
        rng = np.array([i], dtype=np.uint32)
        world = rand_r_numba(rng) % 4
        assert batch.states[i : i + 1].tobytes() == pool[world : world + 1].tobytes()
        assert batch.rngs[i] == rng[0]
    assert np.not_equal(batch.observations[:, 792:], 0).any()
    assert batch.masks[:, Action.NOOP].tolist() == [1, 1]
    assert np.equal(batch.rewards, 0).all()
    assert np.equal(batch.terminals, 0).all()


@pytest.mark.compute_large_fixture
def test_a_noop_step_advances_the_clock_and_the_meters_only_a_little() -> None:
    pool = _pool(1)
    batch = _batch(1, pool)
    _step(batch, pool, Action.NOOP)
    state = env_state(batch.states, 0)
    assert state.timestep == 1
    assert state.player_hunger == np.float32(1.0)
    assert state.player_health == 9.0
    assert batch.rewards[0] == 0.0
    assert batch.terminals[0] == 0.0
    assert env_stats(batch.stats, 0).episode_length_accum == 1
    assert env_stats(batch.stats, 0).steps == 1


@pytest.mark.compute_large_fixture
def test_collecting_wood_pays_the_achievement_once() -> None:
    pool = _pool(1)
    batch = _batch(1, pool)
    states = batch.states
    # Put a tree in front of the player (facing up) and strike it.
    row, col = env_state(states, 0).player_position
    env_state(states, 0).map[0, row - 1, col] = BlockType.TREE
    _step(batch, pool, Action.DO)
    assert env_state(states, 0).inventory.wood == 1
    assert env_state(states, 0).achievements[Achievement.COLLECT_WOOD] == 1
    assert batch.rewards[0] == np.float32(1.0)
    _step(batch, pool, Action.NOOP)
    assert batch.rewards[0] == 0.0


@pytest.mark.compute_large_fixture
def test_sleeping_ticks_until_the_player_wakes() -> None:
    pool = _pool(1)
    batch = _batch(1, pool)
    states = batch.states
    env_state(states, 0).player_energy = 8
    env_state(states, 0).player_fatigue = np.float32(-9.0)
    before = int(env_state(states, 0).timestep)
    _step(batch, pool, Action.SLEEP)
    # The sleep tick lowers fatigue past -10, restores a point of energy, and
    # with energy back at the cap the player wakes in the same step.
    assert env_state(states, 0).is_sleeping == 0
    assert env_state(states, 0).player_energy == 9
    assert env_state(states, 0).timestep - before >= 1
    assert (
        env_stats(batch.stats, 0).last_ticks == env_state(states, 0).timestep - before
    )
    assert env_state(states, 0).achievements[Achievement.WAKE_UP] == 1
    assert batch.masks[0, Action.LEFT] == 1


@pytest.mark.compute_large_fixture
def test_death_ends_the_episode_logs_it_and_restarts_from_the_pool() -> None:
    pool = _pool(3)
    batch = _batch(1, pool)
    states = batch.states
    env_state(states, 0).player_health = np.float32(0.5)
    env_state(states, 0).player_recover = np.float32(-15.5)
    env_state(states, 0).player_food = 0
    _step(batch, pool, Action.NOOP)
    assert batch.terminals[0] == 1.0
    assert batch.rewards[0] == -1.0
    log = env_stats(batch.stats, 0).log
    assert log.n == 1.0
    assert log.episode_length == 1.0
    assert log.floors[0] == 1.0
    assert env_stats(batch.stats, 0).episode_length_accum == 0
    assert env_state(states, 0).player_health == 9.0
    assert env_state(states, 0).timestep == 0
    assert any(states[0:1].tobytes() == pool[k : k + 1].tobytes() for k in range(3))


@pytest.mark.compute_large_fixture
def test_the_unlocked_reward_sums_the_map_over_the_set_bits_in_order() -> None:
    draws = np.random.default_rng(0)
    for _ in range(64):
        flags = draws.random(NUM_ACHIEVEMENTS) < 0.3
        low = sum(1 << i for i in range(64) if flags[i])
        high = sum(1 << (i - 64) for i in range(64, NUM_ACHIEVEMENTS) if flags[i])
        expected = np.float32(0.0)
        for i in range(NUM_ACHIEVEMENTS):
            if flags.item(i):
                expected = np.float32(
                    expected + np.float32(ACHIEVEMENT_REWARD_MAP.item(i)),
                )
        actual = step.unlocked_reward_numba(np.uint64(low), np.uint64(high))
        assert np.float32(actual) == expected
    assert step.unlocked_reward_numba(np.uint64(0), np.uint64(0)) == np.float32(0.0)


def _achievement_words(state: EnvState) -> tuple[np.uint64, np.uint64]:
    """Return :func:`step.achievement_bits_numba` as the ``uint64`` words ``score_numba`` takes."""
    low, high = step.achievement_bits_numba(state)
    # Numba returns them to Python as ints. Passed back as ints, ``score_numba`` reuses
    # whichever integer specialization an earlier test compiled, and an ``int64``
    # one overflows on a word with bit 63 set (order-dependent under xdist).
    return np.uint64(low), np.uint64(high)


@pytest.mark.compute_large_fixture
def test_achievement_bits_pack_every_flag_and_score_reads_both_words() -> None:
    pool = _pool(2)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    for flag in (0, 1, 63, 64, 66):
        env_state(batch.states, 0).achievements[flag] = 1
    low, high = _achievement_words(state)
    assert int(low) == (1 << 0) | (1 << 1) | (1 << 63)
    assert int(high) == (1 << 0) | (1 << 2)
    # A flag unlocked past bit 63 is paid, and one already set before is not.
    env_state(batch.states, 0).achievements[65] = 1
    reward = step.score_numba(
        state,
        env_stats(batch.stats, 0),
        low,
        high,
        0,
        np.float32(9.0),
        False,
        step.Rules(),
    )
    assert reward == ACHIEVEMENT_REWARD_MAP[65]
    low, high = _achievement_words(state)
    reward = step.score_numba(
        state,
        env_stats(batch.stats, 0),
        low,
        high,
        0,
        np.float32(9.0),
        False,
        step.Rules(),
    )
    assert reward == np.float32(0.0)
    for flag in (2, 40):
        env_state(batch.states, 0).achievements[flag] = 1
    reward = step.score_numba(
        state,
        env_stats(batch.stats, 0),
        low,
        high,
        0,
        np.float32(9.0),
        False,
        step.Rules(),
    )
    assert reward == ACHIEVEMENT_REWARD_MAP[2] + ACHIEVEMENT_REWARD_MAP[40]


@pytest.mark.compute_large_fixture
def test_play_leaves_the_ended_world_for_restart_to_replace() -> None:
    pool = _pool(3)
    batch = _batch(1, pool)
    states = batch.states
    env_state(states, 0).player_health = np.float32(0.5)
    env_state(states, 0).player_recover = np.float32(-15.5)
    env_state(states, 0).player_food = 0
    rng = batch.rngs[0:1]
    reward, done = step.play_numba(
        env_state(states, 0),
        rng,
        env_stats(batch.stats, 0),
        0,
        step.Rules(),
    )
    assert done
    assert reward == -1.0
    assert env_stats(batch.stats, 0).log.n == 1.0
    # The terminal world is still there: dead, one tick on.
    assert env_state(states, 0).player_health <= 0.0
    assert env_state(states, 0).timestep == 1
    drawn = rng.copy()
    world = rand_r_numba(drawn) % 3
    step.restart_numba(states, 0, rng, pool, step.Rules())
    assert states[0:1].tobytes() == pool[world : world + 1].tobytes()
    assert rng[0] == drawn[0]


def _dying(batch: step.Batch) -> None:
    """Set environment 0 to lose its last point of health on the next tick."""
    env_state(batch.states, 0).player_health = np.float32(0.5)
    env_state(batch.states, 0).player_recover = np.float32(-15.5)
    env_state(batch.states, 0).player_food = 0


@pytest.mark.compute_large_fixture
def test_the_original_reward_pays_a_tenth_per_health_point_and_no_death_penalty() -> (
    None
):
    pool = _pool(2)
    batch = _batch(1, pool, step.Rules(original_reward=True))
    _dying(batch)
    _step(batch, pool, Action.NOOP)
    assert batch.terminals[0] == 1.0
    # Health 0.5 -> 0 (the clamp), times 0.1 in float32: the default reward pays -1 instead.
    assert batch.rewards[0] == (np.float32(0.0) - np.float32(0.5)) * np.float32(0.1)
    batch = _batch(1, pool, step.Rules(original_reward=True))
    row, col = env_state(batch.states, 0).player_position
    env_state(batch.states, 0).map[0, row - 1, col] = BlockType.TREE
    _step(batch, pool, Action.DO)
    assert batch.rewards[0] == np.float32(1.0)


@pytest.mark.compute_large_fixture
def test_armour_gained_pays_under_the_default_reward_only() -> None:
    pool = _pool(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    low, high = _achievement_words(state)
    health = env_state(batch.states, 0).player_health
    env_state(batch.states, 0).inventory.armour[2] = 1
    rewards = [
        step.score_numba(
            state,
            env_stats(batch.stats, 0),
            low,
            high,
            0,
            health,
            False,
            rules,
        )
        for rules in (step.Rules(), step.Rules(original_reward=True))
    ]
    assert rewards == [np.float32(1.0), np.float32(0.0)]


@pytest.mark.compute_large_fixture
def test_without_the_sleep_collapse_each_tick_is_a_step() -> None:
    pool = _pool(1)
    collapsed = _batch(1, pool)
    env_state(collapsed.states, 0).player_energy = 3
    _step(collapsed, pool, Action.SLEEP)
    assert env_state(collapsed.states, 0).is_sleeping == 0
    assert env_stats(collapsed.stats, 0).last_ticks > 1
    ticking = _batch(1, pool, step.Rules(collapse_sleep=False))
    env_state(ticking.states, 0).player_energy = 3
    for timestep in (1, 2):
        _step(ticking, pool, Action.SLEEP)
        assert env_state(ticking.states, 0).is_sleeping == 1
        assert env_state(ticking.states, 0).timestep == timestep
        assert env_stats(ticking.stats, 0).last_ticks == 1
        # Asleep, the mask offers only the no-op every action plays as.
        assert ticking.masks[0, :].tolist() == [1] + [0] * (ATN_DIM - 1)


@pytest.mark.compute_large_fixture
def test_without_the_action_mask_every_action_is_offered() -> None:
    pool = _pool(1)
    batch = _batch(1, pool, step.Rules(action_mask=False))
    assert batch.masks.all()
    batch.masks[:] = 0
    _step(batch, pool, Action.NOOP)
    assert batch.masks.all()


@pytest.mark.compute_large_fixture
def test_beating_the_necromancer_ends_the_episode_only_when_asked() -> None:
    pool = _pool(1)
    for end in (False, True):
        batch = _batch(1, pool, step.Rules(end_on_boss_defeat=end))
        env_state(batch.states, 0).boss_progress = NUM_LEVELS - 1
        _step(batch, pool, Action.NOOP)
        assert batch.terminals[0] == (1.0 if end else 0.0)


@pytest.mark.compute_large_fixture
def test_the_episode_ends_at_max_timesteps() -> None:
    pool = _pool(1)
    batch = _batch(1, pool, step.Rules(max_timesteps=3))
    for expected in (0.0, 0.0, 1.0):
        _step(batch, pool, Action.NOOP)
        assert batch.terminals[0] == expected


@pytest.mark.compute_large_fixture
def test_fresh_worlds_are_generated_from_the_environment_stream() -> None:
    worlds = _pool(3)
    template = new_states(1)
    batch = _batch(3, template, step.Rules(fresh_worlds=True))
    # Environment i's stream starts at seed i, which is how world i is built.
    for i in range(3):
        assert batch.states[i : i + 1].tobytes() == worlds[i : i + 1].tobytes()
    _dying(batch)
    _step(batch, template, Action.NOOP)
    assert batch.terminals[0] == 1.0
    assert env_state(batch.states, 0).timestep == 0
    assert batch.states[0:1].tobytes() not in {
        worlds[k : k + 1].tobytes() for k in range(3)
    }
    assert template.tobytes() == new_states(1).tobytes()


def _noop_alive(batch: step.Batch, pool: np.ndarray) -> None:
    """Refill environment 0's meters, then step every environment by a no-op."""
    for meter in ("player_food", "player_drink", "player_energy"):
        batch.states[0][meter] = 9
    env_state(batch.states, 0).player_health = np.float32(9.0)
    _step(batch, pool, Action.NOOP)


@pytest.mark.compute_large_fixture
def test_a_stall_ends_the_episode_on_its_limit_step_as_a_terminal() -> None:
    """A 3-step stall ends on step 3, rewarded as the step it is, and the reset redraws."""
    pool = _pool(1)
    batch = _batch(1, pool, step.Rules(stall_limit=3))
    assert training_stats(batch.stats, 0).stall_limit == 3
    for count in (1, 2, 3):
        _noop_alive(batch, pool)
        assert batch.terminals[0] == (1.0 if count == 3 else 0.0)
    # An ordinary end: no death penalty, logged, and the next episode capped anew.
    assert batch.rewards[0] == 0.0
    assert env_stats(batch.stats, 0).log.n == 1.0
    assert env_stats(batch.stats, 0).log.episode_length == 3.0
    assert env_stats(batch.stats, 0).episode_length_accum == 0
    assert training_stats(batch.stats, 0).stall_limit == 3
    assert training_stats(batch.stats, 0).last_gain == 0


@pytest.mark.compute_large_fixture
def test_an_achievement_reward_restarts_the_stall_clock() -> None:
    pool = _pool(1)
    batch = _batch(1, pool, step.Rules(stall_limit=3))
    _noop_alive(batch, pool)
    row, col = env_state(batch.states, 0).player_position
    env_state(batch.states, 0).map[0, row - 1, col] = BlockType.TREE
    _step(batch, pool, Action.DO)
    assert batch.rewards[0] == np.float32(1.0)
    assert training_stats(batch.stats, 0).last_gain == 2
    for count in (3, 4, 5):
        _noop_alive(batch, pool)
        assert batch.terminals[0] == (1.0 if count == 5 else 0.0)


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(("permille", "limit"), [(0, 1), (1000, 0)])
def test_the_escape_draw_comes_from_its_own_stream(permille: int, limit: int) -> None:
    """Each reset draws once from the escape stream; the game's stream never moves."""
    pool = _pool(2)
    capped = _batch(2, pool, step.Rules(stall_limit=1, uncapped_permille=permille))
    plain = _batch(2, pool)
    escape = np.arange(2, dtype=np.uint32) ^ np.uint32(0x9E37_79B9)
    for i in range(2):
        drawn = escape[i : i + 1].copy()
        uncapped = rand_r_numba(drawn) % 1000 < permille
        assert training_stats(capped.stats, i).escape[0] == drawn[0]
        assert training_stats(capped.stats, i).stall_limit == (0 if uncapped else 1)
        assert training_stats(capped.stats, i).stall_limit == limit
    assert capped.states.tobytes() == plain.states.tobytes()
    assert capped.rngs.tolist() == plain.rngs.tolist()
    # A limit of 1 ends a capped episode on its first step, from its reset world.
    _step(capped, pool, Action.NOOP)
    assert capped.terminals.tolist() == [float(limit)] * 2


@pytest.mark.compute_large_fixture
def test_the_previous_action_follows_the_packed_view_and_no_action_starts_episodes() -> (
    None
):
    pool = _pool(2)
    marked = _batch(2, pool, step.Rules(previous_action=True))
    plain = _batch(2, pool)
    assert marked.observations[:, OBS_SIZE].tolist() == [NO_ACTION] * 2
    _dying(marked)
    _dying(plain)
    for batch in (marked, plain):
        _step(batch, pool, Action.UP)
    # Environment 0 died, so its observation starts a new episode.
    assert marked.terminals.tolist() == [1.0, 0.0]
    assert marked.observations[:, OBS_SIZE].tolist() == [NO_ACTION, Action.UP]
    assert np.array_equal(marked.observations[:, :OBS_SIZE], plain.observations)
    assert marked.states.tobytes() == plain.states.tobytes()


@pytest.mark.compute_large_fixture
def test_a_boss_beaten_while_asleep_ends_the_step_at_that_tick() -> None:
    """The in-loop end-on-win check stops a collapsed sleep at the tick the boss falls.

    A check after the tick loop instead would sleep on (the second case's
    ticks) and end the episode on the same step, from a later world. Natural
    play never reaches this state: the boss's progress rises only on a tick
    whose action is ``DO``, and a sleeping or resting player's every tick is
    a no-op, while a ``DO`` tick leaves the player awake. So the two checks
    end every reachable episode on the same step with the same bits.
    """
    pool = _pool(1)
    ticks: list[int] = []
    for end in (True, False):
        batch = _batch(1, pool, step.Rules(end_on_boss_defeat=end))
        # Asleep, one point of energy and one tick of fatigue short of waking.
        env_state(batch.states, 0).is_sleeping = 1
        env_state(batch.states, 0).player_energy = 8
        env_state(batch.states, 0).player_fatigue = np.float32(-9.0)
        env_state(batch.states, 0).boss_progress = NUM_LEVELS - 1
        _step(batch, pool, Action.NOOP)
        assert batch.terminals[0] == (1.0 if end else 0.0)
        ticks.append(int(env_stats(batch.stats, 0).last_ticks))
        if end:
            assert (
                batch.rewards[0]
                >= ACHIEVEMENT_REWARD_MAP[Achievement.DEFEAT_NECROMANCER]
            )
            assert env_stats(batch.stats, 0).log.achievements[
                Achievement.DEFEAT_NECROMANCER
            ]
    assert ticks[0] == 1
    assert ticks[1] > 1


@pytest.mark.compute_large_fixture
def test_a_practice_branch_ends_unlogged_where_a_natural_episode_is_logged() -> None:
    """A branch counts its steps and ends with its episode, which reaches no log."""
    pool = _pool(3)
    batch = _batch(3, pool, step.Rules(practice=True))
    batch.stats["branch"][:2] = 1
    _step(batch, pool, Action.NOOP)
    assert batch.stats["branch_steps"].tolist() == [1, 1, 0]
    for i in (0, 2):
        env_state(batch.states, i).player_health = np.float32(0.5)
        env_state(batch.states, i).player_recover = np.float32(-15.5)
        env_state(batch.states, i).player_food = 0
    _step(batch, pool, Action.NOOP)
    assert batch.terminals.tolist() == [1.0, 0.0, 1.0]
    assert batch.stats["log"]["n"].tolist() == [0.0, 0.0, 1.0]
    assert batch.stats["episode_length_accum"].tolist() == [0, 2, 0]
    assert batch.stats["branch"].tolist() == [0, 1, 0]
    assert batch.stats["branch_steps"].tolist() == [2, 2, 0]
    step.reset_range_numba(batch, pool, 0, 0, 3)
    assert batch.stats["branch"].tolist() == [0, 0, 0]
    assert batch.stats["branch_steps"].tolist() == [2, 2, 0]


def test_the_stats_record_starts_without_an_undefined_event() -> None:
    stats = new_stats(2)
    assert env_stats(stats, 1).first_undefined_step == -1
    assert stats["undefined_spawns"].tolist() == [0, 0]
    assert stats["last_ticks"].tolist() == [0, 0]


@pytest.mark.compute_large_fixture
def test_a_reset_starts_the_step_count_and_the_undefined_event_record_afresh() -> None:
    # first_undefined_step counts steps since the reset: a record kept across
    # it would name a step of the old run and hide the new run's first event.
    pool = _pool(1)
    batch = _batch(1, pool)
    _step(batch, pool, Action.NOOP)
    env_stats(batch.stats, 0).undefined_spawns = 2
    env_stats(batch.stats, 0).first_undefined_step = 0
    log = batch.stats["log"][0:1].tobytes()
    step.reset_range_numba(batch, pool, 0, 0, 1)
    assert env_stats(batch.stats, 0).steps == 0
    assert env_stats(batch.stats, 0).last_ticks == 0
    assert env_stats(batch.stats, 0).undefined_spawns == 0
    assert env_stats(batch.stats, 0).first_undefined_step == -1
    assert batch.stats["log"][0:1].tobytes() == log


@pytest.mark.compute_large_fixture
def test_step_kernels_emit_no_fma_and_no_float64() -> None:
    pool = _pool(1)
    batch = _batch(1, pool)
    _step(batch, pool, Action.NOOP)
    for kernel in (step.step_range_numba, step.reset_range_numba):
        assembly, body = kernel_inspection(kernel)
        assert not FMA_MNEMONIC.search(assembly), kernel
        assert "double" not in body, kernel
        assert "fpext" not in body, kernel


def _llvm_for(kernel: Dispatcher[Callable[..., object]], *args: object) -> str:
    """Return the LLVM of ``kernel`` compiled for ``args`` alone, run once on them."""
    fresh = jit(kernel.py_func)
    fresh(*args)
    return kernel_llvm(fresh)


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("name", ["step_range", "reset_range"])
def test_the_default_rules_compile_neither_world_generation_nor_the_symbolic_view(
    name: str,
) -> None:
    # Compiling the options' code cost every cold start 21 s, which took an env
    # test's first steps to 55 s of compile, against pytest's 60 s timeout.
    states = new_states(1)
    batch = step.Batch(
        states,
        np.zeros(1, dtype=np.uint32),
        new_stats(1),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, OBS_SIZE), dtype=np.float32),
        np.ones((1, ATN_DIM), dtype=np.uint8),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    module = (
        _llvm_for(step.step_range_numba, batch, states, 0, 1)
        if name == "step_range"
        else _llvm_for(step.reset_range_numba, batch, states, 0, 0, 1)
    )
    assert "generate_world_numba" not in module
    assert "compute_symbolic_observations_numba" not in module


@pytest.mark.compute_large_fixture
def test_the_options_compile_their_code_and_play_compiles_once() -> None:
    # The control for the test above: its search finds the options' code where
    # they are on. And the step's play, its bulk, takes the default Rules type
    # (options typed None, as Numba mangles it into the callee's name): typed
    # by the options, it compiled again for each setting.
    states = new_states(1)
    batch = step.Batch(
        states,
        np.zeros(1, dtype=np.uint32),
        new_stats(1),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, SYMBOLIC_OBS_SIZE), dtype=np.float32),
        np.ones((1, ATN_DIM), dtype=np.uint8),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        step.Rules(fresh_worlds=True, symbolic_observation=True),
    )
    rows = batch.observations.view(step.SYMBOLIC_OBSERVATION_RECORD)
    module = _llvm_for(step._step_rows_numba, batch, states, 0, 1, rows)
    assert "generate_world_numba" in module
    assert "compute_symbolic_observations_numba" in module
    plays = {match.group() for match in re.finditer(r"4step10play_numba\w*", module)}
    assert plays
    # The Rules' tail: fresh worlds, the symbolic view, the previous action and
    # the stall cap typed None, the stall draw's permille an int, practice None.
    options = "_2c_20none" * 4 + "_2c_20int64_2c_20none_29"
    assert all(name.endswith(options) for name in plays), plays


def test_the_training_view_of_a_stats_record_is_the_record_itself() -> None:
    record = from_dtype(STATS_DTYPE)
    signature = step._training_signature(None, record)
    assert signature == (record(record), step._emit_identity)
    assert step._training_signature(None, from_dtype(np.dtype(np.int64))) is None
    builder, (pointer,) = ir_builder(ir.IntType(8).as_pointer())
    context = cpu_target.target_context
    assert step._emit_identity(context, builder, record(), [pointer]) is pointer


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
