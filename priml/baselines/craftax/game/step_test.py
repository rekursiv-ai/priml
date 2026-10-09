"""Tests for the step: its order, its ticks, its reward and its auto-reset.

Bit-exactness against PufferLib is the parity tests' job, and the compiled
step's bits are ``env_test``'s goldens'. These check the step's contract, and
each rule a short random play once had to reach by luck, on worlds built by
hand (``testing.small_world``) and with every kernel run as Python: what a
step returns, how sleep makes it tick, how an ended episode is logged and
restarted, and that each option of ``Rules`` changes what it names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llvmlite import ir
from numba.core import (
    ir as numba_ir,
    types,
)
from numba.core.analysis import dead_branch_prune
from numba.core.compiler import run_frontend
from numba.core.registry import cpu_target
from numba.np.numpy_support import from_dtype

import numpy as np
import pytest

from priml.baselines.craftax.eager import eager
from priml.baselines.craftax.game import step
from priml.baselines.craftax.game.observation import (
    compute_observations_numba,
    compute_symbolic_observations_numba,
)
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    ACTION_OBS_SIZE,
    ATN_DIM,
    MAP_SIZE,
    NO_ACTION,
    NUM_ACHIEVEMENTS,
    NUM_LEVELS,
    OBS_SIZE,
    STATS_DTYPE,
    SYMBOLIC_OBS_SIZE,
    TRAINING_STATS_DTYPE,
    Achievement,
    Action,
    BlockType,
    MobType,
    env_state,
    env_stats,
    new_states,
    new_stats,
    training_stats,
)
from priml.baselines.craftax.game.testing import (
    generate_small_world,
    ir_builder,
    small_worlds,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numba.core.dispatcher import Dispatcher

    from priml.baselines.craftax.game.state import Array1, EnvState, EnvStats


@pytest.fixture(autouse=True)
def kernels(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the step as Python: a compile costs seconds, these worlds milliseconds."""
    with eager(monkeypatch=monkeypatch):
        yield


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
    if rules.symbolic_observation is True:
        width = SYMBOLIC_OBS_SIZE
    else:
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


def test_reset_draws_the_pool_world_from_the_seeded_stream() -> None:
    pool = small_worlds(4)
    batch = _batch(3, pool)
    # Env i seeds rand_r with i and draws once for its world.
    worlds: list[int] = []
    for i in range(3):
        rng = np.array([i], dtype=np.uint32)
        world = rand_r_numba(rng) % 4
        assert batch.states[i : i + 1].tobytes() == pool[world : world + 1].tobytes()
        assert batch.rngs[i] == rng[0]
        worlds.append(int(world))
    assert len(set(worlds)) > 1, "the streams must pick apart for the check to bite"
    assert np.not_equal(batch.observations[:, 792:], 0).any()
    assert batch.masks[:, Action.NOOP].tolist() == [1, 1, 1]
    assert np.equal(batch.rewards, 0).all()
    assert np.equal(batch.terminals, 0).all()


def test_a_noop_step_advances_the_clock_and_the_meters_only_a_little() -> None:
    pool = small_worlds(1)
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


def test_collecting_wood_pays_the_achievement_once() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    _put_ahead(state, BlockType.TREE)
    _step(batch, pool, Action.DO)
    assert state.inventory.wood == 1
    assert state.achievements[Achievement.COLLECT_WOOD] == 1
    assert batch.rewards[0] == np.float32(1.0)
    _step(batch, pool, Action.NOOP)
    assert batch.rewards[0] == 0.0


def test_mining_coal_yields_one_coal_and_leaves_a_path() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    state.inventory.pickaxe = 1
    row, col = _put_ahead(state, BlockType.COAL)
    _step(batch, pool, Action.DO)
    assert state.inventory.coal == 1
    assert state.map[0, row, col] == BlockType.PATH
    assert state.achievements[Achievement.COLLECT_COAL] == 1


def test_a_strength_level_up_spends_one_point_on_one_level() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    state.player_xp = 1
    _step(batch, pool, Action.LEVEL_UP_STRENGTH)
    assert (state.player_strength, state.player_xp) == (2, 0)
    assert (state.player_dexterity, state.player_intelligence) == (1, 1)
    # Spent: a second level-up has no point to spend.
    _step(batch, pool, Action.LEVEL_UP_STRENGTH)
    assert state.player_strength == 2


@pytest.mark.parametrize(
    ("mob_class", "food", "hunger", "achievement"),
    [
        (MobType.MELEE, 2, np.float32(11.0), Achievement.DEFEAT_ZOMBIE),
        (MobType.PASSIVE, 2 + 6, np.float32(1.0), Achievement.EAT_COW),
    ],
    ids=["zombie", "cow"],
)
def test_a_kill_feeds_the_player_only_when_the_creature_is_passive(
    mob_class: MobType,
    food: int,
    hunger: np.float32,
    achievement: Achievement,
) -> None:
    """A cow eaten is 6 food and resets hunger; a zombie killed is neither."""
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    state.player_food = 2
    state.player_hunger = np.float32(10.0)
    row, col = _put_ahead(state, BlockType.SAND)
    creatures = (
        state.melee_mobs[0] if mob_class == MobType.MELEE else state.passive_mobs[0]
    )
    creatures.position[0, 0] = row
    creatures.position[0, 1] = col
    creatures.type_id[0] = 0
    creatures.mask[0] = 1
    state.mob_bits[0, row] = np.uint64(1) << np.uint64(col)
    _step(batch, pool, Action.DO)
    assert creatures.mask[0] == 0
    assert state.mob_bits[0, row] == 0
    assert state.achievements[achievement] == 1
    # The tick after the kill adds its one point of hunger.
    assert (state.player_food, state.player_hunger) == (food, hunger)
    assert state.monsters_killed[0] == 10 + (mob_class == MobType.MELEE)


@pytest.mark.parametrize(
    ("row", "col", "action"),
    [
        (0, 5, Action.UP),
        (MAP_SIZE - 1, 6, Action.DOWN),
        (7, 0, Action.LEFT),
        (8, MAP_SIZE - 1, Action.RIGHT),
    ],
    ids=["top", "bottom", "left", "right"],
)
def test_a_move_off_the_map_stays_put_and_turns_the_player(
    row: int,
    col: int,
    action: Action,
) -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    state.player_position[0] = row
    state.player_position[1] = col
    state.player_direction = Action.NOOP
    _step(batch, pool, action)
    assert list(state.player_position) == [row, col]
    assert state.player_direction == action


def test_sleeping_ticks_until_the_player_wakes() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    _drowsy(state)
    _step(batch, pool, Action.SLEEP)
    # The second tick lowers fatigue past -10 and restores a point of energy;
    # with energy back at the cap the player wakes on the third, in one step.
    assert state.is_sleeping == 0
    assert state.player_energy == 9
    assert state.timestep == 3
    assert env_stats(batch.stats, 0).last_ticks == 3
    assert state.achievements[Achievement.WAKE_UP] == 1
    assert batch.masks[0, Action.LEFT] == 1


def test_without_the_sleep_collapse_each_tick_is_a_step() -> None:
    pool = small_worlds(1)
    ticking = _batch(1, pool, step.Rules(collapse_sleep=False))
    _drowsy(env_state(ticking.states, 0))
    for timestep in (1, 2):
        _step(ticking, pool, Action.SLEEP)
        assert env_state(ticking.states, 0).is_sleeping == 1
        assert env_state(ticking.states, 0).timestep == timestep
        assert env_stats(ticking.stats, 0).last_ticks == 1
        # Asleep, the mask offers only the no-op every action plays as.
        assert ticking.masks[0, :].tolist() == [1] + [0] * (ATN_DIM - 1)


def test_death_ends_the_episode_logs_it_and_restarts_from_the_pool() -> None:
    pool = small_worlds(3)
    batch = _batch(1, pool)
    states = batch.states
    _dying(batch)
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


def test_achievement_bits_pack_every_flag_and_score_reads_both_words() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    stats = env_stats(batch.stats, 0)
    for flag in (0, 1, 63, 64, 66):
        state.achievements[flag] = 1
    low, high = step.achievement_bits_numba(state)
    assert int(low) == (1 << 0) | (1 << 1) | (1 << 63)
    assert int(high) == (1 << 0) | (1 << 2)
    # A flag unlocked past bit 63 is paid, and one already set before is not.
    state.achievements[65] = 1
    assert _score(state, stats, low, high) == ACHIEVEMENT_REWARD_MAP[65]
    low, high = step.achievement_bits_numba(state)
    assert _score(state, stats, low, high) == np.float32(0.0)
    for flag in (2, 40):
        state.achievements[flag] = 1
    assert (
        _score(state, stats, low, high)
        == ACHIEVEMENT_REWARD_MAP[2] + ACHIEVEMENT_REWARD_MAP[40]
    )


def test_play_leaves_the_ended_world_for_restart_to_replace() -> None:
    pool = small_worlds(3)
    batch = _batch(1, pool)
    states = batch.states
    _dying(batch)
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


def test_the_original_reward_pays_a_tenth_per_health_point_and_no_death_penalty() -> (
    None
):
    pool = small_worlds(2)
    batch = _batch(1, pool, step.Rules(original_reward=True))
    _dying(batch)
    _step(batch, pool, Action.NOOP)
    assert batch.terminals[0] == 1.0
    # Health 0.5 -> 0 (the clamp), times 0.1 in float32: the default reward pays -1 instead.
    assert batch.rewards[0] == (np.float32(0.0) - np.float32(0.5)) * np.float32(0.1)
    batch = _batch(1, pool, step.Rules(original_reward=True))
    _put_ahead(env_state(batch.states, 0), BlockType.TREE)
    _step(batch, pool, Action.DO)
    assert batch.rewards[0] == np.float32(1.0)


def test_armour_gained_pays_under_the_default_reward_only() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool)
    state = env_state(batch.states, 0)
    low, high = step.achievement_bits_numba(state)
    health = state.player_health
    state.inventory.armour[2] = 1
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


def test_without_the_action_mask_every_action_is_offered() -> None:
    pool = small_worlds(1)
    masked = _batch(1, pool)
    # Nothing to craft or place: the mask leaves most actions out.
    assert not masked.masks.all()
    batch = _batch(1, pool, step.Rules(action_mask=False))
    assert batch.masks.all()
    batch.masks[:] = 0
    _step(batch, pool, Action.NOOP)
    assert batch.masks.all()


def test_beating_the_necromancer_ends_the_episode_only_when_asked() -> None:
    pool = small_worlds(1)
    for end in (False, True):
        batch = _batch(1, pool, step.Rules(end_on_boss_defeat=end))
        env_state(batch.states, 0).boss_progress = NUM_LEVELS - 1
        _step(batch, pool, Action.NOOP)
        assert batch.terminals[0] == (1.0 if end else 0.0)


def test_the_episode_ends_at_max_timesteps() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool, step.Rules(max_timesteps=3))
    for expected in (0.0, 0.0, 1.0):
        _step(batch, pool, Action.NOOP)
        assert batch.terminals[0] == expected


def test_fresh_worlds_are_generated_from_the_environment_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh world is generated into a copy of the zeroed template, from the env's stream.

    The generator stands in as one draw that marks the world, so the step's
    side is checked here and world generation's own bits by ``world_gen_test``
    and ``env_test``'s exp002 golden.
    """
    monkeypatch.setattr(step, "generate_world_numba", generate_small_world)
    template = new_states(1)
    batch = _batch(3, template, step.Rules(fresh_worlds=True))
    marked = small_worlds(MAP_SIZE)
    # Environment i's stream starts at seed i, whose first draw marks world i.
    for i in range(3):
        drawn = np.array([i], dtype=np.uint32)
        world = rand_r_numba(drawn) % MAP_SIZE
        assert batch.states[i : i + 1].tobytes() == marked[world : world + 1].tobytes()
        assert batch.rngs[i] == drawn[0]
    _dying(batch)
    _step(batch, template, Action.NOOP)
    assert batch.terminals[0] == 1.0
    assert env_state(batch.states, 0).timestep == 0
    assert any(
        batch.states[0:1].tobytes() == marked[k : k + 1].tobytes()
        for k in range(MAP_SIZE)
    )
    assert template.tobytes() == new_states(1).tobytes()


def test_the_symbolic_rule_writes_original_craftaxs_one_hot_view() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool, step.Rules(symbolic_observation=True))
    _step(batch, pool, Action.LEFT)
    state = env_state(batch.states, 0)
    symbolic = np.zeros(SYMBOLIC_OBS_SIZE, dtype=np.float32)
    packed = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(ATN_DIM, dtype=np.uint8)
    compute_symbolic_observations_numba(state, symbolic, mask)
    compute_observations_numba(state, packed, mask)
    assert batch.observations[0:1].tobytes() == symbolic.tobytes()
    assert batch.masks[0:1].tobytes() == mask.tobytes()
    # The 843 packed floats would be a different row: block ids, not one-hots.
    assert batch.observations[0:1, :OBS_SIZE].tobytes() != packed.tobytes()


def _noop_alive(batch: step.Batch, pool: np.ndarray) -> None:
    """Refill environment 0's meters, then step every environment by a no-op."""
    for meter in ("player_food", "player_drink", "player_energy"):
        batch.states[0][meter] = 9
    env_state(batch.states, 0).player_health = np.float32(9.0)
    _step(batch, pool, Action.NOOP)


def test_a_stall_ends_the_episode_on_its_limit_step_as_a_terminal() -> None:
    """A 3-step stall ends on step 3, rewarded as the step it is, and the reset redraws."""
    pool = small_worlds(1)
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


def test_an_achievement_reward_restarts_the_stall_clock() -> None:
    pool = small_worlds(1)
    batch = _batch(1, pool, step.Rules(stall_limit=3))
    _noop_alive(batch, pool)
    _put_ahead(env_state(batch.states, 0), BlockType.TREE)
    _step(batch, pool, Action.DO)
    assert batch.rewards[0] == np.float32(1.0)
    assert training_stats(batch.stats, 0).last_gain == 2
    for count in (3, 4, 5):
        _noop_alive(batch, pool)
        assert batch.terminals[0] == (1.0 if count == 5 else 0.0)


@pytest.mark.parametrize(("permille", "limit"), [(0, 1), (1000, 0)])
def test_the_escape_draw_comes_from_its_own_stream(permille: int, limit: int) -> None:
    """Each reset draws once from the escape stream; the game's stream never moves."""
    pool = small_worlds(2)
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


def test_the_previous_action_follows_the_packed_view_and_no_action_starts_episodes() -> (
    None
):
    pool = small_worlds(2)
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


def test_a_boss_beaten_while_asleep_ends_the_step_at_that_tick() -> None:
    """The in-loop end-on-win check stops a collapsed sleep at the tick the boss falls.

    A check after the tick loop instead would sleep on (the second case's
    ticks) and end the episode on the same step, from a later world. Natural
    play never reaches this state: the boss's progress rises only on a tick
    whose action is ``DO``, and a sleeping or resting player's every tick is
    a no-op, while a ``DO`` tick leaves the player awake. So the two checks
    end every reachable episode on the same step with the same bits.
    """
    pool = small_worlds(1)
    ticks: list[int] = []
    for end in (True, False):
        batch = _batch(1, pool, step.Rules(end_on_boss_defeat=end))
        state = env_state(batch.states, 0)
        state.is_sleeping = 1
        _drowsy(state)
        state.boss_progress = NUM_LEVELS - 1
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


def test_a_practice_branch_ends_unlogged_where_a_natural_episode_is_logged() -> None:
    """A branch counts its steps and ends with its episode, which reaches no log."""
    pool = small_worlds(3)
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


def test_a_reset_starts_the_step_count_and_the_undefined_event_record_afresh() -> None:
    # first_undefined_step counts steps since the reset: a record kept across
    # it would name a step of the old run and hide the new run's first event.
    pool = small_worlds(1)
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


@pytest.mark.parametrize(
    ("kernel", "arguments", "option", "code"),
    [
        (step._restart_numba, 5, 4, "generate_world_numba"),
        (step._write_observation_numba, 4, 3, "compute_symbolic_observations_numba"),
        (step._step_layout_numba, 6, 4, "SYMBOLIC_OBSERVATION_RECORD"),
        (step._step_layout_numba, 6, 5, "ACTION_OBSERVATION_RECORD"),
        (step._reset_layout_numba, 7, 5, "SYMBOLIC_OBSERVATION_RECORD"),
        (step._reset_layout_numba, 7, 6, "ACTION_OBSERVATION_RECORD"),
    ],
    ids=[
        "fresh-worlds",
        "symbolic-view",
        "step-symbolic-rows",
        "step-action-rows",
        "reset-symbolic-rows",
        "reset-action-rows",
    ],
)
def test_an_option_typed_none_prunes_its_code_before_the_step_compiles(
    kernel: Dispatcher[Callable[..., object]],
    arguments: int,
    option: int,
    code: str,
) -> None:
    """Numba drops an off option's branch, so the default step compiles none of it.

    Compiling world generation and the symbolic view cost every cold start
    21 s, which took an env test's first steps to 55 s of compile, against
    pytest's 60 s timeout. Numba prunes a branch on an argument typed None
    before it types the rest; its own pass runs here on the kernel's Python,
    with every argument but the option a stand-in, so nothing compiles. The
    option typed as a bool is the control: its code is still there.
    """
    for typed, kept in ((types.none, False), (types.boolean, True)):
        called: list[types.Type] = [types.int64] * arguments
        called[option] = typed
        function = run_frontend(kernel.py_func)
        dead_branch_prune(function, tuple(called))
        names = {
            statement.value.name
            for block in function.blocks.values()
            for statement in block.find_insts(numba_ir.Assign)
            if isinstance(statement.value, numba_ir.Global)
        }
        assert (code in names) is kept, (typed, sorted(names))


def test_the_options_play_never_reads_reach_it_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Play, the bulk of the step, takes ``Rules`` with the options it never reads None.

    Numba mangles an argument's type into the callee's name, so a ``Rules``
    typed by every option compiled play again for each setting a process met.
    """
    played: list[step.Rules] = []
    play = step.play_numba

    def record(
        state: EnvState,
        rng: Array1[np.uint32],
        stats: EnvStats,
        action: int,
        rules: step.Rules,
    ) -> tuple[np.float32, bool]:
        played.append(rules)
        return play(state, rng, stats, action, rules)

    monkeypatch.setattr(step, "play_numba", record)
    rules = step.Rules(
        original_reward=True,
        max_timesteps=7,
        symbolic_observation=True,
        stall_limit=5,
        uncapped_permille=9,
        practice=True,
    )
    pool = small_worlds(1)
    _step(_batch(1, pool, rules), pool, Action.NOOP)
    assert played == [
        step.Rules(
            original_reward=True,
            max_timesteps=7,
            stall_limit=5,
            practice=True,
        ),
    ]


def test_the_training_view_of_a_stats_record_is_the_record_itself() -> None:
    record = from_dtype(STATS_DTYPE)
    signature = step._training_signature(None, record)
    assert signature == (record(record), step._emit_identity)
    assert step._training_signature(None, from_dtype(np.dtype(np.int64))) is None
    builder, (pointer,) = ir_builder(ir.IntType(8).as_pointer())
    context = cpu_target.target_context
    assert step._emit_identity(context, builder, record(), [pointer]) is pointer


def _put_ahead(state: EnvState, block: BlockType) -> tuple[int, int]:
    """Write ``block`` on the tile the player faces, up; return the tile."""
    row = int(state.player_position[0]) - 1
    col = int(state.player_position[1])
    state.map[0, row, col] = block
    return row, col


def _drowsy(state: EnvState) -> None:
    """Leave the player a point of energy and a tick of fatigue short of waking."""
    state.player_energy = 8
    state.player_fatigue = np.float32(-9.0)


def _dying(batch: step.Batch) -> None:
    """Set environment 0 to lose its last point of health on the next tick."""
    env_state(batch.states, 0).player_health = np.float32(0.5)
    env_state(batch.states, 0).player_recover = np.float32(-15.5)
    env_state(batch.states, 0).player_food = 0


def _score(
    state: EnvState,
    stats: EnvStats,
    low: np.uint64,
    high: np.uint64,
) -> np.float32:
    """Score a step of ``Rules()`` that left health at 9 and armour at 0."""
    return step.score_numba(
        state, stats, low, high, 0, np.float32(9.0), False, step.Rules(),
    )  # fmt: skip


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
