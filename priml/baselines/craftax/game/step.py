"""One environment step, in ``puf_step``'s order.

A step is one or more ticks. A tick runs the phases in a fixed order --
change floor, craft, interact, place, abilities, move, creatures move,
creatures spawn, plants grow, survival -- and while the player sleeps or
rests the step repeats the tick with a no-op action until they wake or die,
so credit assignment sees the wake, the hit or the death in one transition.
Then ``score_numba`` rewards the step and closes an ended episode into the log;
``play_numba`` is those two, and leaves an ended world in place for a caller that
reads it. ``restart_numba`` then puts a finished environment in a pool world drawn
from its own stream, and the observation and mask are written last.

Each phase is its own function, so each is checkable on its own.
``step_range_numba`` is the ``nogil`` entry point a buffer thread calls for its
share of environments, which it steps one at a time rather than as a batch
(the package docstring has the measurements).

A training recipe adds three options, each off unless ``Rules`` sets it:
the previous action's id after the packed observation; a stall cap, which
ends an episode as a terminal ``stall_limit`` decisions after its last
achievement reward, its clocks and escape stream in ``TRAINING_STATS_DTYPE``;
and practice's branch flag, under which a branch's end is not logged. The
archive practice restores from is ``archive``'s.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast

from numba.extending import intrinsic

import numba.core.types as nbtypes
import numpy as np

from priml.baselines.craftax.game import mobs
from priml.baselines.craftax.game.jit import (
    jit,
    prefetch,
    trailing_zeros,
    unliteral,
)
from priml.baselines.craftax.game.observation import (
    compute_observations_numba,
    compute_symbolic_observations_numba,
)
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.rules import (
    abilities_numba,
    craft_numba,
    equipped_armour_numba,
    grow_plants_numba,
    interact_numba,
    move_player_numba,
    place_numba,
    tick_numba,
)
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    ACTION_OBS_SIZE,
    ATN_DIM,
    DEFAULT_MAX_TIMESTEPS,
    MAP_SIZE,
    MAX_ACHIEVEMENT_RETURN,
    MOBS_DTYPE,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    NO_ACTION,
    NUM_ACHIEVEMENTS,
    NUM_LEVELS,
    OBS_ROWS,
    OBS_SIZE,
    SYMBOLIC_OBS_SIZE,
    Achievement,
    Action,
    Array1,
    Array2,
    ItemType,
    byte_offset,
)
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    generate_world_numba,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from llvmlite.ir import IRBuilder, Value
    from numba.core.base import BaseContext
    from numba.core.typing.templates import Signature
    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import (
        EnvState,
        EnvStats,
        Records,
        Records2,
        TrainingStats,
    )


class Rules(NamedTuple):
    """How a step differs from original Craftax; the defaults are PufferLib's.

    Each field is one setup or rule difference of ``docs/differences.md``;
    flipping all of them gives original Craftax's (JAX Craftax-Symbolic-v1).

    Attributes:
      original_reward: Reward the achievements unlocked plus 0.1 per point of
        health gained or lost (D1), instead of the achievements plus the
        armour gained, or -1 on death.
      collapse_sleep: Play every tick of sleep or rest within one decision,
        as PufferLib does (D4); False makes each tick a step.
      action_mask: Write the legal-action mask (D5); False writes all ones.
      end_on_boss_defeat: End the episode when the necromancer is beaten (D2).
      max_timesteps: Ticks after which the episode ends (D3).
      fresh_worlds: Start each episode in a newly generated world drawn from
        the environment's own stream (D6), not a copy from the pool; the
        pool then holds one zeroed world, the template a world is generated
        into.
      symbolic_observation: Write original Craftax's 8,268-float one-hot
        observation (D8) instead of the 843 packed floats; the batch's
        observation rows are then that wide.
      previous_action: Append the id of the action that led to the
        observation, ``NO_ACTION`` where it starts an episode, to the packed
        observation; the rows are then 844 wide. A recipe's input, not a
        difference from either game.
      stall_limit: End an episode, as a terminal, this many steps after its
        last achievement reward (a training recipe's cap). The stats are then
        ``TRAINING_STATS_DTYPE``, holding each episode's clocks.
      uncapped_permille: With the stall cap, how many episodes in 1,000 the
        draw at each reset leaves uncapped.
      practice: The stats are ``TRAINING_STATS_DTYPE`` and carry a practice
        branch flag, and an episode that ends as a branch is not logged.
      boss_fight_reward: Reward added per necromancer hit and per kill on the
        final floor (a training recipe's shaping); a step that kills the
        player earns none of it. The stall clock and the episode log count
        achievements alone, so neither sees it.

    ``fresh_worlds``, ``symbolic_observation``, ``previous_action``,
    ``stall_limit``, ``practice`` and ``boss_fight_reward`` default to None,
    which plays as off and also keeps the option's code out of the compiled
    step: each is passed on as an argument, and Numba prunes a branch on an
    argument typed None, so the step compiles neither world generation nor the
    symbolic view.
    Compiling both cost every cold start 21 s (measured on the Xeon: 34 s to
    55 s of compile for an env's first steps). False plays the same and
    compiles them.

    """

    original_reward: bool = False
    collapse_sleep: bool = True
    action_mask: bool = True
    end_on_boss_defeat: bool = False
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS
    fresh_worlds: bool | None = None
    symbolic_observation: bool | None = None
    previous_action: bool | None = None
    stall_limit: int | None = None
    uncapped_permille: int = 0
    practice: bool | None = None
    boss_fight_reward: tuple[float, float] | None = None


class Batch(NamedTuple):
    """The per-environment arrays a buffer step reads and writes, and its rules.

    Attributes:
      states: ``STATE_DTYPE [num_envs]``.
      rngs: ``uint32 [num_envs]``, each environment's ``rand_r`` state.
      stats: ``STATS_DTYPE [num_envs]``, or ``TRAINING_STATS_DTYPE`` when
        ``rules`` turns a training option on.
      actions: float32 ``[num_envs, 1]``; each is truncated to an ``int``.
      observations: float32 ``[num_envs, 843]``, rows overwritten by a step;
        wider when ``rules`` asks for another layout.
      masks: uint8 ``[num_envs, 43]``, rows overwritten by a step.
      rewards: float32 ``[num_envs]``, overwritten by a step.
      terminals: float32 ``[num_envs]``, overwritten by a step.
      rules: What the step plays by.

    """

    states: NDArray[np.void]
    rngs: NDArray[np.uint32]
    stats: NDArray[np.void]
    actions: NDArray[np.float32]
    observations: NDArray[np.float32]
    masks: NDArray[np.uint8]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.float32]
    rules: Rules = Rules()


class Stream(Protocol):
    """One environment's ``RNG_RECORD``: its ``rand_r`` state as a ``uint32 [1]``."""

    state: Array1[np.uint32]


class Row[T](Protocol):
    """One row of a batch array viewed as one record (``MASK_RECORD`` ...)."""

    values: Array1[T]


class Streams(Array1[np.uint32], Protocol):
    """``Batch.rngs`` as a kernel reads it, and views it a stream per record."""

    def view(self, dtype: np.dtype[np.void], /) -> Records[Stream]:
        """Return the array viewed as ``dtype`` records."""
        ...


class Rows[T](Array2[T], Protocol):
    """A ``[num_envs, width]`` batch array as a kernel reads it, and views it a row per record."""

    def view(self, dtype: np.dtype[np.void], /) -> Records2[Row[T]]:
        """Return the array viewed as ``dtype`` records, ``[num_envs, 1]``."""
        ...


class BatchView(Protocol):
    """A :class:`Batch` as the kernels read it."""

    @property
    def states(self) -> Records[EnvState]:
        """``Batch.states``."""
        ...

    @property
    def rngs(self) -> Streams:
        """``Batch.rngs``."""
        ...

    @property
    def stats(self) -> Records[TrainingStats]:
        """``Batch.stats``: a training field is read only under the option it belongs to."""
        ...

    @property
    def actions(self) -> Array2[np.float32]:
        """``Batch.actions``."""
        ...

    @property
    def observations(self) -> Rows[np.float32]:
        """``Batch.observations``."""
        ...

    @property
    def masks(self) -> Rows[int]:
        """``Batch.masks``."""
        ...

    @property
    def rewards(self) -> Array1[np.float32]:
        """``Batch.rewards``."""
        ...

    @property
    def terminals(self) -> Array1[np.float32]:
        """``Batch.terminals``."""
        ...

    @property
    def rules(self) -> Rules:
        """``Batch.rules``."""
        ...


RNG_RECORD: Final = np.dtype([("state", np.uint32, (1,))])
OBSERVATION_RECORD: Final = np.dtype([("values", np.float32, (OBS_SIZE,))])
SYMBOLIC_OBSERVATION_RECORD: Final = np.dtype(
    [("values", np.float32, (SYMBOLIC_OBS_SIZE,))],
)
ACTION_OBSERVATION_RECORD: Final = np.dtype(
    [("values", np.float32, (ACTION_OBS_SIZE,))],
)
MASK_RECORD: Final = np.dtype([("values", np.uint8, (ATN_DIM,))])
"""One environment's stream, observation row and mask row, each as a record.

``step_range_numba`` views the batch's arrays through these, so what it hands each
environment's kernels is a record's array field. Numba gives such a field no
meminfo, where a slice or row of the batch array carries the batch's: every
kernel boundary that row crossed was an atomic incref and decref on a counter
all of a buffer's threads share (measured: 13 of each per environment in
``step_range_numba``, 34 and 54 across the step's module).
"""

_GRID_BYTES: Final = MAP_SIZE * MAP_SIZE
_MOBS_BYTES: Final = MOBS_DTYPE.itemsize

_GRIDS: Final = tuple(byte_offset(name) for name in ("map", "item_map", "light_map"))
_CREATURES: Final = tuple(
    byte_offset(name)
    for name in (
        "melee_mobs",
        "passive_mobs",
        "ranged_mobs",
        "mob_projectiles",
        "player_projectiles",
    )
)
_PLANTS: Final = byte_offset("growing_plants_mask")
_PLANTS_BYTES: Final = byte_offset("potion_mapping") - _PLANTS

FLOOR_ACHIEVEMENTS: Final = cast(
    "Array1[int]",
    np.array(
        [
            -1,
            Achievement.ENTER_DUNGEON,
            Achievement.ENTER_GNOMISH_MINES,
            Achievement.ENTER_SEWERS,
            Achievement.ENTER_VAULT,
            Achievement.ENTER_TROLL_MINES,
            Achievement.ENTER_FIRE_REALM,
            Achievement.ENTER_ICE_REALM,
            Achievement.ENTER_GRAVEYARD,
        ],
        dtype=np.int64,
    ),
)
"""The achievement for first entering each floor; none for the surface."""

_ACHIEVEMENT_REWARDS: Final = cast("Array1[np.float32]", ACHIEVEMENT_REWARD_MAP)
"""``ACHIEVEMENT_REWARD_MAP``, typed for the kernels' element reads."""


@jit
def change_floor_numba(state: EnvState, action: int) -> None:
    """Descend or ascend a ladder the player stands on, if the action and floor allow.

    Args:
      state: The environment's world, mutated in place.
      action: The action taken this tick.

    """
    level = state.player_level
    row = state.player_position[0]
    col = state.player_position[1]
    on_down_ladder = state.item_map[level, row, col] == ItemType.LADDER_DOWN
    on_up_ladder = state.item_map[level, row, col] == ItemType.LADDER_UP
    can_move_down = (
        action == Action.DESCEND
        and on_down_ladder
        and state.monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        and level < NUM_LEVELS - 1
    )
    can_move_up = action == Action.ASCEND and on_up_ladder and level > 0
    if not (can_move_down or can_move_up):
        return
    new_level = level + (1 if can_move_down else -1)
    if can_move_down:
        state.player_position[0] = state.up_ladders[new_level, 0]
        state.player_position[1] = state.up_ladders[new_level, 1]
    else:
        state.player_position[0] = state.down_ladders[new_level, 0]
        state.player_position[1] = state.down_ladders[new_level, 1]
    state.player_level = new_level
    achievement = FLOOR_ACHIEVEMENTS[new_level]
    if achievement >= 0 and not state.achievements[achievement]:
        state.achievements[achievement] = 1
        state.player_xp += 1


@jit
def play_numba(
    state: EnvState,
    rng: Array1[np.uint32],
    stats: EnvStats,
    action: int,
    rules: Rules,
) -> tuple[np.float32, bool]:
    """Play one decision: its ticks and its score, leaving an ended world in place.

    Args:
      state: The environment's world, stepped in place.
      rng: The environment's stream, ``uint32 [1]``.
      stats: The environment's accumulators.
      action: The action, already truncated to an ``int``.
      rules: What the step plays by; ``Rules()`` is PufferLib's.

    Returns:
      reward: The step's reward.
      done: Whether the episode ended on this step; :func:`restart_numba` is then
        what the step does next.

    """
    initial_low, initial_high = achievement_bits_numba(state)
    initial_armour = equipped_armour_numba(state)
    initial_health = state.player_health
    initial_hits = state.boss_progress
    initial_kills = state.monsters_killed[NUM_LEVELS - 1]
    stats.steps += 1
    prefetch_creatures_numba(state)
    ticks = 0
    # Per environment: a batch would tick every environment until its slowest
    # sleeper woke, 22x the real tick work (the package docstring).
    while True:
        ticks += 1
        if state.is_sleeping or state.is_resting:
            action = Action.NOOP.value
        change_floor_numba(state, action)
        craft_numba(state, action)
        interact_numba(state, rng, action)
        place_numba(state, action)
        abilities_numba(state, rng, action)
        move_player_numba(state, action)
        mobs.update_mobs_numba(state, rng)
        prefetch_view_numba(state)
        undefined = mobs.spawn_mobs_numba(state, rng)
        if undefined:
            stats.undefined_spawns += undefined
            if stats.first_undefined_step < 0:
                # This step's index since the reset, 0-based, as the bits
                # comparators that stop at it count steps.
                stats.first_undefined_step = stats.steps - 1
        grow_plants_numba(state)
        stats.max_floor_accum = max(stats.max_floor_accum, state.player_level)
        done = tick_numba(state, action, rules.max_timesteps) or (
            rules.end_on_boss_defeat and state.boss_progress >= NUM_LEVELS - 1
        )
        if (
            done
            or not (state.is_sleeping or state.is_resting)
            or not rules.collapse_sleep
        ):
            break
    stats.last_ticks = ticks
    done = _stalled_numba(
        state,
        stats,
        initial_low,
        initial_high,
        done,
        rules.stall_limit,
    )
    reward = score_numba(
        state, stats, initial_low, initial_high, initial_armour, initial_health, done, rules,
    )  # fmt: skip
    reward = _boss_fight_numba(
        state, reward, initial_hits, initial_kills, rules.boss_fight_reward,
    )  # fmt: skip
    return reward, done


@jit
def achievement_bits_numba(state: EnvState) -> tuple[np.uint64, np.uint64]:
    """Pack the 67 achievement flags into two words, one bit each.

        The reward compares the flags before and after the step. The C copies
        the 67 ints; an ``np.empty`` copy here was an NRT allocation on every
        environment step (measured, 1 of the 7 per tick), and two words carry
        the same 0/1 values with none.

    Args:
      state: The environment's world.

    Returns:
      low: Flags 0..63, flag ``i`` at bit ``i``.
      high: Flags 64..66, flag ``i`` at bit ``i - 64``.

    """
    low = np.uint64(0)
    high = np.uint64(0)
    for i in range(64):
        if state.achievements[i]:
            low |= np.uint64(1) << np.uint64(i)
    for i in range(64, NUM_ACHIEVEMENTS):
        if state.achievements[i]:
            high |= np.uint64(1) << np.uint64(i - 64)
    return low, high


@jit
def score_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured); mirrors the tail of puf_step.
    state: EnvState,
    stats: EnvStats,
    initial_low: np.uint64,
    initial_high: np.uint64,
    initial_armour: int,
    initial_health: np.float32,
    done: bool,
    rules: Rules,
) -> np.float32:
    """Score the step and close an ended episode into the log.

        PufferLib's reward is the achievement value unlocked this step plus
        armour gained, or -1 on death; original Craftax's is the achievement
        value plus 0.1 per point of health gained or lost
        (``game_logic.py:3073-3079``). The log's sums are fp32 in the C's order
        and count achievements only.

    Args:
      state: The environment's world after the step.
      stats: The environment's accumulators.
      initial_low: Achievement flags 0..63 before the step, from
        :func:`achievement_bits_numba`.
      initial_high: Achievement flags 64..66 before the step, likewise.
      initial_armour: Armour equipped before the step.
      initial_health: Health before the step.
      done: Whether the step ended the episode.
      rules: Which reward.

    Returns:
      reward: The step's float32 reward.

    """
    final_low, final_high = achievement_bits_numba(state)
    achievement_reward = unlocked_reward_numba(
        final_low & ~initial_low,
        final_high & ~initial_high,
    )
    if rules.original_reward:
        reward = achievement_reward + (
            state.player_health - initial_health
        ) * np.float32(0.1)
    else:
        reward = achievement_reward + np.float32(
            equipped_armour_numba(state) - initial_armour,
        )
        if state.player_health <= np.float32(0.0):
            reward = np.float32(-1.0)

    stats.episode_return_accum += achievement_reward
    stats.episode_length_accum += 1

    if done:
        if _logged_numba(stats, rules.practice):
            log = stats.log
            unlocked = 0
            achievement_return = np.float32(0.0)
            for i in range(NUM_ACHIEVEMENTS):
                if state.achievements[i]:
                    unlocked += 1
                    achievement_return += _ACHIEVEMENT_REWARDS[i]
                    log.achievements[i] += np.float32(1.0)
            log.achievement_rate += np.float32(unlocked) / np.float32(
                NUM_ACHIEVEMENTS,
            )
            log.perf += achievement_return / MAX_ACHIEVEMENT_RETURN
            log.score += stats.episode_return_accum
            log.episode_return += stats.episode_return_accum
            log.episode_length += np.float32(stats.episode_length_accum)
            for floor in range(stats.max_floor_accum + 1):
                log.floors[floor] += np.float32(1.0)
            log.n += np.float32(1.0)

        stats.episode_return_accum = np.float32(0.0)
        stats.episode_length_accum = 0
        stats.max_floor_accum = 0
    return np.float32(reward)


@jit
def unlocked_reward_numba(low: np.uint64, high: np.uint64) -> np.float32:
    """Sum the reward of the flags newly set this step, ``low`` 0..63 and ``high`` 64..66.

        The C sums ``delta * reward[i]`` over all 67 flags. A flag only rises
        within a step, so each delta is 0 or 1, and a zero term leaves an fp32
        sum's bits alone; the set bits, visited in ascending order, give the
        same float32 as the 67-term loop.

    Args:
      low: Flags 0..63 set by this step, flag ``i`` at bit ``i``.
      high: Flags 64..66 set by this step, flag ``i`` at bit ``i - 64``.

    Returns:
      reward: The float32 sum.

    """
    total = np.float32(0.0)
    while low:
        total += np.float32(_ACHIEVEMENT_REWARDS[trailing_zeros(low)])
        low &= low - np.uint64(1)
    while high:
        total += np.float32(_ACHIEVEMENT_REWARDS[64 + trailing_zeros(high)])
        high &= high - np.uint64(1)
    return total


@jit
def restart_numba(
    states: Records[EnvState],
    index: int,
    rng: Array1[np.uint32],
    pool: Records[EnvState],
    rules: Rules,
) -> None:
    """Start ``states[index]`` over: the pool world one ``rand_r`` draw picks, or a new one.

    With ``rules.fresh_worlds`` the world is generated from the environment's
    own stream into a copy of ``pool[0]``, a zeroed world, as the C does
    without a pool (its ``generate_world_numba`` begins with a ``memset``).

    Args:
      states: The ``STATE_DTYPE`` batch.
      index: Which environment.
      rng: Its stream, advanced by the draws.
      pool: The worlds to draw from, or the zeroed template.
      rules: Whether worlds are fresh.

    """
    _restart_numba(states, index, rng, pool, rules.fresh_worlds)


@jit
def observe_numba(
    state: EnvState,
    obs: Array1[np.float32],
    mask: Array1[int],
    rules: Rules,
) -> None:
    """Write the observation in ``rules``' layout and the mask, all ones when it masks nothing.

    Args:
      state: The world after its step.
      obs: The observation row, 843, 844 or 8,268 floats as the layout needs;
        :func:`write_previous_action_numba` writes the 844th.
      mask: The mask row.
      rules: The layout and whether to mask.

    """
    _write_observation_numba(state, obs, mask, rules.symbolic_observation)
    if not rules.action_mask:
        for action in range(ATN_DIM):
            mask[action] = 1


@jit
def write_previous_action_numba(
    obs: Array1[np.float32],
    action: int,
    rules: Rules,
) -> None:
    """Write ``action`` after the packed observation, if ``rules.previous_action``.

    Args:
      obs: The observation row, 844 floats when the rule is on.
      action: The id of the action that led to the observation, or
        ``NO_ACTION`` where it starts an episode.
      rules: Whether the observation carries the field.

    """
    _write_previous_action_numba(obs, action, rules.previous_action)


@jit
def step_range_numba(
    batch: BatchView,
    pool: Records[EnvState],
    start: int,
    stop: int,
) -> None:
    """Step environments ``start..stop-1`` by their ``actions`` rows, with the GIL released.

    Each environment is played, restarted from ``pool`` if its episode ended,
    and observed, as ``puf_step``.

    Args:
      batch: The per-environment arrays; observations, masks, rewards and
        terminals are overwritten for the stepped rows.
      pool: ``STATE_DTYPE [num_worlds]``, the worlds an ended episode restarts in.
      start: First environment.
      stop: One past the last.

    """
    rules = batch.rules
    _step_layout_numba(
        batch, pool, start, stop, rules.symbolic_observation, rules.previous_action,
    )  # fmt: skip


@jit
def reset_range_numba(
    batch: BatchView,
    pool: Records[EnvState],
    seed: int,
    start: int,
    stop: int,
) -> None:
    """Reset environments ``start..stop-1`` as ``puf_reset`` does.

        Each environment's stream restarts from ``seed + i``, draws one
        ``rand_r`` for its pool world, and writes its first observation. The
        episode log is kept; the accumulators, the step count and the
        undefined-spawn record start afresh.

    Args:
      batch: The per-environment arrays.
      pool: ``STATE_DTYPE [num_worlds]``.
      seed: The seed offset: environment ``i`` seeds its stream with ``seed + i``.
      start: First environment.
      stop: One past the last.

    """
    rules = batch.rules
    _reset_layout_numba(
        batch, pool, seed, start, stop, rules.symbolic_observation, rules.previous_action,
    )  # fmt: skip


@jit
def prefetch_creatures_numba(state: EnvState) -> None:
    """Ask the cache for the creature records of the player's floor and the plants.

        With a batch larger than the cache, a phase that first reads a line of
        its world waits for memory. ``play_numba`` asks for these as its step begins,
        so the creature moves and the plants find them loaded. A hint only: no
        value changes.

    Args:
      state: The world about to step.

    """
    level = state.player_level
    for creatures in _CREATURES:
        record = creatures + level * _MOBS_BYTES
        prefetch(state, record)
        prefetch(state, record + _MOBS_BYTES - 1)
    prefetch(state, _PLANTS)
    prefetch(state, _PLANTS + _PLANTS_BYTES - 1)


@jit
def prefetch_view_numba(state: EnvState) -> None:
    """Ask the cache for the grid rows the observation reads, once the player has moved.

        The rows the view spans on the three grids at the player's floor:
        measured, an observation taken right after its step costs twice one
        taken again, the difference these rows' misses. A hint only.

    Args:
      state: The world, its player moved for this tick.

    """
    level = state.player_level
    row = state.player_position[0]
    top = max(row - OBS_ROWS // 2, 0) * MAP_SIZE
    bottom = (min(row + OBS_ROWS // 2, MAP_SIZE - 1) + 1) * MAP_SIZE
    for grid in _GRIDS:
        base = grid + level * _GRID_BYTES
        for offset in range(
            base + top,
            base + bottom,
            64,
        ):  # One request per cache line.
            prefetch(state, offset)
        prefetch(state, base + bottom - 1)


@jit
def _step_rows_numba(
    batch: BatchView,
    pool: Records[EnvState],
    start: int,
    stop: int,
    observations: Records2[Row[np.float32]],
) -> None:
    """:func:`step_range_numba`'s loop, over the observations viewed as one record per row."""
    rngs = batch.rngs.view(RNG_RECORD)
    masks = batch.masks.view(MASK_RECORD)
    rules = batch.rules
    played = _played_numba(rules)
    # One environment per iteration, not a batch: see the package docstring.
    for i in range(start, stop):
        rng = rngs[i].state
        action = int(batch.actions[i, 0])
        reward, done = play_numba(batch.states[i], rng, batch.stats[i], action, played)
        _branch_step_numba(batch.stats[i], done, rules.practice)
        if done:
            restart_numba(batch.states, i, rng, pool, rules)
            _start_clocks_numba(
                batch.stats[i],
                rules.stall_limit,
                rules.uncapped_permille,
            )
        obs = observations[i, 0].values
        observe_numba(batch.states[i], obs, masks[i, 0].values, rules)
        write_previous_action_numba(obs, NO_ACTION if done else action, rules)
        batch.rewards[i] = reward
        batch.terminals[i] = np.float32(1.0) if done else np.float32(0.0)


@jit
def _reset_rows_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    batch: BatchView,
    pool: Records[EnvState],
    seed: int,
    start: int,
    stop: int,
    observations: Records2[Row[np.float32]],
) -> None:
    """:func:`reset_range_numba`'s loop, over the observations viewed as one record per row."""
    rngs = batch.rngs.view(RNG_RECORD)
    masks = batch.masks.view(MASK_RECORD)
    rules = batch.rules
    for i in range(start, stop):
        batch.rewards[i] = np.float32(0.0)
        batch.terminals[i] = np.float32(0.0)
        stats = batch.stats[i]
        stats.episode_return_accum = np.float32(0.0)
        stats.episode_length_accum = 0
        stats.max_floor_accum = 0
        stats.steps = 0
        stats.last_ticks = 0
        stats.undefined_spawns = 0
        stats.first_undefined_step = -1
        _clear_branch_numba(stats, rules.practice)
        batch.rngs[i] = np.uint32(seed + i)
        restart_numba(batch.states, i, rngs[i].state, pool, rules)
        _start_clocks_numba(stats, rules.stall_limit, rules.uncapped_permille)
        obs = observations[i, 0].values
        observe_numba(batch.states[i], obs, masks[i, 0].values, rules)
        write_previous_action_numba(obs, unliteral(NO_ACTION), rules)
        stats.max_floor_accum = max(stats.max_floor_accum, batch.states[i].player_level)


@jit
def _restart_numba(
    states: Records[EnvState],
    index: int,
    rng: Array1[np.uint32],
    pool: Records[EnvState],
    fresh: bool | None,
) -> None:
    """:func:`restart_numba` on ``Rules.fresh_worlds`` as an argument, so None prunes the generator."""
    if fresh is not None and fresh:
        states[index] = pool[0]
        generate_world_numba(
            states[index],
            rng,
            SMOOTH_LEVEL_CONFIGS,
            DUNGEON_LEVEL_CONFIGS,
        )
    else:
        states[index] = pool[rand_r_numba(rng) % pool.shape[0]]


# The cap reads the step's achievement flags where its reward is known and before the
# episode is scored: a capped step is then scored, logged and restarted like any other
# terminal, and the learner sees a termination, not a truncation to bootstrap across.
# A step that unlocks an achievement (every one pays, so its achievement reward is
# positive) moves ``last_gain`` to itself, counted in decisions as
# ``episode_length_accum`` counts them; the episode ends once ``stall_limit`` decisions
# have passed since, where ``stats.stall_limit``, this episode's draw, is not 0.
@jit
def _stalled_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    state: EnvState,
    stats: EnvStats,
    initial_low: np.uint64,
    initial_high: np.uint64,
    done: bool,
    stall_limit: int | None,
) -> bool:
    """:func:`play_numba`'s stall cap on ``Rules.stall_limit``, so None prunes the clocks."""
    if stall_limit is not None:
        clocks = _training(stats)
        low, high = achievement_bits_numba(state)
        steps = clocks.episode_length_accum + 1
        if (low & ~initial_low) | (high & ~initial_high):
            clocks.last_gain = steps
        return done or (
            clocks.stall_limit != 0 and steps - clocks.last_gain >= clocks.stall_limit
        )
    return done


# Added after the score, as PufferLib's reward plus the shaping: a step that kills the
# player keeps the death penalty's -1, whatever it hit.
@jit
def _boss_fight_numba(
    state: EnvState,
    reward: np.float32,
    initial_hits: int,
    initial_kills: int,
    boss_fight_reward: tuple[float, float] | None,
) -> np.float32:
    """:func:`play_numba`'s ``Rules.boss_fight_reward``, so None prunes it."""
    if boss_fight_reward is None:
        return reward
    if state.player_health <= np.float32(0.0):
        return reward
    hit, kill = boss_fight_reward
    hits = np.float32(state.boss_progress - initial_hits)
    kills = np.float32(state.monsters_killed[NUM_LEVELS - 1] - initial_kills)
    return reward + (np.float32(hit) * hits + np.float32(kill) * kills)


# None prunes the draw, so the escape stream is untouched without the cap. The draw is
# ``rand_r % 1000 < uncapped_permille``: the episode escapes the cap, with limit 0, with
# probability ``uncapped_permille / 1000``.
@jit
def _start_clocks_numba(
    stats: TrainingStats,
    stall_limit: int | None,
    uncapped_permille: int,
) -> None:
    """Draw a new episode's stall limit from its escape stream, and zero its last gain."""
    if stall_limit is not None:
        uncapped = rand_r_numba(stats.escape) % 1000 < uncapped_permille
        stats.stall_limit = 0 if uncapped else stall_limit
        stats.last_gain = 0


@jit
def _logged_numba(stats: EnvStats, practice: bool | None) -> bool:
    """Whether an ended episode reaches the log: always, but a practice branch never."""
    if practice is not None and practice:
        return _training(stats).branch == 0
    return True


# Row by row, where the step's shares run, so no thread walks every row's stats
# afterwards; None prunes it, as there is no flag.
@jit
def _branch_step_numba(stats: TrainingStats, done: bool, practice: bool | None) -> None:
    """Count a practice branch's scored step, and end the branch with its episode."""
    if practice is not None and practice and stats.branch:
        stats.branch_steps += 1
        if done:
            stats.branch = 0


@jit
def _clear_branch_numba(stats: TrainingStats, practice: bool | None) -> None:
    """Clear a reset environment's branch flag; None prunes it, as there is no flag."""
    if practice is not None and practice:
        stats.branch = 0


@jit
def _write_previous_action_numba(
    obs: Array1[np.float32],
    action: int,
    previous_action: bool | None,
) -> None:
    """:func:`write_previous_action_numba` on its rule, so None prunes the field."""
    if previous_action is not None and previous_action:
        obs[OBS_SIZE] = np.float32(action)


@jit
def _write_observation_numba(
    state: EnvState,
    obs: Array1[np.float32],
    mask: Array1[int],
    symbolic: bool | None,
) -> None:
    """:func:`observe_numba`'s layout choice on an argument, so None prunes the symbolic view."""
    if symbolic is not None and symbolic:
        compute_symbolic_observations_numba(state, obs, mask)
    else:
        compute_observations_numba(state, obs, mask)


@jit
def _step_layout_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    batch: BatchView,
    pool: Records[EnvState],
    start: int,
    stop: int,
    symbolic: bool | None,
    previous_action: bool | None,
) -> None:
    """:func:`step_range_numba` over the observation rows viewed in the layout the options name."""
    if symbolic is not None and symbolic:
        _step_rows_numba(
            batch, pool, start, stop, batch.observations.view(SYMBOLIC_OBSERVATION_RECORD),
        )  # fmt: skip
    elif previous_action is not None and previous_action:
        _step_rows_numba(
            batch, pool, start, stop, batch.observations.view(ACTION_OBSERVATION_RECORD),
        )  # fmt: skip
    else:
        _step_rows_numba(
            batch, pool, start, stop, batch.observations.view(OBSERVATION_RECORD),
        )  # fmt: skip


@jit
def _reset_layout_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    batch: BatchView,
    pool: Records[EnvState],
    seed: int,
    start: int,
    stop: int,
    symbolic: bool | None,
    previous_action: bool | None,
) -> None:
    """:func:`reset_range_numba` over the observation rows viewed in the layout the options name."""
    if symbolic is not None and symbolic:
        _reset_rows_numba(
            batch, pool, seed, start, stop, batch.observations.view(SYMBOLIC_OBSERVATION_RECORD),
        )  # fmt: skip
    elif previous_action is not None and previous_action:
        _reset_rows_numba(
            batch, pool, seed, start, stop, batch.observations.view(ACTION_OBSERVATION_RECORD),
        )  # fmt: skip
    else:
        _reset_rows_numba(
            batch, pool, seed, start, stop, batch.observations.view(OBSERVATION_RECORD),
        )  # fmt: skip


# A Rules typed by the options is another Numba type, so play, the bulk of the step,
# compiled again for every option setting a process met; typed None, it compiles once.
# The options play does read, the stall cap, practice's logs and the boss-fight
# reward, keep their types.
@jit
def _played_numba(rules: Rules) -> Rules:
    """``rules`` with the options :func:`play_numba` never reads set to None."""
    return Rules(
        rules.original_reward,
        rules.collapse_sleep,
        rules.action_mask,
        rules.end_on_boss_defeat,
        rules.max_timesteps,
        None,
        None,
        None,
        rules.stall_limit,
        0,
        rules.practice,
        rules.boss_fight_reward,
    )


def _training_signature(
    typingctx: object,
    stats: nbtypes.Type,
) -> tuple[Signature, Callable[..., Value]] | None:
    """Type :data:`_training` as the identity on a stats record."""
    del typingctx
    if not isinstance(stats, nbtypes.Record):
        return None
    return stats(stats), _emit_identity


def _emit_identity(
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit the record argument itself: a record is a pointer to its bytes."""
    del context, builder, signature
    return args[0]


_training = cast("Callable[[EnvStats], TrainingStats]", intrinsic(_training_signature))
"""A stats record typed ``TrainingStats``, for the checker: the identity in kernels.

A rule that reads the training fields has made the record
``TRAINING_STATS_DTYPE``, which the checker cannot see through ``EnvStats``;
``typing.cast`` would do, but Numba cannot compile it. Callable from kernels only.
"""
