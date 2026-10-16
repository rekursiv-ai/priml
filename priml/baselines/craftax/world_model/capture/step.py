"""Capture's buffer step: one ``nogil`` kernel plays and records a buffer's rows.

``record_rows_numba`` steps every row of a buffer as the game's step does -- play,
then observe -- and, for each row recording an episode, takes what capture
records of the decision before it is played (the epsilon override, the action
byte, the floor, the state hash before every 256th decision, and whether
health lies off its 0.05-HP grid) and after it (the reward, the done flag, the
clocks of the stall cap, and the achievement return). It writes into each
row's chunk, ``chunk_decisions`` long, and flags the rows that need Python:
a chunk filled, an episode ended or truncated, a failure, or a world to
reset. ``env.CaptureEnv`` handles the flagged rows after each call.

The token frames are not taken here: replay regenerates them from the record,
and checks every state hash taken here on the way (``env.py``). The hash is
FNV-1a with the archive's offset basis, ``replay.fnv1a_numba``'s, written here again
because a kernel calls only kernels of ``game/`` and of its own file: its
Numba cache is stamped with those sources alone (``game.jit``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast

import numpy as np

from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.state import ATN_DIM
from priml.baselines.craftax.game.step import (
    achievement_bits_numba,
    observe_numba,
    play_numba,
    unlocked_reward_numba,
    write_previous_action_numba,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        EnvState,
        EnvStats,
        Records,
    )
    from priml.baselines.craftax.game.step import Rules


ENDED: Final = 1
"""The row's episode ended on this decision; its world awaits a reset."""

RECORDED_END: Final = 2
"""The row's recorded episode ended or was truncated: hand it over."""

FULL: Final = 4
"""The row's chunk is full: move it into the episode before the next step."""

TOO_LONG: Final = 8
"""The row's recorded episode reached ``max_decisions``; it is dropped."""

BAD_ACTION: Final = 16
"""The row's action lies outside the action space; its episode is dropped."""

HASH_STRIDE: Final = 256
"""Decisions between recorded state hashes, ``replay.HASH_STRIDE``."""


class Rows(NamedTuple):
    """The arrays a buffer step reads and writes, one row per environment.

    Attributes:
      states: ``STATE_DTYPE [N]``, the live worlds.
      state_bytes: uint8 ``[N, state bytes]``, ``states``' bytes.
      rngs: uint32 ``[N]``, each world's stream.
      stats: ``STATS_DTYPE [N]``, each world's accumulators.
      actions: float32 ``[N, 1]``, the rollout's actions.
      masks: uint8 ``[N, 43]``, the legal-action masks.
      observations: float32 ``[N, observation size]``.
      rewards: float32 ``[N]``, each step's reward.
      terminals: float32 ``[N]``, 1 where the step ended an episode.
      recording: uint8 ``[N]``, 1 while the row records an episode.
      decisions: int64 ``[N]``, the recorded episode's decisions so far.
      fill: int64 ``[N]``, decisions in the row's chunk.
      chunk_actions: uint8 ``[N, C]``, the chunk's action bytes.
      chunk_rewards: float32 ``[N, C]``, its rewards.
      chunk_done: uint8 ``[N, C]``, its done flags.
      chunk_floors: uint8 ``[N, C]``, the floor before each decision.
      chunk_hashes: uint64 ``[N, C / 256]``, the chunk's state hashes.
      hash_fill: int64 ``[N]``, hashes in the row's chunk.
      final_hash: uint64 ``[N]``, the last hash of an episode handed over.
      epsilon: float32 ``[N]``, each episode's epsilon.
      epsilon_rng: uint64 ``[N]``, each episode's SplitMix64 override stream.
      last_gain: int64 ``[N]``, decisions up to the last positive reward.
      health_snaps: int64 ``[N]``, decisions whose health was off its grid.
      returned: float32 ``[N]``, the achievement return so far.
      split: uint8 ``[N]``, the episode's split, 1 for validation.
      flags: uint8 ``[N]``, what each row needs of Python after the step.

    """

    states: NDArray[np.void]
    state_bytes: NDArray[np.uint8]
    rngs: NDArray[np.uint32]
    stats: NDArray[np.void]
    actions: NDArray[np.float32]
    masks: NDArray[np.uint8]
    observations: NDArray[np.float32]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.float32]
    recording: NDArray[np.uint8]
    decisions: NDArray[np.int64]
    fill: NDArray[np.int64]
    chunk_actions: NDArray[np.uint8]
    chunk_rewards: NDArray[np.float32]
    chunk_done: NDArray[np.uint8]
    chunk_floors: NDArray[np.uint8]
    chunk_hashes: NDArray[np.uint64]
    hash_fill: NDArray[np.int64]
    final_hash: NDArray[np.uint64]
    epsilon: NDArray[np.float32]
    epsilon_rng: NDArray[np.uint64]
    last_gain: NDArray[np.int64]
    health_snaps: NDArray[np.int64]
    returned: NDArray[np.float32]
    split: NDArray[np.uint8]
    flags: NDArray[np.uint8]


class RowsView(Protocol):
    """A :class:`Rows` as the kernels read it."""

    @property
    def states(self) -> Records[EnvState]:
        """``Rows.states``."""
        ...

    @property
    def state_bytes(self) -> Array2[int]:
        """``Rows.state_bytes``."""
        ...

    @property
    def rngs(self) -> Array1[np.uint32]:
        """``Rows.rngs``."""
        ...

    @property
    def stats(self) -> Records[EnvStats]:
        """``Rows.stats``."""
        ...

    @property
    def actions(self) -> Array2[np.float32]:
        """``Rows.actions``."""
        ...

    @property
    def masks(self) -> Array2[int]:
        """``Rows.masks``."""
        ...

    @property
    def observations(self) -> Array2[np.float32]:
        """``Rows.observations``."""
        ...

    @property
    def rewards(self) -> Array1[np.float32]:
        """``Rows.rewards``."""
        ...

    @property
    def terminals(self) -> Array1[np.float32]:
        """``Rows.terminals``."""
        ...

    @property
    def recording(self) -> Array1[int]:
        """``Rows.recording``."""
        ...

    @property
    def decisions(self) -> Array1[int]:
        """``Rows.decisions``."""
        ...

    @property
    def fill(self) -> Array1[int]:
        """``Rows.fill``."""
        ...

    @property
    def chunk_actions(self) -> Array2[int]:
        """``Rows.chunk_actions``."""
        ...

    @property
    def chunk_rewards(self) -> Array2[np.float32]:
        """``Rows.chunk_rewards``."""
        ...

    @property
    def chunk_done(self) -> Array2[int]:
        """``Rows.chunk_done``."""
        ...

    @property
    def chunk_floors(self) -> Array2[int]:
        """``Rows.chunk_floors``."""
        ...

    @property
    def chunk_hashes(self) -> Array2[np.uint64]:
        """``Rows.chunk_hashes``."""
        ...

    @property
    def hash_fill(self) -> Array1[int]:
        """``Rows.hash_fill``."""
        ...

    @property
    def final_hash(self) -> Array1[np.uint64]:
        """``Rows.final_hash``."""
        ...

    @property
    def epsilon(self) -> Array1[np.float32]:
        """``Rows.epsilon``."""
        ...

    @property
    def epsilon_rng(self) -> Array1[np.uint64]:
        """``Rows.epsilon_rng``."""
        ...

    @property
    def last_gain(self) -> Array1[int]:
        """``Rows.last_gain``."""
        ...

    @property
    def health_snaps(self) -> Array1[int]:
        """``Rows.health_snaps``."""
        ...

    @property
    def returned(self) -> Array1[np.float32]:
        """``Rows.returned``."""
        ...

    @property
    def split(self) -> Array1[int]:
        """``Rows.split``."""
        ...

    @property
    def flags(self) -> Array1[int]:
        """``Rows.flags``."""
        ...


@jit
def record_rows_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    rows: RowsView,
    start: int,
    stop: int,
    stall_limit: int,
    max_decisions: int,
    max_timesteps: int,
    played: Rules,
    rules: Rules,
) -> None:
    """Step rows ``start..stop-1`` by their actions, recording the recorded ones.

    Args:
      rows: The arrays of every row.
      start: First row.
      stop: One past the last.
      stall_limit: Decisions without a positive reward that truncate a training
        episode; negative truncates none.
      max_decisions: Longest episode recorded.
      max_timesteps: The timeout, which a truncated episode's row is set to
        reach on its next decision.
      played: The rules the game plays, its defaults.
      rules: The rules the observation is written by.

    """
    for i in range(start, stop):
        _record_row_numba(
            rows,
            i,
            stall_limit,
            max_decisions,
            max_timesteps,
            played,
            rules,
        )


@jit
def fnv1a_numba(data: Array1[int]) -> np.uint64:
    """Return the archive's 64-bit FNV-1a hash of ``data``'s bytes (``replay.fnv1a_numba``).

    Args:
      data: uint8 bytes.

    Returns:
      digest: The hash.

    """
    value = np.uint64(1_469_598_103_934_665_603)
    prime = np.uint64(1_099_511_628_211)
    for byte in data:
        value ^= np.uint64(byte)
        value *= prime
    return value


@jit
def _record_row_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    rows: RowsView,
    i: int,
    stall_limit: int,
    max_decisions: int,
    max_timesteps: int,
    played: Rules,
    rules: Rules,
) -> None:
    """Record and play one row's decision, as ``record_rows_numba`` describes."""
    state = rows.states[i]
    flags = 0
    action = rows.actions[i, 0]
    recording = rows.recording[i] != 0
    t = rows.decisions[i]
    if recording and t >= max_decisions:
        flags |= TOO_LONG
        recording = False
    if recording and rows.epsilon[i] > np.float32(0.0):
        action = _explore_numba(rows, i, action)
    if recording and not (action >= np.float32(0.0) and action < np.float32(ATN_DIM)):
        flags |= BAD_ACTION
        recording = False
        action = np.float32(0.0)
    position = rows.fill[i]
    if recording:
        rows.chunk_actions[i, position] = np.uint8(int(action))
        rows.chunk_floors[i, position] = state.player_level
        if t % HASH_STRIDE == 0:
            rows.chunk_hashes[i, rows.hash_fill[i]] = fnv1a_numba(rows.state_bytes[i])
            rows.hash_fill[i] += 1
        scaled = np.float64(state.player_health) * 20.0
        if abs(scaled - _lround_numba(scaled)) > 0.25:
            rows.health_snaps[i] += 1
    else:
        rows.recording[i] = 0
    played_action = int(action)
    low, high = achievement_bits_numba(state)
    reward, ended = play_numba(
        state,
        rows.rngs[i : i + 1],
        rows.stats[i],
        played_action,
        played,
    )
    if recording:
        rows.chunk_rewards[i, position] = reward
        rows.chunk_done[i, position] = 1 if ended else 0
        rows.fill[i] = position + 1
        rows.decisions[i] = t + 1
        if reward > np.float32(0.0):
            rows.last_gain[i] = t + 1
        if (
            not ended
            and stall_limit >= 0
            and rows.split[i] == 0
            and t + 1 - rows.last_gain[i] >= stall_limit
        ):
            # The last hash is the State after this decision; the next decision,
            # unrecorded, ends at the timeout. The return gains nothing here: a
            # truncating decision has no positive reward.
            rows.final_hash[i] = fnv1a_numba(rows.state_bytes[i])
            state.timestep = max_timesteps - 1
            rows.recording[i] = 0
            flags |= RECORDED_END
        else:
            final_low, final_high = achievement_bits_numba(state)
            rows.returned[i] += unlocked_reward_numba(
                final_low & ~low,
                final_high & ~high,
            )
            if ended:
                rows.final_hash[i] = fnv1a_numba(rows.state_bytes[i])
                rows.recording[i] = 0
                flags |= RECORDED_END
        if rows.fill[i] == rows.chunk_actions.shape[1]:
            flags |= FULL
    rows.rewards[i] = reward
    rows.terminals[i] = np.float32(1.0) if ended else np.float32(0.0)
    if ended:
        flags |= ENDED
    else:
        observation = rows.observations[i]
        observe_numba(state, observation, rows.masks[i], rules)
        write_previous_action_numba(observation, played_action, rules)
    rows.flags[i] = flags


@jit
def _explore_numba(rows: RowsView, i: int, action: np.float32) -> np.float32:
    """Return a uniform legal action with the episode's epsilon, else ``action``."""
    draw = _splitmix64_numba(rows, i)
    if np.float64(draw >> np.uint64(11)) * 2.0**-53 >= np.float64(rows.epsilon[i]):
        return action
    count = 0
    for candidate in range(ATN_DIM):
        if rows.masks[i, candidate]:
            count += 1
    if count == 0:
        return action
    pick = _splitmix64_numba(rows, i) % np.uint64(count)
    for candidate in range(ATN_DIM):
        if rows.masks[i, candidate]:
            if pick == 0:
                return np.float32(candidate)
            pick -= np.uint64(1)
    return action


@jit
def _splitmix64_numba(rows: RowsView, i: int) -> np.uint64:
    """Advance row ``i``'s override stream and return its next output."""
    state = np.uint64(rows.epsilon_rng[i]) + np.uint64(0x9E37_79B9_7F4A_7C15)
    rows.epsilon_rng[i] = state
    z = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58_476D_1CE4_E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D0_49BB_1331_11EB)
    return z ^ (z >> np.uint64(31))


_trunc = cast("Callable[[np.float64], np.float64]", np.trunc)
"""``np.trunc`` on a float64, typed: numpy's stubs type a ufunc's scalar result ``Any``."""


@jit
def _lround_numba(value: np.float64) -> np.float64:
    """Return C's ``lround`` of ``value`` as a double: halves away from zero."""
    whole = _trunc(value)
    if abs(value - whole) >= 0.5:
        return np.float64(whole + (1.0 if value > 0 else -1.0))
    return whole
