"""Replay archived episodes on the port's game: token frames, snapshots, and records.

The world model reads the game as tokens, not as the policy's float
observation, which cannot hold the 0.05-HP health grid. A decision's token
frame (``token_frame_numba``) has two parts:

- ``cells``, uint8 ``[99, 8]``: the observation window's channel values as the
  game holds them (block, item plus one, lit, and one creature class per
  channel). They are written exactly as ``compute_observations_numba`` writes the
  packed observation -- the same window, light threshold, map-edge clamp and
  creature order -- so a later creature overwrites an earlier one of its own
  class in the same tile; each class has its own channel.
- ``aux``, int16 ``[51]``: the integer token of each scalar, read from the
  State: counts, tools, potions, meters (health on its 0.05-HP grid), facing,
  armour, light, flags and floor.

A value outside the schema is an error, never a clip: it means the game or the
schema changed, and a frame built from it would be silently wrong.

The archives were captured from the C game this port reproduces byte for byte,
padding included, and replayed with a C replay library, so an episode replays
here from its record alone, with that library's semantics:

- the world is the one the library's reset builds from ``world_seed``: the
  environment stream starts at the seed and ``generate_world_numba`` fills a zeroed
  State. A branch starts from its ``origin`` instead, the XOR of its start
  state's snapshot with that reset world's;
- before decisions 0, 256, 512, ... the FNV-1a hash of the State's bytes must
  equal the recorded one;
- each decision's token frame is taken from the State before its step, and its
  reward and done tokens after it;
- the last recorded hash is the State after the last decision: before the
  environment's autoreset would replace an ended world, or the live one of a
  truncated episode.

A :class:`Snapshot` is laid out as the library's: the State's bytes followed by
the uint32 environment stream, little-endian, ``SNAPSHOT_BYTES`` in all.
Snapshots the archives store therefore restore here, and snapshots taken here
restore in the library.

The token kernels and the replay kernels that inline them share this file: a
kernel's Numba cache is stamped with ``game/`` and its own file only
(``game.jit.source_stamp``), so a kernel calling another file's would keep stale
machine code after that file changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import dataclasses
import math

import numpy as np
import torch

from priml.baselines.craftax.game.jit import clampi_numba, jit
from priml.baselines.craftax.game.rules import boss_vulnerable_numba
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    INVENTORY_OBS_SIZE,
    MAP_SIZE,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PASSIVE_MOBS,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    NUM_POTIONS,
    OBS_COLS,
    OBS_ROWS,
    OBS_SIZE,
    OBS_TILE_CHANNELS,
    STATE_DTYPE,
    VISIBLE_LIGHT_THRESHOLD,
    Action,
    env_state,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, observe_numba, play_numba
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    generate_world_numba,
)
from priml.baselines.craftax.world_model.archive import Episode, Receipt
from priml.baselines.craftax.world_model.schema import (
    craftax_schema,
    number_id,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        EnvState,
        EnvStats,
        Mobs,
        Records,
    )


HASH_STRIDE: Final = 256
"""Decisions between recorded state hashes."""

MATCHED: Final = 0
"""Every recorded hash matched, and every frame is written."""

UNFINISHED: Final = 1
"""``record`` reached its decision limit before a terminal."""

REJECTED: Final = -3
"""A frame or reward token is out of schema (the C library's ``REPLAY_REJECTED``)."""

SNAPSHOT_BYTES: Final = STATE_DTYPE.itemsize + np.dtype(np.uint32).itemsize
"""The C library's ``replay_snapshot_size``: the State, then the environment stream."""

CELL_VALUES: Final = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS
"""Values in a frame's cells: 99 tiles of 8 channels."""

AUX_LOW: Final = np.array(
    [low - number_id(0) for low, _ in craftax_schema().scalar_ranges],
    dtype=np.int64,
)
"""Inclusive lower bound of each aux token, the schema's."""

AUX_HIGH: Final = np.array(
    [high - number_id(0) for _, high in craftax_schema().scalar_ranges],
    dtype=np.int64,
)
"""Inclusive upper bound of each aux token, the schema's."""

CELL_LIMIT: Final = np.array(
    [field.valid for field in craftax_schema().cell_fields],
    dtype=np.int64,
)
"""Exclusive bound of each cell channel's value, the schema's."""


class Replayable(Protocol):
    """What replay reads of an episode: its receipt, actions, hashes, and start."""

    @property
    def receipt(self) -> Receipt:
        """How to regenerate the episode."""
        ...

    @property
    def actions(self) -> torch.Tensor:
        """Executed actions, uint8 ``[T]``."""
        ...

    @property
    def hashes(self) -> torch.Tensor:
        """State hashes, as ``archive.Episode.hashes``."""
        ...

    @property
    def origin(self) -> bytes:
        """A branch's start state, as ``archive.Episode.origin``; empty otherwise."""
        ...

    @property
    def truncated(self) -> bool:
        """Whether the last decision is not terminal, as ``archive.Episode``'s."""
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Snapshot:
    """The game's whole state before one decision of an episode.

    Attributes:
      decision: The decision the state precedes.
      state: ``SNAPSHOT_BYTES`` bytes: the State, then the environment stream.

    """

    decision: int
    state: bytes


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Frames:
    """Token frames of a run of decisions, as ``archive.Episode`` holds them.

    Attributes:
      cells: The game's cell values, uint8 ``[T, 99, 8]``.
      aux: Auxiliary token values, int16 ``[T, 51]``.
      reward: Realized rewards, int16 ``[T]``.
      done: Terminal flags, bool ``[T]``.

    """

    cells: torch.Tensor
    aux: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


@jit
def fnv1a_numba(data: Array1[int]) -> np.uint64:
    """Return the C library's 64-bit ``state_hash`` of ``data``'s bytes.

    FNV-1a, except for the offset basis: the library's is 1469598103934665603, the
    standard 14695981039346656037 without its last digit. Every recorded hash
    uses it, so this does too.

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


def reset_world(world_seed: int) -> tuple[NDArray[np.void], NDArray[np.uint32]]:
    """Build the world the C library's replay reset builds from ``world_seed``.

    Args:
      world_seed: The episode's recorded seed, below 2**32.

    Returns:
      states: ``STATE_DTYPE [1]``, the reset world.
      rng: uint32 ``[1]``, the environment stream after world generation.

    Raises:
      ValueError: The seed does not fit the library's ``unsigned int``.

    """
    if world_seed < 0 or world_seed > np.iinfo(np.uint32).max:
        raise ValueError("world_seed must fit in 32 bits")
    states = new_states(1)
    rng = np.array([world_seed], dtype=np.uint32)
    generate_world_numba(
        env_state(states, 0),
        rng,
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
    )
    return states, rng


def save(states: NDArray[np.void], rng: NDArray[np.uint32]) -> bytes:
    """Return the snapshot of a one-world game: its State, then its stream."""
    return states.tobytes() + rng.astype("<u4").tobytes()


def load(snapshot: bytes) -> tuple[NDArray[np.void], NDArray[np.uint32]]:
    """Return the one-world game a snapshot holds.

    Args:
      snapshot: ``SNAPSHOT_BYTES`` bytes from :func:`save` or the C library.

    Returns:
      states: ``STATE_DTYPE [1]``.
      rng: uint32 ``[1]``.

    Raises:
      ValueError: The snapshot is not ``SNAPSHOT_BYTES`` long.

    """
    if len(snapshot) != SNAPSHOT_BYTES:
        raise ValueError(
            f"A snapshot is {SNAPSHOT_BYTES} bytes, not {len(snapshot)}.",
        )
    states = new_states(1)
    states.view(np.uint8)[:] = np.frombuffer(snapshot, np.uint8, STATE_DTYPE.itemsize)
    rng = np.frombuffer(snapshot, "<u4", 1, STATE_DTYPE.itemsize).astype(np.uint32)
    return states, rng


def xor_bytes(left: bytes, *, right: bytes) -> bytes:
    """Return the bytewise XOR of two equal-length strings.

    Args:
      left: One string.
      right: The other, as long.

    Returns:
      xored: Their bytewise XOR.

    Raises:
      ValueError: The lengths differ.

    """
    if len(left) != len(right):
        raise ValueError("Only equal lengths XOR.")
    return np.bitwise_xor(
        np.frombuffer(left, np.uint8),
        np.frombuffer(right, np.uint8),
    ).tobytes()


def initial(episode: Replayable) -> Snapshot:
    """Return the snapshot of ``episode``'s start, before decision 0.

    Args:
      episode: Archived episode; only its receipt and origin are read.

    Returns:
      snapshot: The reset world, or a branch's origin state: the base of a
        store's snapshot deltas.

    Raises:
      ValueError: The start does not match the episode's initial state hash.

    """
    return Snapshot(decision=0, state=save(*_begin(episode)))


def snapshots(episode: Replayable, *, stride: int) -> list[Snapshot]:
    """Replay ``episode`` from its start and snapshot it every ``stride`` decisions.

    Args:
      episode: Archived episode; only its receipt, actions, hashes, and start
        are read.
      stride: Decisions between snapshots, a positive multiple of 256, so the
        hash taken before each snapshot's decision checks its restore.

    Returns:
      snapshots: The state before decisions 0, ``stride``, ``2 * stride``, ...
        below the episode's length; the first is its start.

    Raises:
      ValueError: The stride is invalid, or the replay does not match one of
        the episode's hashes, which are all checked.

    """
    if stride <= 0 or stride % HASH_STRIDE:
        raise ValueError(f"Snapshot stride {stride} is not a multiple of 256.")
    states, rng = _begin(episode)
    taken: list[Snapshot] = []
    first = 0
    decisions = len(episode.actions)
    for last in [*range(stride, decisions, stride), decisions]:
        taken.append(Snapshot(decision=first, state=save(states, rng)))
        _steps(episode, states, rng, first=first, last=last, outputs=None)
        first = last
    return taken


def origin(
    episode: Replayable,
    *,
    decision: int,
    snapshot: Snapshot | None = None,
) -> bytes:
    """Return the origin of a branch of ``episode`` before ``decision``.

    Args:
      episode: The parent; every hash met on the way is checked.
      decision: The branch's first decision in the parent, below its length.
      snapshot: A snapshot of the parent at or before ``decision`` to step
        from; None steps from its start.

    Returns:
      origin: The state before ``decision`` XORed with the reset world of the
        parent's world seed, as ``archive.Episode.origin`` holds it.

    Raises:
      ValueError: The decision or snapshot is invalid, or a hash differs.

    """
    first = 0 if snapshot is None else snapshot.decision
    if decision < first or decision >= len(episode.actions):
        raise ValueError(f"Branch decision {decision} is not in the episode.")
    states, rng = _restore(episode, snapshot)
    _steps(episode, states, rng, first=first, last=decision, outputs=None)
    # The reset world alone: a parent that is itself a branch has an initial
    # hash of its origin, not of this world, so no hash is checked here.
    return xor_bytes(
        save(states, rng),
        right=save(*reset_world(episode.receipt.world_seed)),
    )


def segment(
    episode: Replayable,
    *,
    start: int,
    stop: int,
    snapshot: Snapshot | None = None,
) -> Frames:
    """Regenerate the token frames of decisions ``[start, stop)`` of ``episode``.

    Args:
      episode: Archived episode; only its receipt, actions, hashes, and start
        are read.
      start: First decision.
      stop: One past the last decision, at most the episode's length.
      snapshot: A snapshot of ``episode`` at or before ``start`` to replay
        from, on a hash decision (a multiple of 256); None replays from its
        start.

    Returns:
      frames: The regenerated frames, byte-identical to capture's.

    Raises:
      ValueError: The range or snapshot is invalid, or the replay does not
        match a hash met on the way (every multiple of 256 from the snapshot
        or start through ``stop - 1``, and the last hash when ``stop`` is the
        episode's end).

    """
    first = 0 if snapshot is None else snapshot.decision
    if start < first or start >= stop or stop > len(episode.actions):
        raise ValueError(f"Invalid segment [{start}, {stop}) from decision {first}.")
    states, rng = _restore(episode, snapshot)
    _steps(episode, states, rng, first=first, last=start, outputs=None)
    frames = _empty_frames(stop - start)
    _steps(episode, states, rng, first=start, last=stop, outputs=frames)
    return frames


def replay(episode: Episode) -> Episode:
    """Return ``episode`` with its token frames regenerated by replay.

    Args:
      episode: Archived episode; its frames are ignored.

    Returns:
      regenerated: ``episode`` with replayed cells, aux, reward, and done.

    Raises:
      ValueError: The replay does not match the episode's hashes.

    """
    frames = segment(episode, start=0, stop=len(episode.actions))
    return dataclasses.replace(
        episode,
        cells=frames.cells,
        aux=frames.aux,
        reward=frames.reward,
        done=frames.done,
    )


def verify(episode: Replayable) -> int:
    """Replay ``episode`` and check its hashes, writing no frames.

    Args:
      episode: Archived episode; its frames are ignored.

    Returns:
      status: 0 when every hash matches, ``k + 1`` when hash ``k`` is the
        first mismatch, -1 when the initial state hash differs. As the library's
        ``replay_run``, an episode that ends where its record does not, or
        does not end where it does, fails the first hash after that decision:
        ``k + 2`` for a decision in the stretch before hash ``k + 1``.

    Raises:
      ValueError: The episode is malformed.

    """
    if not len(episode.actions):
        raise ValueError("Invalid replay arguments.")
    start = _start(episode)
    if start is None:
        return -1
    status, _ = _status(
        episode,
        *start,
        first=0,
        last=len(episode.actions),
        outputs=None,
    )
    return status


def record(*, world_seed: int, sampling_seed: int, max_decisions: int) -> Episode:
    """Record one episode of uniformly random legal actions (the C library's ``replay_record``).

    Each action is drawn among the legal ones with a splitmix64 generator
    seeded by ``sampling_seed``, so a seed pair gives the library's episode.

    Args:
      world_seed: Seed of the generated world.
      sampling_seed: Seed of the action sampler, below 2**64.
      max_decisions: Longest episode to accept.

    Returns:
      episode: The complete episode with arm 0, split 0, and no summary.

    Raises:
      ValueError: The episode did not end within ``max_decisions``, or a token
        value is out of schema.

    """
    states, rng = reset_world(world_seed)
    initial_hash = int(fnv1a_numba(states.view(np.uint8)))
    frames = _empty_frames(max_decisions)
    actions = np.zeros(max_decisions, dtype=np.uint8)
    hashes = np.zeros(-(-max_decisions // HASH_STRIDE) + 1, dtype=np.uint64)
    status, decisions = _record_numba(
        states,
        states.view(np.uint8),
        rng,
        new_stats(1),
        np.uint64(sampling_seed),
        actions,
        hashes,
        *_arrays(frames),
        Rules(),
    )
    if status == REJECTED:
        raise ValueError("A token value lies outside the schema.")
    if status == UNFINISHED:
        raise ValueError(f"No terminal within {max_decisions} decisions.")
    count = int(decisions)
    return Episode(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=sampling_seed,
            initial_state_hash=initial_hash,
            arm=0,
            split=0,
        ),
        actions=torch.from_numpy(actions[:count].copy()),
        hashes=torch.from_numpy(
            hashes[: -(-count // HASH_STRIDE) + 1].view(np.int64).copy(),
        ),
        cells=frames.cells[:count].clone(),
        aux=frames.aux[:count].clone(),
        reward=frames.reward[:count].clone(),
        done=frames.done[:count].clone(),
        summary={},
    )


@jit
def token_frame_numba(state: EnvState, cells: Array1[int], aux: Array1[int]) -> int:
    """Write ``state``'s token frame (``craftax_tokens``).

    Args:
      state: The world before the decision.
      cells: uint8 ``[CELL_VALUES]``, overwritten.
      aux: int16 ``[INVENTORY_OBS_SIZE]``, overwritten.

    Returns:
      status: 0, or 1 plus the observation index (cells first, then aux) of
        the first value outside the schema; the frame is then incomplete.

    """
    _board_numba(state, cells)
    for i in range(CELL_VALUES):
        if cells[i] >= CELL_LIMIT[i % OBS_TILE_CHANNELS]:
            return 1 + i
    values = np.zeros(INVENTORY_OBS_SIZE, dtype=np.int64)
    _aux_values_numba(state, values)
    for i in range(INVENTORY_OBS_SIZE):
        if values[i] < AUX_LOW[i] or values[i] > AUX_HIGH[i]:
            return 1 + CELL_VALUES + i
        aux[i] = values[i]
    return 0


@jit
def reward_token_numba(reward: np.float32) -> int:
    """Return the reward token: the game's reward, an integer in -1 through 234.

    Args:
      reward: The step's reward.

    Returns:
      token: The reward itself, or -2 when it is not such an integer.

    """
    if math.isnan(reward) or reward < np.float32(-1.0) or reward > np.float32(234.0):
        return -2
    if reward != np.floor(reward):
        return -2
    return int(reward)


def _begin(episode: Replayable) -> tuple[NDArray[np.void], NDArray[np.uint32]]:
    """Start the episode at its reset world or origin; raise unless its hash matches."""
    start = _start(episode)
    if start is None:
        raise ValueError("Replay does not match the initial state hash.")
    return start


def _start(
    episode: Replayable,
) -> tuple[NDArray[np.void], NDArray[np.uint32]] | None:
    """Start the episode (``replay_begin``); None when its initial hash differs."""
    receipt = episode.receipt
    if (
        receipt.world_seed < 0
        or receipt.world_seed > np.iinfo(np.uint32).max
        or len(episode.origin) not in {0, SNAPSHOT_BYTES}
    ):
        raise ValueError("Invalid replay arguments.")
    states, rng = reset_world(receipt.world_seed)
    if episode.origin:
        states, rng = load(xor_bytes(save(states, rng), right=episode.origin))
    if int(fnv1a_numba(states.view(np.uint8))) != receipt.initial_state_hash:
        return None
    return states, rng


def _restore(
    episode: Replayable,
    snapshot: Snapshot | None,
) -> tuple[NDArray[np.void], NDArray[np.uint32]]:
    """Start at ``snapshot``, on a hash decision and checked by its hash, or the start."""
    if snapshot is None:
        return _begin(episode)
    if snapshot.decision % HASH_STRIDE or len(snapshot.state) != SNAPSHOT_BYTES:
        raise ValueError("The snapshot is not a hash decision of this game.")
    states, rng = load(snapshot.state)
    index = snapshot.decision // HASH_STRIDE
    if int(fnv1a_numba(states.view(np.uint8))) != int(episode.hashes[index]) % (
        1 << 64
    ):
        raise ValueError(f"The snapshot does not match hash {index} of the episode.")
    return states, rng


def _steps(
    episode: Replayable,
    states: NDArray[np.void],
    rng: NDArray[np.uint32],
    *,
    first: int,
    last: int,
    outputs: Frames | None,
) -> None:
    """Step decisions ``[first, last)``; raise on a hash or an episode-end mismatch."""
    status, ends_elsewhere = _status(
        episode,
        states,
        rng,
        first=first,
        last=last,
        outputs=outputs,
    )
    if ends_elsewhere:
        raise ValueError(
            f"Replay's episode end differs from its record's before hash {status - 1}.",
        )
    if status:
        raise ValueError(f"Replay does not match hash {status - 1} of the episode.")


def _status(
    episode: Replayable,
    states: NDArray[np.void],
    rng: NDArray[np.uint32],
    *,
    first: int,
    last: int,
    outputs: Frames | None,
) -> tuple[int, bool]:
    """Step decisions ``[first, last)``; return ``replay_run``'s status, as ``_run_numba``."""
    if first == last:
        return MATCHED, False
    actions = episode.actions.to(torch.uint8).contiguous().numpy()
    hashes = episode.hashes.to(torch.int64).contiguous().numpy().view(np.uint64)
    decisions = len(actions)
    if (
        not decisions
        or len(hashes) != -(-decisions // HASH_STRIDE) + 1
        or first > last
        or last > decisions
        or actions[first:last].max() >= ATN_DIM
    ):
        raise ValueError("Invalid replay arguments.")
    empty = _empty_frames(0)
    status, ends_elsewhere = _run_numba(
        states,
        states.view(np.uint8),
        rng,
        new_stats(1),
        actions,
        hashes,
        first,
        last,
        episode.truncated,
        outputs is not None,
        *_arrays(empty if outputs is None else outputs),
        Rules(),
    )
    if status == REJECTED:
        raise ValueError("A token value lies outside the schema.")
    return int(status), bool(ends_elsewhere)


def _empty_frames(decisions: int) -> Frames:
    """Return zeroed frames for ``decisions`` decisions."""
    return Frames(
        cells=torch.zeros(
            decisions,
            OBS_ROWS * OBS_COLS,
            OBS_TILE_CHANNELS,
            dtype=torch.uint8,
        ),
        aux=torch.zeros(decisions, INVENTORY_OBS_SIZE, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=torch.zeros(decisions, dtype=torch.bool),
    )


def _arrays(frames: Frames) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return writable numpy views of ``frames``: flat cells, aux, reward, done."""
    return (
        frames.cells.numpy().reshape(len(frames.cells), CELL_VALUES),
        frames.aux.numpy(),
        frames.reward.numpy(),
        frames.done.numpy().view(np.uint8),
    )


# Returns the C library's status -- 0, ``k + 1`` when hash ``k`` is the first mismatch, the
# hash count when the last hash differs, or :data:`REJECTED` -- and whether the
# fault is instead decision ``t`` of stride ``k`` ending the episode where the record
# does not or not where it does, which the library reports as ``k + 2``. Nothing here
# observes: the step's observation writes only its own buffers, never the State.
@jit
def _run_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    states: Records[EnvState],
    state_bytes: Array1[int],
    rng: Array1[np.uint32],
    stats: Records[EnvStats],
    actions: Array1[int],
    hashes: Array1[np.uint64],
    first: int,
    last: int,
    truncated: bool,
    write: bool,
    cells: Array2[int],
    aux: Array2[int],
    reward: Array1[int],
    done: Array1[int],
    rules: Rules,
) -> tuple[int, bool]:
    """Step decisions ``[first, last)`` from the state before ``first`` (``replay_run``)."""
    decisions = len(actions)
    for t in range(first, last):
        stride = t // HASH_STRIDE
        if t % HASH_STRIDE == 0 and fnv1a_numba(state_bytes) != hashes[stride]:
            return stride + 1, False
        if (
            write
            and token_frame_numba(states[0], cells[t - first], aux[t - first]) != 0
        ):
            return REJECTED, False
        step_reward, ended = play_numba(
            states[0],
            rng,
            stats[0],
            int(actions[t]),
            rules,
        )
        if write:
            token = reward_token_numba(step_reward)
            if token == -2:
                return REJECTED, False
            reward[t - first] = token
            done[t - first] = 1 if ended else 0
        if ended != (not truncated and t == decisions - 1):
            return stride + 2, True
    # ``play_numba`` leaves an ended world in place, so the live State is the one
    # the capture hashed before its autoreset, and a truncated episode's own.
    if last == decisions and fnv1a_numba(state_bytes) != hashes[len(hashes) - 1]:
        return len(hashes), False
    return MATCHED, False


@jit
def _record_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    states: Records[EnvState],
    state_bytes: Array1[int],
    rng: Array1[np.uint32],
    stats: Records[EnvStats],
    sampler: np.uint64,
    actions: Array1[int],
    hashes: Array1[np.uint64],
    cells: Array2[int],
    aux: Array2[int],
    reward: Array1[int],
    done: Array1[int],
    rules: Rules,
) -> tuple[int, int]:
    """Play random legal actions until a terminal (``replay_record``)."""
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(ATN_DIM, dtype=np.uint8)
    legal = np.zeros(ATN_DIM, dtype=np.uint8)
    # The reset observes the world it built, which sets the first mask.
    observe_numba(states[0], observation, mask, rules)
    for t in range(len(actions)):
        if t % HASH_STRIDE == 0:
            hashes[t // HASH_STRIDE] = fnv1a_numba(state_bytes)
        if token_frame_numba(states[0], cells[t], aux[t]) != 0:
            return REJECTED, 0
        count = 0
        for action in range(ATN_DIM):
            if mask[action]:
                legal[count] = action
                count += 1
        sampler, draw = _splitmix64_numba(sampler)
        actions[t] = legal[draw % np.uint64(count)]
        step_reward, ended = play_numba(
            states[0],
            rng,
            stats[0],
            int(actions[t]),
            rules,
        )
        token = reward_token_numba(step_reward)
        if token == -2:
            return REJECTED, 0
        reward[t] = token
        done[t] = 1 if ended else 0
        if ended:
            hashes[(t + HASH_STRIDE) // HASH_STRIDE] = fnv1a_numba(state_bytes)
            return MATCHED, t + 1
        observe_numba(states[0], observation, mask, rules)
    return UNFINISHED, 0


@jit
def _splitmix64_numba(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    """Return splitmix64's advanced state and its next output."""
    state = state + np.uint64(0x9E37_79B9_7F4A_7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58_476D_1CE4_E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D0_49BB_1331_11EB)
    return state, z ^ (z >> np.uint64(31))


@jit
def _byte_numba(value: int) -> int:
    """Return ``value`` as a byte; out-of-byte values become 255, which every limit rejects."""
    return 255 if value < 0 or value > 255 else value


@jit
def _board_numba(state: EnvState, cells: Array1[int]) -> None:
    """Write the window's cells, then the creatures over them."""
    for i in range(CELL_VALUES):
        cells[i] = 0
    level = state.player_level
    row = state.player_position[0]
    col = state.player_position[1]
    row_radius = OBS_ROWS // 2
    col_radius = OBS_COLS // 2
    r0 = clampi_numba(-row, -row_radius, row_radius)
    r1 = clampi_numba(MAP_SIZE - 1 - row, -row_radius, row_radius)
    c0 = clampi_numba(-col, -col_radius, col_radius)
    c1 = clampi_numba(MAP_SIZE - 1 - col, -col_radius, col_radius)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            world_row = row + r
            world_col = col + c
            if state.light_map[level, world_row, world_col] <= VISIBLE_LIGHT_THRESHOLD:
                continue
            tile = ((r + row_radius) * OBS_COLS + (c + col_radius)) * OBS_TILE_CHANNELS
            cells[tile] = state.map[level, world_row, world_col]
            cells[tile + 1] = _byte_numba(
                state.item_map[level, world_row, world_col] + 1,
            )
            cells[tile + 2] = 1
    _mobs_numba(cells, state, state.melee_mobs[level], MAX_MELEE_MOBS, 0)
    _mobs_numba(cells, state, state.passive_mobs[level], MAX_PASSIVE_MOBS, 1)
    _mobs_numba(cells, state, state.ranged_mobs[level], MAX_RANGED_MOBS, 2)
    _mobs_numba(
        cells,
        state,
        state.mob_projectiles[level],
        MAX_MOB_PROJECTILES,
        3,
    )
    _mobs_numba(
        cells,
        state,
        state.player_projectiles[level],
        MAX_PLAYER_PROJECTILES,
        4,
    )


@jit
def _mobs_numba(
    cells: Array1[int],
    state: EnvState,
    mobs: Mobs,
    slots: int,
    channel: int,
) -> None:
    """Write ``species + 1`` for each live, lit, in-window slot of one class."""
    level = state.player_level
    for i in range(slots):
        if not mobs.mask[i]:
            continue
        world_row = mobs.position[i, 0]
        world_col = mobs.position[i, 1]
        local_row = world_row - state.player_position[0] + OBS_ROWS // 2
        local_col = world_col - state.player_position[1] + OBS_COLS // 2
        if (
            local_row < 0
            or local_row >= OBS_ROWS
            or local_col < 0
            or local_col >= OBS_COLS
        ):
            continue
        if (
            world_row < 0
            or world_row >= MAP_SIZE
            or world_col < 0
            or world_col >= MAP_SIZE
            or state.light_map[level, world_row, world_col] <= VISIBLE_LIGHT_THRESHOLD
        ):
            continue
        base = (local_row * OBS_COLS + local_col) * OBS_TILE_CHANNELS
        cells[base + 3 + channel] = _byte_numba(mobs.type_id[i] + 1)


_floor = cast("Callable[[np.float64], np.float64]", np.floor)
"""``np.floor`` on a float64, typed: numpy's stubs type a ufunc's scalar result ``Any``."""


# Play leaves the grid only by half steps, so the quarter-step margin keeps every tie
# clear of float32 rounding (``codec._encode_aux`` agrees).
@jit
def _health_numba(health: np.float32) -> int:
    """Return health's token on the 0.05-HP grid, ties toward the living side."""
    return int(_floor(np.float64(health) * 20.0 + 0.75))


@jit
def _round_half_away_numba(value: np.float64) -> int:
    """Return C's ``lround`` for a nonnegative double: halves round up."""
    whole = _floor(value)
    return int(whole) + (1 if value - whole >= 0.5 else 0)


@jit
def _aux_values_numba(state: EnvState, values: Array1[int]) -> None:
    """Write the 51 aux token values in the schema's order."""
    inventory = state.inventory
    values[0] = inventory.wood
    values[1] = inventory.stone
    values[2] = inventory.coal
    values[3] = inventory.iron
    values[4] = inventory.diamond
    values[5] = inventory.sapphire
    values[6] = inventory.ruby
    values[7] = inventory.sapling
    values[8] = inventory.torches
    values[9] = inventory.arrows
    values[10] = inventory.books
    values[11] = inventory.pickaxe
    values[12] = inventory.sword
    values[13] = state.sword_enchantment
    values[14] = state.bow_enchantment
    values[15] = inventory.bow
    n = 16
    for potion in range(NUM_POTIONS):
        values[n + potion] = inventory.potions[potion]
    n += NUM_POTIONS
    values[n] = _health_numba(state.player_health)
    values[n + 1] = state.player_food
    values[n + 2] = state.player_drink
    values[n + 3] = state.player_energy
    values[n + 4] = state.player_mana
    values[n + 5] = state.player_xp
    values[n + 6] = state.player_dexterity
    values[n + 7] = state.player_strength
    values[n + 8] = state.player_intelligence
    n += 9
    direction = state.player_direction - Action.LEFT
    for d in range(4):
        values[n + d] = 1 if d == direction else 0
    n += 4
    for a in range(4):
        values[n + a] = inventory.armour[a]
    n += 4
    for a in range(4):
        values[n + a] = state.armour_enchantments[a]
    n += 4
    level = state.player_level
    values[n] = _round_half_away_numba(np.float64(state.light_level) * 255.0)
    values[n + 1] = 1 if state.is_sleeping else 0
    values[n + 2] = 1 if state.is_resting else 0
    values[n + 3] = 1 if state.learned_spells[0] else 0
    values[n + 4] = 1 if state.learned_spells[1] else 0
    values[n + 5] = level
    values[n + 6] = (
        1 if state.monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL else 0
    )
    values[n + 7] = 1 if boss_vulnerable_numba(state) else 0
