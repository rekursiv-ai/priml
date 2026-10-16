"""Replay one recorded episode into its ghost: what the site draws of it.

One Numba kernel replays the episode from its world's reset, as
``replay._run_numba`` does, and writes per decision the player byte, every block and
item change, every achievement unlocked, an escape wherever the byte cannot say
where the player went, and every ``stride``-th decision the creatures on the
player's floor, all in ``layout``'s units. It also flags each decision's
activity: the player first enters a tile, the map changes, an achievement
unlocks, the floor changes, or the ghost ends, which the site's quiet-stretch
timeline sums over episodes (``TIMELINE``); and what the player holds changes,
or a creature or the boss is hurt or killed, which with the others marks the
decisions a win's time map keeps (``sets.PROGRESS``). The record is first checked against every state
hash it carries (``replay.verify``), and the extraction's own replay against
the last one, so a ghost is the trajectory capture played.

A sleep is one decision: the capture's rules (``Rules()``, ``collapse_sleep``)
play every tick of it in that decision's step. For each sleep the kernel also
plays the step a tick at a time, on a copy of the world taken before it, by the
same game code with ``collapse_sleep`` off (a NOOP each tick while the player
sleeps or rests, as the collapsed loop plays), and keeps every ``SLEEP_STRIDE``-th
tick's creatures on the player's floor and the tiles of that floor and floor 0
changed since the decision began. The copy must end on the very bytes, stream
and tick count the collapsed step did, or the extraction fails: the samples are
the ticks the game played.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

import dataclasses

from numpy.typing import NDArray

import numpy as np

from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.rules import (
    action_to_direction_numba,
    mobs_for_class_numba,
)
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    INVENTORY_DTYPE,
    MAP_CELLS,
    MAP_SIZE,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PASSIVE_MOBS,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    MOB_SLOTS,
    NUM_ACHIEVEMENTS,
    NUM_LEVELS,
    Achievement,
    Action,
    byte_offset,
    env_state,
    new_stats,
)
from priml.baselines.craftax.game.step import (
    Rules,
    achievement_bits_numba,
    play_numba,
)
from priml.baselines.craftax.ghosts.layout import (
    CREATURE_BYTES,
    FLOOR_CHANGE,
    MOVED,
    Events,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.replay import HASH_STRIDE


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        Array3,
        EnvState,
        EnvStats,
        Mobs,
        Records,
    )
    from priml.baselines.craftax.world_model.replay import Replayable


OUTCOMES: Final = ("death", "timeout", "win", "truncated")
"""How a ghost ends, indexed by the kernel's outcome code; a truncated record that
never won stops without an end of its own."""

NEW_TILE: Final = 1
"""Activity flag: the player stands on a tile it never stood on before."""

MAP_CHANGE: Final = 2
"""Activity flag: the decision changed a block or item."""

ACHIEVED: Final = 4
"""Activity flag: the decision unlocked an achievement."""

FLOOR_CHANGED: Final = 8
"""Activity flag: the decision changed the player's floor."""

ENDED: Final = 16
"""Activity flag: the ghost's last decision."""

HELD: Final = 32
"""Activity flag: the decision changed what the player holds (collected, crafted,
placed, shot, drank or read something)."""

FOUGHT: Final = 64
"""Activity flag: the decision hurt or killed a creature on the player's floor,
added to a floor's kill count, or advanced the boss fight."""

TIMELINE: Final = NEW_TILE | MAP_CHANGE | ACHIEVED | FLOOR_CHANGED | ENDED
"""The flags a quiet-stretch timeline counts."""

SLEEP_STRIDE: Final = 4
"""Ticks between a sleep's samples: one after its 4th tick, its 8th, and so on."""

CREATURE_SLOTS: Final = (
    MAX_MELEE_MOBS
    + MAX_PASSIVE_MOBS
    + MAX_RANGED_MOBS
    + MAX_MOB_PROJECTILES
    + MAX_PLAYER_PROJECTILES
)
"""Creatures one floor holds at most: 14."""

_INVENTORY: Final = byte_offset("inventory")
_HELD_BYTES: Final = INVENTORY_DTYPE.itemsize
_TRUNCATED: Final = 3
_UNTRACKED: Final = -1
_OFF_GRID: Final = -2
_SLEPT_ELSEWHERE: Final = -3
# A sleep's samples and change rows a decision may add at most before the
# kernel stops to grow its buffers; one sleep beyond them is a fault.
_SLEEP_ROOM: Final = 4096
# ``_Trace.counts`` slots.
_MAP: Final = 0
_ACHIEVED: Final = 1
_ESCAPES: Final = 2
_BYTES: Final = 3
_END: Final = 4
_OUTCOME: Final = 5
_SLEEPS: Final = 6
_SLEEP_SAMPLES: Final = 7
_SLEEP_BYTES: Final = 8
_SLEEP_CHANGES: Final = 9
_AT: Final = 10


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ghost:
    """One episode as the site draws it.

    Attributes:
      ordinal: Its capture ordinal, which orders a tier's ghosts.
      world_seed: The world it starts from.
      sampling_seed: Its receipt's sampling seed.
      decisions: Decisions it plays: the episode's, or through its first win.
      outcome: One of ``OUTCOMES``.
      end: Floor, row, column and facing after its last decision.
      floor_first: Per floor, the first decision before which the player
        stands on it (its length for the end state), or -1.
      achievement_return: The achievements' summed reward at its end.
      players: One player byte per decision.
      creatures: The creature samples, back to back.
      samples: int64 ``[ceil(decisions / stride) + 1]``: where each sample
        starts in ``creatures``, then where the last ends.
      events: Its map, achievement and escape rows.
      active: Per decision, its activity flags (``NEW_TILE`` to ``FOUGHT``).
      sleeps: int64 ``[n, 2]``: each sleep's decision and the ticks its step
        played, in order.
      sleep_samples: int64 ``[m + 1]``: where each sleep sample starts in
        ``sleep_creatures``, then where the last ends; a sleep of ``k`` ticks
        has ``(k - 1) // SLEEP_STRIDE``, in order.
      sleep_creatures: The sleep samples' creatures, as ``creatures``.
      sleep_changes: int64 ``[c, 6]``: per sleep sample (its index), the floor,
        row, column, block and item of each tile of the player's floor and
        floor 0 that differs from before the sleep's decision.

    """

    ordinal: int
    world_seed: int
    sampling_seed: int
    decisions: int
    outcome: str
    end: tuple[int, int, int, int]
    floor_first: tuple[int, ...]
    achievement_return: int
    players: bytes
    creatures: bytes
    samples: NDArray[np.int64]
    events: Events
    active: bytes
    sleeps: NDArray[np.int64]
    sleep_samples: NDArray[np.int64]
    sleep_creatures: bytes
    sleep_changes: NDArray[np.int64]


def extract(
    record: Replayable,
    *,
    ordinal: int,
    stride: int = 4,
    win: int = Achievement.DEFEAT_NECROMANCER.value,
) -> Ghost:
    """Replay one recorded episode and return its ghost.

    Args:
      record: An episode from its world's reset, whole or truncated.
      ordinal: Its capture ordinal.
      stride: Decisions between creature samples.
      win: The achievement whose first unlock ends the ghost as a win.

    Returns:
      ghost: The episode as the site draws it.

    Raises:
      ValueError: The record is a branch, its replay misses a state hash, a
        tile changed on a floor the extraction does not diff, a creature lies
        off its byte grid or a projectile flies off the four facings, a sleep
        played a tick at a time ends elsewhere than its step, or a whole
        record neither died, timed out nor won.

    """
    receipt = record.receipt
    if record.origin:
        raise ValueError(
            f"World seed {receipt.world_seed}: a ghost starts at its world's "
            "reset, not at a branch's origin.",
        )
    status = replay.verify(record)
    if status != replay.MATCHED:
        raise ValueError(
            f"World seed {receipt.world_seed}, sampling seed "
            f"{receipt.sampling_seed}: replay status {status} (replay.verify).",
        )
    actions = np.ascontiguousarray(record.actions.numpy(), dtype=np.uint8)
    states, rng = replay.reset_world(receipt.world_seed)
    trace = _new_trace(
        states,
        decisions=len(actions),
        stride=stride,
        sleeps=int(np.count_nonzero(np.equal(actions, Action.SLEEP.value))),
    )
    stats = new_stats(1)
    first = 0
    raw = states.view(np.uint8)
    copy = _Copy(states=states.copy(), rng=rng.copy(), stats=stats.copy())
    while first < len(actions):
        first = _trace_numba(
            states, raw, rng, stats, actions, (Rules(), Rules(collapse_sleep=False)),
            trace, copy, first, stride, win,
        )  # fmt: skip
        if first < 0:
            raise ValueError(_FAULTS[first])
        if first < len(actions):
            trace = _grown(trace)
    if int(replay.fnv1a_numba(states.view(np.uint8))) != int(record.hashes[-1]) % (
        1 << 64
    ):
        raise ValueError("The extraction's replay ends off the record's last hash.")
    return _ghost(trace, record=record, ordinal=ordinal, stride=stride)


class _Copy(NamedTuple):
    """The world, stream and stats a sleep is played on a tick at a time."""

    states: NDArray[np.void]
    rng: NDArray[np.uint32]
    stats: NDArray[np.void]


class _CopyView(Protocol):
    """A :class:`_Copy` as the kernels read it."""

    @property
    def states(self) -> Records[EnvState]:
        """``_Copy.states``."""
        ...

    @property
    def rng(self) -> Array1[np.uint32]:
        """``_Copy.rng``."""
        ...

    @property
    def stats(self) -> Records[EnvStats]:
        """``_Copy.stats``."""
        ...


class _Trace(NamedTuple):
    """The extraction kernel's outputs for one episode, and its running counts."""

    players: NDArray[np.uint8]
    creatures: NDArray[np.uint8]
    samples: NDArray[np.int64]
    map_events: NDArray[np.int64]
    achievements: NDArray[np.int64]
    escapes: NDArray[np.int64]
    floor_first: NDArray[np.int64]
    block: NDArray[np.uint8]
    item: NDArray[np.uint8]
    visited: NDArray[np.uint8]
    active: NDArray[np.uint8]
    held: NDArray[np.uint8]
    health: NDArray[np.float32]
    sleeps: NDArray[np.int64]
    sleep_samples: NDArray[np.int64]
    sleep_creatures: NDArray[np.uint8]
    sleep_changes: NDArray[np.int64]
    counts: NDArray[np.int64]


class _TraceView(Protocol):
    """A :class:`_Trace` as the kernels read it."""

    @property
    def players(self) -> Array1[int]:
        """``_Trace.players``."""
        ...

    @property
    def creatures(self) -> Array1[int]:
        """``_Trace.creatures``."""
        ...

    @property
    def samples(self) -> Array1[int]:
        """``_Trace.samples``."""
        ...

    @property
    def map_events(self) -> Array2[int]:
        """``_Trace.map_events``."""
        ...

    @property
    def achievements(self) -> Array2[int]:
        """``_Trace.achievements``."""
        ...

    @property
    def escapes(self) -> Array2[int]:
        """``_Trace.escapes``."""
        ...

    @property
    def floor_first(self) -> Array1[int]:
        """``_Trace.floor_first``."""
        ...

    @property
    def block(self) -> Array3[int]:
        """``_Trace.block``."""
        ...

    @property
    def item(self) -> Array3[int]:
        """``_Trace.item``."""
        ...

    @property
    def visited(self) -> Array3[int]:
        """``_Trace.visited``."""
        ...

    @property
    def active(self) -> Array1[int]:
        """``_Trace.active``."""
        ...

    @property
    def held(self) -> Array1[int]:
        """``_Trace.held``."""
        ...

    @property
    def health(self) -> Array1[np.float32]:
        """``_Trace.health``."""
        ...

    @property
    def sleeps(self) -> Array2[int]:
        """``_Trace.sleeps``."""
        ...

    @property
    def sleep_samples(self) -> Array1[int]:
        """``_Trace.sleep_samples``."""
        ...

    @property
    def sleep_creatures(self) -> Array1[int]:
        """``_Trace.sleep_creatures``."""
        ...

    @property
    def sleep_changes(self) -> Array2[int]:
        """``_Trace.sleep_changes``."""
        ...

    @property
    def counts(self) -> Array1[int]:
        """``_Trace.counts``."""
        ...


_FAULTS: Final = {
    _UNTRACKED: "A tile changed on a floor the extraction does not diff.",
    _OFF_GRID: (
        "A creature's species or tile does not fit its byte, or a projectile "
        "flies other than one tile left, right, up or down."
    ),
    _SLEPT_ELSEWHERE: (
        "A sleep played a tick at a time ends off the world, stream or tick "
        "count its collapsed step reached, or outgrows the sleep buffers."
    ),
}


# ``sleeps`` bounds its sleeps: the SLEEP actions it holds.
def _new_trace(
    states: NDArray[np.void],
    *,
    decisions: int,
    stride: int,
    sleeps: int = 0,
) -> _Trace:
    """Return empty outputs for an episode of ``decisions``, its maps tracked from reset."""
    samples = -(-decisions // stride)
    state = env_state(states, 0)
    floor_first = np.full(NUM_LEVELS, -1, np.int64)
    floor_first[state.player_level] = 0
    visited = np.zeros((NUM_LEVELS, MAP_SIZE, MAP_SIZE), np.uint8)
    visited[
        int(state.player_level),
        int(state.player_position[0]),
        int(state.player_position[1]),
    ] = 1
    counts = np.zeros(_AT + 4, np.int64)
    counts[_END] = -1
    return _Trace(
        players=np.zeros(decisions, np.uint8),
        creatures=np.zeros(samples * (1 + CREATURE_BYTES * CREATURE_SLOTS), np.uint8),
        samples=np.zeros(samples + 1, np.int64),
        map_events=np.zeros((3 * MAP_CELLS, 6), np.int64),
        achievements=np.zeros((NUM_ACHIEVEMENTS, 2), np.int64),
        escapes=np.zeros((decisions, 5), np.int64),
        floor_first=floor_first,
        block=np.array(state.map, dtype=np.uint8),
        item=np.array(state.item_map, dtype=np.uint8),
        visited=visited,
        active=np.zeros(decisions, np.uint8),
        held=np.zeros(_HELD_BYTES, np.uint8),
        health=np.zeros(3 * MOB_SLOTS, np.float32),
        sleeps=np.zeros((max(1, sleeps), 3), np.int64),
        sleep_samples=np.zeros(2 * _SLEEP_ROOM + 1, np.int64),
        sleep_creatures=np.zeros(
            2 * _SLEEP_ROOM * (1 + CREATURE_BYTES * CREATURE_SLOTS),
            np.uint8,
        ),
        sleep_changes=np.zeros((2 * _SLEEP_ROOM, 6), np.int64),
        counts=counts,
    )


def _grown(trace: _Trace) -> _Trace:
    """Return ``trace`` with its map events and sleep buffers twice as large."""
    return trace._replace(
        map_events=np.concatenate([trace.map_events, np.zeros_like(trace.map_events)]),
        sleep_samples=np.concatenate(
            [trace.sleep_samples, np.zeros_like(trace.sleep_samples)],
        ),
        sleep_creatures=np.concatenate(
            [trace.sleep_creatures, np.zeros_like(trace.sleep_creatures)],
        ),
        sleep_changes=np.concatenate(
            [trace.sleep_changes, np.zeros_like(trace.sleep_changes)],
        ),
    )


def _ghost(trace: _Trace, *, record: Replayable, ordinal: int, stride: int) -> Ghost:
    """Return the ghost a finished trace holds."""
    counts = trace.counts
    if counts.item(_OUTCOME) == _TRUNCATED and not record.truncated:
        raise ValueError("The episode neither died, timed out nor won.")
    decisions = counts.item(_END)
    samples = -(-decisions // stride)
    achievements = trace.achievements[: counts.item(_ACHIEVED)].copy()
    sleeps = trace.sleeps[: counts.item(_SLEEPS)]
    kept = int(np.searchsorted(sleeps[:, 0], decisions))
    held = sleeps.item(kept, 2) if kept < len(sleeps) else counts.item(_SLEEP_SAMPLES)
    changes = trace.sleep_changes[: counts.item(_SLEEP_CHANGES)]
    return Ghost(
        ordinal=ordinal,
        world_seed=record.receipt.world_seed,
        sampling_seed=record.receipt.sampling_seed,
        decisions=decisions,
        outcome=OUTCOMES[counts.item(_OUTCOME)],
        end=(
            counts.item(_AT),
            counts.item(_AT + 1),
            counts.item(_AT + 2),
            counts.item(_AT + 3),
        ),
        floor_first=tuple(trace.floor_first.item(i) for i in range(NUM_LEVELS)),
        achievement_return=int(ACHIEVEMENT_REWARD_MAP[achievements[:, 1]].sum()),
        players=trace.players[:decisions].tobytes(),
        creatures=trace.creatures[: trace.samples[samples]].tobytes(),
        samples=trace.samples[: samples + 1].copy(),
        events=Events(
            map=trace.map_events[: counts.item(_MAP)].copy(),
            achievements=achievements,
            escapes=trace.escapes[: counts.item(_ESCAPES)].copy(),
        ),
        active=trace.active[:decisions].tobytes(),
        sleeps=sleeps[:kept, :2].copy(),
        sleep_samples=trace.sleep_samples[: held + 1].copy(),
        sleep_creatures=trace.sleep_creatures[: trace.sleep_samples[held]].tobytes(),
        sleep_changes=changes[changes[:, 0] < held].copy(),
    )


# Returns the decision to resume at (the episode's length once done) or a fault. A
# decision's map changes lie on the floors it starts and ends on and on floor 0,
# where sown plants ripen wherever the player is (``rules.grow_plants_numba``); every
# 256 decisions and at the ghost's end the tracked maps are compared with all nine
# floors, so a change elsewhere fails the build instead of going missing.
# ``rules`` are the capture's and the same with ``collapse_sleep`` off.
@jit
def _trace_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    states: Records[EnvState],
    raw: Array1[int],
    rng: Array1[np.uint32],
    stats: Records[EnvStats],
    actions: Array1[int],
    rules: tuple[Rules, Rules],
    trace: _TraceView,
    copy: _CopyView,
    first: int,
    stride: int,
    win: int,
) -> int:
    """Replay decisions from ``first``, writing the ghost until it ends."""
    state = states[0]
    counts = trace.counts
    last = len(actions)
    collapsed, ticked = rules
    for t in range(first, last):
        recording = counts[_END] < 0
        if recording:
            if t % HASH_STRIDE == 0 and not _same_maps_numba(state, trace):
                return _UNTRACKED
            if (
                counts[_MAP] + 3 * MAP_CELLS > trace.map_events.shape[0]
                or counts[_SLEEP_SAMPLES] + _SLEEP_ROOM >= trace.sleep_samples.shape[0]
                or counts[_SLEEP_CHANGES] + _SLEEP_ROOM > trace.sleep_changes.shape[0]
            ):
                return t
            if t % stride == 0 and not _sample_numba(state, trace, t // stride):
                return _OFF_GRID
        level = state.player_level
        row = state.player_position[0]
        col = state.player_position[1]
        facing = state.player_direction
        action = int(actions[t])
        effective = (
            Action.NOOP.value if state.is_sleeping or state.is_resting else action
        )
        low, high = achievement_bits_numba(state)
        kills = 0
        sleeping = recording and action == Action.SLEEP.value
        if recording:
            trace.held[:] = raw[_INVENTORY : _INVENTORY + _HELD_BYTES]
            kills = _snapshot_fight_numba(state, trace, level)
        if sleeping:
            copy.states.view(np.uint8)[:] = raw
            copy.rng[:] = rng
            copy.stats.view(np.uint8)[:] = stats.view(np.uint8)
        play_numba(state, rng, stats[0], action, collapsed)
        if not recording:
            continue
        if (
            sleeping
            and stats[0].last_ticks > 1
            and not _sleep_numba(raw, rng, stats, ticked, trace, copy, t, action)
        ):
            return _SLEPT_ELSEWHERE
        _move_numba(state, trace, t, effective, level, row, col, facing)
        changes = counts[_MAP]
        unlocked = counts[_ACHIEVED]
        for floor in range(NUM_LEVELS):
            if floor == 0 or floor == level or floor == state.player_level:  # noqa: PLR1714 -- Numba cannot type `in` over a tuple of mixed int widths (measured: TypingError).
                _diff_floor_numba(state, trace, t, floor)
        won = _unlock_numba(state, trace, t, low, high, win)
        if trace.floor_first[state.player_level] < 0:
            trace.floor_first[state.player_level] = t + 1
        trace.active[t] = (
            _first_visit_numba(state, trace)
            | (MAP_CHANGE if counts[_MAP] > changes else 0)
            | (ACHIEVED if counts[_ACHIEVED] > unlocked else 0)
            | (FLOOR_CHANGED if state.player_level != level else 0)
            | (ENDED if won or t == last - 1 else 0)
            | (HELD if _held_changed_numba(raw, trace) else 0)
            | (FOUGHT if _fought_numba(state, trace, level, kills) else 0)
        )
        if won or t == last - 1:
            _end_numba(state, trace, t + 1, won, collapsed.max_timesteps)
            if not _same_maps_numba(state, trace):
                return _UNTRACKED
    return last


# ``copy`` holds the world, stream and stats before the decision, and ``raw``, ``rng``
# and ``stats`` those after its collapsed step. Returns whether the ticks end where the
# step did, in as many ticks.
@jit
def _sleep_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    raw: Array1[int],
    rng: Array1[np.uint32],
    stats: Records[EnvStats],
    ticked: Rules,
    trace: _TraceView,
    copy: _CopyView,
    t: int,
    action: int,
) -> bool:
    """Play decision ``t``'s sleep again a tick at a time on ``copy``, sampling it."""
    counts = trace.counts
    n = counts[_SLEEPS]
    state = copy.states[0]
    level = state.player_level
    trace.sleeps[n, 0] = t
    trace.sleeps[n, 2] = counts[_SLEEP_SAMPLES]
    _, done = play_numba(state, copy.rng, copy.stats[0], action, ticked)
    ticks = 1
    while not done and (state.is_sleeping or state.is_resting):
        if ticks % SLEEP_STRIDE == 0:
            k = counts[_SLEEP_SAMPLES]
            if k + 1 >= trace.sleep_samples.shape[0] - 1:
                return False
            if not _sleep_sample_numba(state, trace, k, level):
                return False
            counts[_SLEEP_SAMPLES] = k + 1
        _, done = play_numba(state, copy.rng, copy.stats[0], Action.NOOP.value, ticked)
        ticks += 1
    trace.sleeps[n, 1] = ticks
    counts[_SLEEPS] = n + 1
    return (
        ticks == stats[0].last_ticks
        and np.array_equal(copy.states.view(np.uint8), raw)
        and np.array_equal(copy.rng, rng)
    )


# A change is a tile of ``level`` or floor 0 whose block or item differs from the
# tracked map, the world before the sleep's decision. Returns False if a creature is off
# grid or the change rows run out.
@jit
def _sleep_sample_numba(state: EnvState, trace: _TraceView, k: int, level: int) -> bool:
    """Write sleep sample ``k``: the creatures on ``level`` and the tiles changed."""
    start = trace.counts[_SLEEP_BYTES]
    at = start + 1
    at = _sample_class_numba(
        trace.sleep_creatures,
        at,
        state.melee_mobs[level],
        MAX_MELEE_MOBS,
        0,
    )
    at = _sample_class_numba(
        trace.sleep_creatures,
        at,
        state.passive_mobs[level],
        MAX_PASSIVE_MOBS,
        1,
    )
    at = _sample_class_numba(
        trace.sleep_creatures,
        at,
        state.ranged_mobs[level],
        MAX_RANGED_MOBS,
        2,
    )
    at = _sample_projectiles_numba(
        trace.sleep_creatures, at, state.mob_projectiles[level], MAX_MOB_PROJECTILES, 3,
        state.mob_projectile_dirs[level],
    )  # fmt: skip
    at = _sample_projectiles_numba(
        trace.sleep_creatures, at, state.player_projectiles[level], MAX_PLAYER_PROJECTILES, 4,
        state.player_projectile_directions[level],
    )  # fmt: skip
    if at < 0:
        return False
    trace.sleep_creatures[start] = (at - start - 1) // CREATURE_BYTES
    trace.counts[_SLEEP_BYTES] = at
    trace.sleep_samples[k] = start
    trace.sleep_samples[k + 1] = at
    n = trace.counts[_SLEEP_CHANGES]
    for floor in range(NUM_LEVELS):
        if floor != 0 and floor != level:  # noqa: PLR1714 -- Numba cannot type `in` over a tuple of mixed int widths (measured: TypingError).
            continue
        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                block = state.map[floor, row, col]
                item = state.item_map[floor, row, col]
                if (
                    block == trace.block[floor, row, col]
                    and item == trace.item[floor, row, col]
                ):
                    continue
                if n >= trace.sleep_changes.shape[0]:
                    return False
                trace.sleep_changes[n, 0] = k
                trace.sleep_changes[n, 1] = floor
                trace.sleep_changes[n, 2] = row
                trace.sleep_changes[n, 3] = col
                trace.sleep_changes[n, 4] = block
                trace.sleep_changes[n, 5] = item
                n += 1
    trace.counts[_SLEEP_CHANGES] = n
    return True


@jit
def _held_changed_numba(raw: Array1[int], trace: _TraceView) -> bool:
    """Whether the inventory's bytes differ from the snapshot in ``trace.held``."""
    return not np.array_equal(raw[_INVENTORY : _INVENTORY + _HELD_BYTES], trace.held)


@jit
def _snapshot_fight_numba(state: EnvState, trace: _TraceView, level: int) -> int:
    """Record the health of ``level``'s live passive, melee and ranged creatures."""
    for mob_class in range(3):
        mobs = mobs_for_class_numba(state, level, mob_class)
        for i in range(MOB_SLOTS):
            # An empty slot snapshots as -inf, below any health it may later hold.
            alive = mobs.mask[i] != 0
            trace.health[mob_class * MOB_SLOTS + i] = (
                mobs.health[i] if alive else -np.inf
            )
    return int(state.monsters_killed.sum()) + int(state.boss_progress)


@jit
def _fought_numba(state: EnvState, trace: _TraceView, level: int, score: int) -> bool:
    """Whether a creature snapshotted on ``level`` lost health, or kills or the boss advanced."""
    if int(state.monsters_killed.sum()) + int(state.boss_progress) > score:
        return True
    for mob_class in range(3):
        mobs = mobs_for_class_numba(state, level, mob_class)
        for i in range(MOB_SLOTS):
            # A kill leaves the slot's health at or below zero (rules.attack_mob),
            # so a lethal hit reads as a loss of health too.
            if mobs.health[i] < trace.health[mob_class * MOB_SLOTS + i]:
                return True
    return False


@jit
def _first_visit_numba(state: EnvState, trace: _TraceView) -> int:
    """Mark the player's tile visited; return ``NEW_TILE`` if it was not before."""
    level = state.player_level
    row = state.player_position[0]
    col = state.player_position[1]
    if trace.visited[level, row, col]:
        return 0
    trace.visited[level, row, col] = 1
    return NEW_TILE


@jit
def _move_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    state: EnvState,
    trace: _TraceView,
    t: int,
    effective: int,
    level: int,
    row: int,
    col: int,
    facing: int,
) -> None:
    """Write decision ``t``'s player byte, and an escape where the byte cannot tell it."""
    byte = _player_byte_numba(state, effective, level, row, col, facing)
    if byte < 0:
        n = trace.counts[_ESCAPES]
        trace.escapes[n, 0] = t
        trace.escapes[n, 1] = state.player_level
        trace.escapes[n, 2] = state.player_position[0]
        trace.escapes[n, 3] = state.player_position[1]
        trace.escapes[n, 4] = state.player_direction
        trace.counts[_ESCAPES] = n + 1
        byte = effective
    trace.players[t] = byte


@jit
def _player_byte_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    state: EnvState,
    effective: int,
    level: int,
    row: int,
    col: int,
    facing: int,
) -> int:
    """Return the byte that decodes to the player's new place and facing, or -1."""
    turned = effective if Action.LEFT <= effective <= Action.DOWN else facing
    if state.player_direction != turned:
        return -1
    new_row = state.player_position[0]
    new_col = state.player_position[1]
    if state.player_level == level:
        if new_row == row and new_col == col:
            return effective
        dr, dc = action_to_direction_numba(effective)
        if (dr != 0 or dc != 0) and new_row == row + dr and new_col == col + dc:
            return effective | MOVED
        return -1
    if effective == Action.DESCEND and state.player_level == level + 1:
        ladder = state.up_ladders[state.player_level]
    elif effective == Action.ASCEND and state.player_level == level - 1:
        ladder = state.down_ladders[state.player_level]
    else:
        return -1
    if new_row == ladder[0] and new_col == ladder[1]:
        return effective | FLOOR_CHANGE
    return -1


@jit
def _diff_floor_numba(state: EnvState, trace: _TraceView, t: int, floor: int) -> None:
    """Record decision ``t``'s block and item changes on ``floor`` and track them."""
    # Most decisions change no cell: one comparison of the floor, not a scan.
    if np.array_equal(state.map[floor], trace.block[floor]) and np.array_equal(
        state.item_map[floor],
        trace.item[floor],
    ):
        return
    n = trace.counts[_MAP]
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            block = state.map[floor, row, col]
            item = state.item_map[floor, row, col]
            if (
                block == trace.block[floor, row, col]
                and item == trace.item[floor, row, col]
            ):
                continue
            trace.map_events[n, 0] = t
            trace.map_events[n, 1] = floor
            trace.map_events[n, 2] = row
            trace.map_events[n, 3] = col
            trace.map_events[n, 4] = block
            trace.map_events[n, 5] = item
            trace.block[floor, row, col] = block
            trace.item[floor, row, col] = item
            n += 1
    trace.counts[_MAP] = n


@jit
def _same_maps_numba(state: EnvState, trace: _TraceView) -> bool:
    """Whether the tracked blocks and items equal the State's on every floor."""
    return np.array_equal(state.map, trace.block) and np.array_equal(
        state.item_map,
        trace.item,
    )


@jit
def _unlock_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    state: EnvState,
    trace: _TraceView,
    t: int,
    low: np.uint64,
    high: np.uint64,
    win: int,
) -> bool:
    """Record the achievements decision ``t`` unlocked; return whether ``win`` was one."""
    won = False
    for i in range(NUM_ACHIEVEMENTS):
        word = low if i < 64 else high
        had = (word >> np.uint64(i % 64)) & np.uint64(1)
        if state.achievements[i] and not had:
            n = trace.counts[_ACHIEVED]
            trace.achievements[n, 0] = t
            trace.achievements[n, 1] = i
            trace.counts[_ACHIEVED] = n + 1
            won = won or i == win
    return won


@jit
def _end_numba(
    state: EnvState,
    trace: _TraceView,
    decisions: int,
    won: bool,
    max_timesteps: int,
) -> None:
    """Close the ghost after ``decisions``: its outcome and where it ended."""
    counts = trace.counts
    counts[_END] = decisions
    if won:
        counts[_OUTCOME] = 2
    elif state.player_health <= np.float32(0.0):
        counts[_OUTCOME] = 0
    elif state.timestep >= max_timesteps:
        counts[_OUTCOME] = 1
    else:
        counts[_OUTCOME] = _TRUNCATED
    counts[_AT] = state.player_level
    counts[_AT + 1] = state.player_position[0]
    counts[_AT + 2] = state.player_position[1]
    counts[_AT + 3] = state.player_direction


@jit
def _sample_numba(state: EnvState, trace: _TraceView, k: int) -> bool:
    """Write sample ``k``: the creatures on the player's floor; False if one is off grid."""
    start = trace.counts[_BYTES]
    trace.samples[k] = start
    level = state.player_level
    at = start + 1
    at = _sample_class_numba(
        trace.creatures,
        at,
        state.melee_mobs[level],
        MAX_MELEE_MOBS,
        0,
    )
    at = _sample_class_numba(
        trace.creatures,
        at,
        state.passive_mobs[level],
        MAX_PASSIVE_MOBS,
        1,
    )
    at = _sample_class_numba(
        trace.creatures,
        at,
        state.ranged_mobs[level],
        MAX_RANGED_MOBS,
        2,
    )
    at = _sample_projectiles_numba(
        trace.creatures, at, state.mob_projectiles[level], MAX_MOB_PROJECTILES, 3,
        state.mob_projectile_dirs[level],
    )  # fmt: skip
    at = _sample_projectiles_numba(
        trace.creatures, at, state.player_projectiles[level], MAX_PLAYER_PROJECTILES, 4,
        state.player_projectile_directions[level],
    )  # fmt: skip
    if at < 0:
        return False
    trace.creatures[start] = (at - start - 1) // CREATURE_BYTES
    trace.counts[_BYTES] = at
    trace.samples[k + 1] = at
    return True


# Returns the next free byte, or -1 if a slot is off its byte grid.
@jit
def _sample_class_numba(
    creatures: Array1[int],
    at: int,
    mobs: Mobs,
    slots: int,
    klass: int,
) -> int:
    """Append one class's live slots as (class << 4 | species, row, column, 0)."""
    if at < 0:
        return at
    for i in range(slots):
        if not mobs.mask[i]:
            continue
        species = mobs.type_id[i]
        row = mobs.position[i, 0]
        col = mobs.position[i, 1]
        if (
            species < 0
            or species > 15
            or min(row, col) < 0
            or max(row, col) >= MAP_SIZE
        ):
            return -1
        creatures[at] = klass << 4 | species
        creatures[at + 1] = row
        creatures[at + 2] = col
        creatures[at + 3] = 0
        at += CREATURE_BYTES
    return at


# Returns the next free byte, or -1 if a slot is off its byte grid or its direction
# (``directions[slot]``, (drow, dcol)) is not one tile along an axis.
@jit
def _sample_projectiles_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters.
    creatures: Array1[int],
    at: int,
    mobs: Mobs,
    slots: int,
    klass: int,
    directions: Array2[int],
) -> int:
    """Append one projectile class's live slots, each with its facing as an action id."""
    first = at
    at = _sample_class_numba(creatures, at, mobs, slots, klass)
    if at < 0:
        return at
    for i in range(slots):
        if not mobs.mask[i]:
            continue
        facing = _facing_numba(directions[i, 0], directions[i, 1])
        if facing < 0:
            return -1
        creatures[first + 3] = facing
        first += CREATURE_BYTES
    return at


@jit
def _facing_numba(drow: int, dcol: int) -> int:
    """Return the move action (LEFT, RIGHT, UP, DOWN) that steps (drow, dcol); -1 if none."""
    for action in range(Action.LEFT.value, Action.DOWN.value + 1):
        dr, dc = action_to_direction_numba(action)
        if dr == drow and dc == dcol:
            return action
    return -1
