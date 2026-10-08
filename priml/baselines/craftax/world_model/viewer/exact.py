"""Exact game bundles: a recorded episode replayed on the port's game, frame by frame.

An exact game is one the game played: a capture archive's record (world seed,
actions, a state hash every 256 decisions and after the last; ``capture/``)
replayed from its start on the port's game. Each decision becomes a 7,858-byte
frame the viewer draws, read from the game State before the decision, with its
after-values (tick, floor, health, mana, score) from the State after it; one
final frame holds the last State, under action 255. The frame keeps the
policy's observation window, the HUD, the whole 48x48 map of the player's
floor and every creature on it; a 120-byte sidecar per frame holds the current
and maximum health of each melee, passive and ranged creature, in the frame's
creature order (projectiles have none, and the player's is in the frame). The
final frame of an episode that ended is its terminal State: the game leaves an
ended world in place until a reset replaces it.

Replay checks the record: the initial State's hash, the State before every
256th decision, the State after the last, and that the episode ends at its
last decision unless the record was truncated there. A replay of
this game is exact, so a mismatch is an error, never a tolerance: the record
belongs to another game build, or to another libm (world generation and the
daylight curve call it), or it is damaged.

A sleep is one decision: the capture's rules (``Rules()``, ``collapse_sleep``)
play every tick of it in that decision's step, so its frames show the player
lying down and waking. With ``sleep_stride``, the replay also plays each sleep
a tick at a time on a copy of the world before it, through the same game code
with ``collapse_sleep`` off (NOOP each tick while the player sleeps or rests,
as the collapsed loop plays), takes a frame after every ``sleep_stride``-th
tick the player sleeps through, and refuses the replay unless the copy ends
on the collapsed step's world, stream and tick count.

Beside each frame the replay keeps its view's projectile facings (``facings``):
per window cell, the move action (LEFT, RIGHT, UP, DOWN: 1-4) of the tile the
mob projectile the observation shows there flies each tick, in bits 0-3, and
the player projectile's in bits 4-7, from the game's ``mob_projectile_dirs``
and ``player_projectile_directions``; 0 where it shows none. The policy-view
bundle (``policy_view.py``) carries them; an exact game bundle does not.

A bundle is ``frames.bin.gz``, ``health.bin.gz`` and ``manifest.json``, whose
SHA-256s the manifest records; ``games.mjs build`` checks them and inlines
the bundle into one page beside model bundles (``bundle.py``), which share the
frame layout with the fields only a State knows left zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import dataclasses
import functools
import gzip
import hashlib
import json
import re

from numpy.typing import NDArray

import numpy as np

from priml.baselines.craftax.game.mobs import (
    MELEE_HEALTH,
    PASSIVE_HEALTH,
    RANGED_HEALTH,
)
from priml.baselines.craftax.game.rules import (
    max_health_numba,
    max_mana_numba,
    max_need_numba,
)
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    ATN_DIM,
    MAX_ACHIEVEMENT_RETURN,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PASSIVE_MOBS,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    NUM_MOB_TYPES,
    OBS_COLS,
    OBS_ROWS,
    OBS_SIZE,
    VISIBLE_LIGHT_THRESHOLD,
    Action,
    Array1,
    EnvState,
    Mobs,
    env_state,
    env_stats,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, observe_numba, play_numba
from priml.baselines.craftax.lib.arrays import Shaped, ints
from priml.baselines.craftax.world_model import replay
from priml.lib.codec import MutablePlainTree, from_plain, loads
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from _typeshed import DataclassInstance

    from priml.baselines.craftax.world_model.replay import Replayable


CELL_BYTES: Final = 792
"""The observation window's cell values a frame keeps: 9 x 11 tiles of 8 channels."""

CREATURES: Final = (
    ("melee_mobs", MAX_MELEE_MOBS, MELEE_HEALTH),
    ("passive_mobs", MAX_PASSIVE_MOBS, PASSIVE_HEALTH),
    ("ranged_mobs", MAX_RANGED_MOBS, RANGED_HEALTH),
    ("mob_projectiles", MAX_MOB_PROJECTILES, None),
    ("player_projectiles", MAX_PLAYER_PROJECTILES, None),
)
"""Each creature class in the frame's order (its index is the class byte): the
State field, its slots, and each species' full health, None for projectiles."""

PROJECTILES: Final = (
    ("mob_projectiles", MAX_MOB_PROJECTILES, "mob_projectile_dirs", 0),
    ("player_projectiles", MAX_PLAYER_PROJECTILES, "player_projectile_directions", 4),
)
"""Each projectile class, in its observation channel's order (6, then 7): the
State field, its slots, its direction field and the bit its facing starts at."""

FACINGS: Final = {
    (0, -1): Action.LEFT.value,
    (0, 1): Action.RIGHT.value,
    (-1, 0): Action.UP.value,
    (1, 0): Action.DOWN.value,
}
"""The move action of each direction a projectile flies, (drow, dcol)."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Replayed:
    """An episode's frames, as the viewer reads them, and its State hashes.

    Attributes:
      frames: ``frame_dtype() [T + 1]``: each decision's, then the final State's.
      health: ``health_dtype() [T + 1]``, the creatures' health in each frame.
      hashes: uint64 ``[T]``, the State's hash (``replay.fnv1a_numba``) after each
        decision.
      ended: Whether the last decision ended the episode.
      sleeps: int64 ``[n, 3]``: each sleep's decision, the ticks its step
        played and its first frame in ``sleep_frames``; empty unless the
        replay was asked for sleep frames.
      sleep_frames: ``frame_dtype() [m]``: each sleep's frames, every
        ``sleep_stride``-th tick it sleeps through, the before-values of the
        State after that tick.
      sleep_health: ``health_dtype() [m]``, the creatures' health in each.
      facings: uint8 ``[T + 1, 99]``: each frame's view cells' projectile
        facings (the module docstring).
      sleep_facings: uint8 ``[m, 99]``: each sleep frame's.

    """

    frames: NDArray[np.void]
    health: NDArray[np.void]
    hashes: NDArray[np.uint64]
    ended: bool
    facings: Shaped[np.uint8]
    sleeps: Shaped[np.int64] = dataclasses.field(
        default_factory=lambda: np.zeros((0, 3), np.int64),
    )
    sleep_frames: NDArray[np.void] = dataclasses.field(
        default_factory=lambda: np.zeros(0, frame_dtype()),
    )
    sleep_health: NDArray[np.void] = dataclasses.field(
        default_factory=lambda: np.zeros(0, health_dtype()),
    )
    sleep_facings: Shaped[np.uint8] = dataclasses.field(
        default_factory=lambda: np.zeros((0, OBS_ROWS * OBS_COLS), np.uint8),
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    """``manifest.json`` of an exact game bundle; ``games.mjs build`` verifies it.

    ``tick`` is the final frame's tick, ``score`` its achievement return and
    ``achievements`` its count of achievements; ``mean_score`` and
    ``mean_return_pct`` are the policy's average beside the game, when known.
    """

    schema_name: str = "craftax-exact-game/v1"
    title: str
    description: str
    end_label: str
    provenance: str
    frames: int
    actions: int
    tick: int
    score: int
    achievements: int
    mean_score: float | None
    mean_return_pct: float | None
    frames_sha256: str
    gzip_sha256: str
    health_sha256: str
    health_gzip_sha256: str


def manifest_json(manifest: DataclassInstance) -> str:
    """Return a viewer manifest as ``manifest.json`` holds it, for ``games.mjs``.

    Keys are camelCase and ``schema_name`` is ``schema``, nested manifests
    alike; two-space indent, UTF-8 text, a final line feed.

    Args:
      manifest: An exact, model or policy-view bundle's manifest.

    Returns:
      text: The file's contents.

    """
    fields = from_plain(dataclasses.asdict(manifest), dict[str, MutablePlainTree])
    renamed = _renamed(fields, _wire_name)
    return json.dumps(renamed, indent=2, ensure_ascii=False) + "\n"


def read_manifest[T](text: str, kind: type[T]) -> T:
    """Return the ``kind`` manifest a ``manifest.json`` holds; ``manifest_json``'s inverse.

    Args:
      text: The file's contents.
      kind: The manifest's class.

    Returns:
      manifest: The manifest.

    """
    return from_plain(_renamed(loads(text), _field_name), kind)


@functools.cache
def frame_dtype() -> np.dtype[np.void]:
    """Return the packed 7,858-byte frame record the viewer reads.

    Returns:
      dtype: A decision's before-values, action and after-values, the
        observation window's 792 cell values, the HUD, the floor's 48x48 map
        (block, item and light per cell, row-major) and up to 15 creatures as
        class, species, row and column bytes.

    """
    u8, u16 = "u1", "<u2"
    return np.dtype(
        [
            *[(name, "<u4") for name in ("step", "tick_before", "tick_after")],
            *[(name, u8) for name in ("action", "floor_before", "floor_after")],
            *[(name, u8) for name in ("row", "col", "direction", "sleeping")],
            ("terminal", u8),
            ("score", u16),
            ("health_before", "<f4"),
            ("health_after", "<f4"),
            ("mana_before", "<i2"),
            ("mana_after", "<i2"),
            ("observation", u8, (CELL_BYTES,)),
            *[(name, u8) for name in ("health_max", "mana_max", "need_max")],
            *[(name, u8) for name in ("food", "drink", "energy", "achievements")],
            ("inventory", u16, (24,)),
            ("spells", u8, (2,)),
            ("sword_enchant", u8),
            ("bow_enchant", u8),
            ("map", u8, (48 * 48 * 3,)),
            ("mob_count", u8),
            ("mobs", u8, (60,)),
        ],
    )


@functools.cache
def health_dtype() -> np.dtype[np.void]:
    """Return the 120-byte sidecar record: current, then full, health of 15 creatures."""
    return np.dtype([("current", "<f4", (15,)), ("maximum", "<f4", (15,))])


class Frame(Protocol):
    """One ``frame_dtype()`` record as :func:`read_frame` hands it to Python."""

    step: int
    tick_before: int
    tick_after: int
    action: int
    floor_before: int
    floor_after: int
    row: int
    col: int
    direction: int
    sleeping: int
    terminal: int
    score: int
    health_before: np.float32
    health_after: np.float32
    mana_before: int
    mana_after: int
    observation: Array1[int]
    health_max: int
    mana_max: int
    need_max: int
    food: int
    drink: int
    energy: int
    achievements: int
    inventory: Array1[int]
    spells: Array1[int]
    sword_enchant: int
    bow_enchant: int
    map: Array1[int]
    mob_count: int
    mobs: Array1[int]


class Health(Protocol):
    """One ``health_dtype()`` record as :func:`read_health` hands it to Python."""

    current: Array1[np.float32]
    maximum: Array1[np.float32]


def read_frame(frames: NDArray[np.void], index: int) -> Frame:
    """Return frame ``index`` of ``frames``, its fields read as attributes; a view."""
    if frames.dtype != frame_dtype():
        raise ValueError("frames must be a frame_dtype() array")
    return cast("Frame", frames.view(np.recarray)[index])


def read_health(health: NDArray[np.void], index: int) -> Health:
    """Return health record ``index`` of ``health``, as :func:`read_frame`."""
    if health.dtype != health_dtype():
        raise ValueError("health must be a health_dtype() array")
    return cast("Health", health.view(np.recarray)[index])


def replay_episode(
    record: Replayable,
    *,
    limit: int | None = None,
    sleep_stride: int | None = None,
) -> Replayed:
    """Replay a recorded episode from its start, checking every hash it records.

    Args:
      record: A capture archive's record (``archive.Record``) or episode.
      limit: Replay only the first ``limit`` decisions, a prefix that need not
        end; None replays them all.
      sleep_stride: Take a frame every ``sleep_stride`` ticks of each sleep;
        None takes none.

    Returns:
      replayed: The episode's frames, health and hashes.

    Raises:
      ValueError: The start, a State before a 256th decision, or the State
        after the last decision differs from its recorded hash, or the
        episode does not end where the record does.

    """
    states, rng = replay.load(replay.initial(record).state)
    initial = replay.fnv1a_numba(states.view(np.uint8))
    actions = record.actions.numpy()[:limit]
    played = replay_frames(states, rng, actions=actions, sleep_stride=sleep_stride)
    # The State before each decision, then after the last one replayed. The
    # kernel returns a Python int, which numpy would widen with the hashes to
    # float64, too coarse to tell two hashes apart.
    before = np.concatenate([np.array([initial], dtype=np.uint64), played.hashes])
    recorded = record.hashes.numpy().view(np.uint64)
    at = np.append(
        np.arange(len(recorded) - 1) * replay.HASH_STRIDE,
        len(record.actions),
    )
    checked = at <= len(actions)
    missed = np.flatnonzero(np.not_equal(before[at[checked]], recorded[checked]))
    if len(missed):
        raise ValueError(f"Replay does not match hash {missed[0]} of the episode.")
    if checked[-1] and played.ended == record.truncated:
        raise ValueError(
            "The replayed episode ended, though its record was truncated."
            if played.ended
            else "The replayed episode did not end at its last decision.",
        )
    return played


def replay_frames(
    states: NDArray[np.void],
    rng: NDArray[np.uint32],
    *,
    actions: NDArray[np.uint8],
    sleep_stride: int | None = None,
) -> Replayed:
    """Play ``actions`` from one world and take each decision's frame.

    Args:
      states: ``STATE_DTYPE [1]``, the world before the first action, stepped
        in place.
      rng: uint32 ``[1]``, its environment stream, advanced in place.
      actions: uint8 ``[T]``, at least one.
      sleep_stride: Take a frame every ``sleep_stride`` ticks of each sleep;
        None takes none.

    Returns:
      replayed: The frames, health and hashes; ``ended`` when the last
        action ended the episode.

    Raises:
      ValueError: There is no action, the episode ends before its last
        action, a creature's species is outside the game's eight, or a sleep
        played a tick at a time ends elsewhere than its step.

    """
    decisions = len(actions)
    if not decisions:
        raise ValueError("An exact game needs at least one decision.")
    frames = np.zeros(decisions + 1, dtype=frame_dtype())
    health = np.zeros(decisions + 1, dtype=health_dtype())
    facings = np.zeros((decisions + 1, OBS_ROWS * OBS_COLS), np.uint8)
    hashes = np.zeros(decisions, dtype=np.uint64)
    stats = new_stats(1)
    ended = False
    sleeps: list[tuple[int, int, int]] = []
    slept: list[tuple[np.void, np.void, NDArray[np.uint8]]] = []
    state, stat = env_state(states, 0), env_stats(stats, 0)
    for t, action in enumerate(ints(actions)):
        if ended:
            raise ValueError(
                f"The episode ended at decision {t - 1}, before its last action.",
            )
        frame = _record(frames, t)
        _before(frame, _record(health, t), facings[t, :], states)
        frame["step"] = t
        frame["action"] = action
        sleeping = sleep_stride is not None and action == Action.SLEEP.value
        before = (states.copy(), rng.copy(), stats.copy()) if sleeping else None
        _, ended = play_numba(state, rng, stat, action, Rules())
        _after(frame, state)
        frame["terminal"] = ended
        hashes[t] = replay.fnv1a_numba(states.view(np.uint8))
        if before is not None and sleep_stride is not None and stat.last_ticks > 1:
            ticks, taken = _sleep_frames(
                *before,
                action=action,
                stride=sleep_stride,
                after=(states, rng),
            )
            if ticks != stat.last_ticks:
                raise ValueError(
                    f"The sleep at decision {t} played {ticks} ticks one at a time, not its step's {stat.last_ticks}.",
                )
            sleeps.append((t, ticks, len(slept)))
            for sleep_frame, creatures, facing in taken:
                sleep_frame["step"] = t
                sleep_frame["action"] = action
                slept.append((sleep_frame, creatures, facing))
    final = _record(frames, decisions)
    _before(final, _record(health, decisions), facings[decisions, :], states)
    _after(final, state)
    final["step"] = decisions
    final["action"] = 255
    final["terminal"] = ended
    return Replayed(
        frames=frames,
        health=health,
        hashes=hashes,
        ended=bool(ended),
        facings=facings,
        sleeps=np.array(sleeps, np.int64).reshape(-1, 3),
        sleep_frames=np.array([frame for frame, _, _ in slept], frame_dtype()).reshape(
            -1,
        ),
        sleep_health=np.array(
            [creatures for _, creatures, _ in slept],
            health_dtype(),
        ).reshape(-1),
        sleep_facings=np.array(
            [facing for _, _, facing in slept],
            np.uint8,
        ).reshape(-1, OBS_ROWS * OBS_COLS),
    )


def write_bundle(
    replayed: Replayed,
    output: Path,
    *,
    title: str,
    description: str = "",
    end_label: str = "",
    provenance: str = "",
    mean_score: float | None = None,
) -> Manifest:
    """Write ``replayed`` as an exact game bundle for ``games.mjs build``.

    Args:
      replayed: The episode's frames and health.
      output: New directory, under ``/opt/scratch/artifacts/``.
      title: Picker title.
      description: Page description; defaults to the decision count.
      end_label: Name of the final frame; defaults to whether the episode ended.
      provenance: Where the game came from.
      mean_score: The policy's average achievement return, shown beside the
        game's; None when unknown.

    Returns:
      manifest: What ``manifest.json`` holds.

    Raises:
      FileExistsError: If ``output`` exists.

    """
    output = validated_output_path(output)
    output.mkdir(parents=True)
    digests: dict[str, str] = {}
    for name, records in (("frames", replayed.frames), ("health", replayed.health)):
        raw = records.tobytes()
        packed = gzip.compress(raw, mtime=0)
        (output / f"{name}.bin.gz").write_bytes(packed)
        digests[name] = hashlib.sha256(raw).hexdigest()
        digests[f"{name}_gzip"] = hashlib.sha256(packed).hexdigest()
    final = read_frame(replayed.frames, -1)
    actions = len(replayed.frames) - 1
    manifest = Manifest(
        title=title,
        description=description or f"{actions:,} recorded actions",
        end_label=end_label
        or ("Episode end" if replayed.ended else "End of recording"),
        provenance=provenance
        or "Replayed on the port's game; every recorded hash matched.",
        frames=actions + 1,
        actions=actions,
        tick=int(final.tick_before),
        score=int(final.score),
        achievements=int(final.achievements),
        mean_score=mean_score,
        mean_return_pct=None
        if mean_score is None
        else 100 * mean_score / float(MAX_ACHIEVEMENT_RETURN),
        frames_sha256=digests["frames"],
        gzip_sha256=digests["frames_gzip"],
        health_sha256=digests["health"],
        health_gzip_sha256=digests["health_gzip"],
    )
    (output / "manifest.json").write_text(manifest_json(manifest))
    return manifest


def _before(
    frame: np.void,
    health: np.void,
    facings: NDArray[np.uint8],
    states: NDArray[np.void],
) -> None:
    """Write what a frame, its health and its facings read from the State before its decision."""
    state = env_state(states, 0)
    floor = int(state.player_level)
    frame["tick_before"] = state.timestep
    frame["floor_before"] = floor
    frame["row"], frame["col"] = state.player_position
    frame["direction"] = state.player_direction
    frame["sleeping"] = state.is_sleeping
    frame["health_before"] = state.player_health
    frame["mana_before"] = state.player_mana
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    observe_numba(state, observation, np.zeros(ATN_DIM, np.uint8), Rules())
    cells = observation[:CELL_BYTES].astype(np.uint8)
    frame["observation"] = cells
    _hud(frame, state)
    grids = [state.map[floor], state.item_map[floor], state.light_map[floor]]
    frame["map"] = np.stack(grids, axis=-1).reshape(-1)
    _creatures(frame, health, state)
    facings[:] = _facings(state, cells)


def _hud(frame: np.void, state: EnvState) -> None:
    """Write the meters' caps, the needs, the achievement count and what is held."""
    frame["health_max"] = max_health_numba(state)
    frame["mana_max"] = max_mana_numba(state)
    frame["need_max"] = max_need_numba(state)
    frame["food"] = state.player_food
    frame["drink"] = state.player_drink
    frame["energy"] = state.player_energy
    frame["achievements"] = np.count_nonzero(state.achievements)
    held = state.inventory
    frame["inventory"] = [
        *(held.wood, held.stone, held.coal, held.iron, held.diamond),
        *(held.sapling, held.pickaxe, held.sword, held.bow, held.arrows),
        *held.armour,
        *(held.torches, held.ruby, held.sapphire),
        *held.potions,
        held.books,
    ]
    frame["spells"] = state.learned_spells
    frame["sword_enchant"] = state.sword_enchantment
    frame["bow_enchant"] = state.bow_enchantment


def _creatures(frame: np.void, health: np.void, state: EnvState) -> None:
    """List the floor's creatures, and the health of each that has any."""
    floor = int(state.player_level)
    count = 0
    for klass, (_, slots, full) in enumerate(CREATURES):
        mobs = _classes(state)[klass][floor]
        for slot in range(slots):
            if not mobs.mask[slot]:
                continue
            kind = int(mobs.type_id[slot])
            # A negative species would index the health table from its end.
            if kind < 0 or kind >= NUM_MOB_TYPES:
                raise ValueError(f"A creature of class {klass} has species {kind}.")
            frame["mobs"][4 * count : 4 * count + 4] = (
                klass,
                kind,
                mobs.position[slot, 0],
                mobs.position[slot, 1],
            )
            if full is not None:
                health["current"][count] = mobs.health[slot]
                health["maximum"][count] = full[kind]
            count += 1
    frame["mob_count"] = count


def _classes(state: EnvState) -> tuple[Sequence[Mobs], ...]:
    """Return the State's creature classes in ``CREATURES``' order."""
    return (
        state.melee_mobs,
        state.passive_mobs,
        state.ranged_mobs,
        state.mob_projectiles,
        state.player_projectiles,
    )


def _facings(state: EnvState, cells: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return each view cell's projectile facings, as the module docstring lays them out."""
    out = np.zeros(OBS_ROWS * OBS_COLS, np.uint8)
    level = int(state.player_level)
    row, col = (int(v) for v in state.player_position)
    light = state.light_map[level]
    flights = (
        (state.mob_projectiles, state.mob_projectile_dirs),
        (state.player_projectiles, state.player_projectile_directions),
    )
    for channel, ((name, slots, _, shift), (projectiles, directions)) in enumerate(
        zip(PROJECTILES, flights, strict=True),
        start=6,
    ):
        mobs = projectiles[level]
        # As the observation writes them (observation._visible_tile_numba): live,
        # in view and lit, a later slot over an earlier one on the same cell.
        for slot in range(slots):
            if not mobs.mask[slot]:
                continue
            at_row, at_col = int(mobs.position[slot, 0]), int(mobs.position[slot, 1])
            view_row, view_col = (
                at_row - row + OBS_ROWS // 2,
                at_col - col + OBS_COLS // 2,
            )
            if (
                view_row < 0
                or view_row >= OBS_ROWS
                or view_col < 0
                or view_col >= OBS_COLS
                or light[at_row, at_col] <= VISIBLE_LIGHT_THRESHOLD
            ):
                continue
            direction = int(directions[level, slot, 0]), int(directions[level, slot, 1])
            if direction not in FACINGS:
                raise ValueError(f"A projectile of {name} flies {direction}.")
            cell = view_row * OBS_COLS + view_col
            out[cell] = out[cell] & (0xF0 >> shift) | FACINGS[direction] << shift
        shown = np.not_equal(cells.reshape(OBS_ROWS * OBS_COLS, -1)[:, channel], 0)
        if not np.array_equal(shown, np.not_equal(out >> shift & 0xF, 0)):
            raise ValueError(
                f"The view's {name} are not where its observation shows them.",
            )
    return out


def _after(frame: np.void, state: EnvState) -> None:
    """Write what a frame reads from the State after its decision."""
    frame["tick_after"] = state.timestep
    frame["floor_after"] = state.player_level
    frame["health_after"] = state.player_health
    frame["mana_after"] = state.player_mana
    unlocked = ACHIEVEMENT_REWARD_MAP[np.flatnonzero(state.achievements)]
    frame["score"] = int(np.sum(unlocked))


def _record(records: NDArray[np.void], index: int) -> np.void:
    """Return record ``index`` of ``records``, a writable view."""
    return cast("np.void", records[index])


# Returns its ticks and a frame, a health record and the facings after every
# ``stride``-th tick the player sleeps through.
def _sleep_frames(
    states: NDArray[np.void],
    rng: NDArray[np.uint32],
    stats: NDArray[np.void],
    *,
    action: int,
    stride: int,
    after: tuple[NDArray[np.void], NDArray[np.uint32]],
) -> tuple[int, list[tuple[np.void, np.void, NDArray[np.uint8]]]]:
    """Play a sleep a tick at a time from copies of the world before it."""
    ticked = Rules(collapse_sleep=False)
    state = env_state(states, 0)
    _, done = play_numba(state, rng, env_stats(stats, 0), action, ticked)
    ticks, taken = 1, list[tuple[np.void, np.void, NDArray[np.uint8]]]()
    while not done and (state.is_sleeping or state.is_resting):
        if ticks % stride == 0:
            frames, creatures = np.zeros(1, frame_dtype()), np.zeros(1, health_dtype())
            frame, health = _record(frames, 0), _record(creatures, 0)
            facings = np.zeros(OBS_ROWS * OBS_COLS, np.uint8)
            _before(frame, health, facings, states)
            _after(frame, state)
            taken.append((frame, health, facings))
        _, done = play_numba(
            state,
            rng,
            env_stats(stats, 0),
            Action.NOOP.value,
            ticked,
        )
        ticks += 1
    if states.tobytes() != after[0].tobytes() or rng.tobytes() != after[1].tobytes():
        raise ValueError(
            "A sleep played a tick at a time ends off its collapsed step's world.",
        )
    return ticks, taken


def _renamed(
    tree: MutablePlainTree,
    rename: Callable[[str], str],
) -> MutablePlainTree:
    """Return ``tree`` with every object key passed through ``rename``."""
    if isinstance(tree, dict):
        return {rename(key): _renamed(value, rename) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_renamed(value, rename) for value in tree]
    return tree


def _wire_name(field: str) -> str:
    """Return the manifest key of a field: camelCase, ``schema_name`` as ``schema``."""
    if field == "schema_name":
        return "schema"
    head, *rest = field.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _field_name(key: str) -> str:
    """Return the field a manifest key names; ``_wire_name``'s inverse."""
    if key == "schema":
        return "schema_name"
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", key).lower()
