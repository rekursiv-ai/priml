"""The ghost site's data files: their layout, and the reference decoder of them.

``FORMAT.md`` beside this file specifies every file ``build.py`` writes; this
module holds the pieces both sides share (the bit and column layouts, the
world file, the event tables, the creature windows, the JSON records) and
decodes them the way the page must. Tests compare what it decodes with the
replayed game, and ``fixture.py`` writes what it decodes as the page's
expected values, so the page's decoder is checked against the game through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import dataclasses
import struct

from numpy.typing import NDArray

import numpy as np

from priml.baselines.craftax.game.state import (
    MAP_SIZE,
    NUM_LEVELS,
    Action,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


FORMAT: Final = "craftax-ghosts/4"
"""The ``format`` every manifest names; a layout change bumps it."""

CREATURE_BYTES: Final = 4
"""A sampled creature's bytes: class << 4 | species, row, column, facing."""

ACTION_MASK: Final = 0x3F
"""Player byte bits 0-5: the effective action."""

MOVED: Final = 0x40
"""Player byte bit 6: the player stepped one tile in the action's direction."""

FLOOR_CHANGE: Final = 0x80
"""Player byte bit 7: the player took the ladder the action names."""

INTERACTIONS: Final = frozenset(
    {
        Action.DO,
        Action.PLACE_STONE,
        Action.PLACE_TABLE,
        Action.PLACE_FURNACE,
        Action.PLACE_PLANT,
        Action.SHOOT_ARROW,
        Action.CAST_FIREBALL,
        Action.CAST_ICEBALL,
        Action.PLACE_TORCH,
    },
)
"""Actions aimed at the faced tile, as the single-game viewer counts them."""

STEPS: Final = {
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
}
"""``(row, column)`` step of each movement action, which is also the facing it sets."""

MAP_COLUMNS: Final = ("<u4", "u1", "u1", "u1", "u1", "u1")
"""Map events: decision, floor, row, column, block, item."""

ACHIEVEMENT_COLUMNS: Final = ("<u4", "u1")
"""Achievement events: decision, achievement."""

ESCAPE_COLUMNS: Final = ("<u4", "u1", "u1", "u1", "u1")
"""Escapes: decision, then the floor, row, column and facing after it."""

ACTIVITY_COLUMNS: Final = ("<u2",)
"""A timeline's activity: per decision, the shown episodes active at it."""

SEGMENT_COLUMNS: Final = ("<u4", "<u4")
"""A timeline's kept segments: first decision, one past the last."""

KEEP_COUNT_COLUMNS: Final = ("<u4",)
"""A time map file's first table: per episode, how many kept runs it has."""

KEEP_COLUMNS: Final = ("<u4", "<u4")
"""A time map file's second table: each kept run's first decision, one past its last."""

SLEEP_COUNT_COLUMNS: Final = ("<u4",)
"""A sleep file's first table: per episode, how many sleeps it has."""

SLEEP_COLUMNS: Final = ("<u4", "<u4")
"""A sleep file's second table: each sleep's decision and the ticks its step played."""

SLEEP_SAMPLE_COLUMNS: Final = ("<u4", "<u4")
"""A sleep file's third table: each sample's creature run offset, and its change rows."""

SLEEP_CHANGE_COLUMNS: Final = ("u1", "u1", "u1", "u1", "u1")
"""A sleep file's fourth table: a changed tile's floor, row, column, block and item."""

SLEEP_RUN_COLUMNS: Final = ("u1",)
"""A sleep file's fifth table: the samples' creature runs, byte by byte."""

_GRID: Final = (NUM_LEVELS, MAP_SIZE, MAP_SIZE)
_U32: Final = struct.Struct("<I")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class World:
    """The shared world every episode starts in, as ``world.bin`` holds it.

    Attributes:
      block: uint8 ``[9, 48, 48]``, each tile's block at the reset.
      item: uint8 ``[9, 48, 48]``, each tile's item at the reset.
      light: uint8 ``[9, 48, 48]``, each tile's light at the reset.
      down_ladders: int64 ``[9, 2]``, each floor's down ladder (row, column).
      up_ladders: int64 ``[9, 2]``, each floor's up ladder (row, column).

    """

    block: NDArray[np.uint8]
    item: NDArray[np.uint8]
    light: NDArray[np.uint8]
    down_ladders: NDArray[np.int64]
    up_ladders: NDArray[np.int64]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Events:
    """One group's event tables, int64 rows in the column order of their layouts.

    Attributes:
      map: ``[n, 6]``: decision, floor, row, column, block, item.
      achievements: ``[n, 2]``: decision, achievement.
      escapes: ``[n, 5]``: decision, floor, row, column, facing.

    """

    map: NDArray[np.int64]
    achievements: NDArray[np.int64]
    escapes: NDArray[np.int64]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeEntry:
    """One episode's line of ``episodes.json``; FORMAT.md defines each field."""

    index: int
    source: int
    ordinal: int
    sampling_seed: str
    decisions: int
    outcome: str
    end: tuple[int, int, int, int]
    achievement_return: int
    floor_first: tuple[int, ...]
    players: int
    map: tuple[int, int]
    achievements: tuple[int, int]
    escapes: tuple[int, int]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GroupFile:
    """``episodes.json``: one group's episodes, in index order."""

    tier: str
    set: str
    group: int
    first: int
    episodes: tuple[EpisodeEntry, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Group:
    """A group as the manifest lists it."""

    path: str
    first: int
    count: int
    decisions: int
    windows: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Stats:
    """A set's first ``count`` episodes in numbers."""

    count: int
    mean_return: float
    mean_decisions: float
    reached: tuple[int, ...]
    deaths: tuple[int, ...]
    timeouts: int
    wins: int
    escapes: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Composition:
    """What a set was chosen from and what it holds; FORMAT.md defines each field."""

    run: int
    qualified: int
    deaths: int
    wins: int
    timeouts: int
    truncated: int
    added_wins: int
    natural_win_share: float
    death_decisions: tuple[int, ...]
    win_decisions: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Timeline:
    """One count's activity and quiet-stretch timeline file, as the manifest lists it."""

    count: int
    path: str
    decisions: int
    kept: int
    segments: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TimeMap:
    """A set's time maps in numbers; FORMAT.md defines each field."""

    rule: TimeRule
    steps: int
    kept: int
    shortest: int
    median: int
    levels: tuple[int, ...]
    unbroken: int | None
    whole: int
    samples: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeSet:
    """One of a tier's episode sets: ``all``, ``short`` or ``wins``."""

    name: str
    rule: str
    counts: tuple[int, ...]
    episodes: int
    decisions: int
    max_decisions: int
    composition: Composition
    groups: tuple[Group, ...]
    stats: tuple[Stats, ...]
    timelines: tuple[Timeline, ...]
    time_map: TimeMap | None
    sleep_map: TimeMap | None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Source:
    """A capture a tier's episodes came from."""

    root: str
    capped: bool
    episodes: int
    provenance: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Tier:
    """A tier as the manifest lists it."""

    name: str
    arm: int
    sources: tuple[Source, ...]
    sets: tuple[EpisodeSet, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QuietRule:
    """When a decision is quiet, and how a run of quiet decisions collapses.

    Attributes:
      per_live: A decision is quiet when fewer than ``max(1, ceil(live /
        per_live))`` of the shown episodes are active at it, ``live`` being
        those not yet ended.
      min_run: A run of at least this many quiet decisions collapses.
      keep: Decisions a collapsed run keeps at each of its ends.

    """

    per_live: int = 50
    min_run: int = 64
    keep: int = 8


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TimeRule:
    """How a win's time map compresses it (``sets.keep_runs``).

    Attributes:
      steps: Decisions a time map keeps at most: the longest the set plays.
      levels: ``(context, idle)`` pairs, most generous first: decisions kept on
        each side of a progress decision, and decisions kept of each other
        stretch. The first that fits ``steps`` is used.

    """

    steps: int = 5000
    levels: tuple[tuple[int, int], ...] = (
        (2, 8),
        (2, 4),
        (1, 4),
        (1, 2),
        (0, 2),
        (0, 1),
        (0, 0),
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    """``manifest.json``; FORMAT.md defines each field."""

    format: str
    git_commit: str
    game_package_digest: str
    platform: str
    world_seed: int
    start: tuple[int, int, int, int]
    creature_stride: int
    sleep_stride: int
    window_decisions: int
    counts: tuple[int, ...]
    short_decisions: int
    quiet: QuietRule
    achievement_rewards: tuple[int, ...]
    tiers: tuple[Tier, ...]
    files: dict[str, str]
    sizes: dict[str, int]


def world_bytes(world: World) -> bytes:
    """Return ``world.bin``: the three grids, then the down and up ladders, as bytes.

    Args:
      world: The shared world.

    Returns:
      data: The file's bytes before compression.

    """
    return b"".join(
        np.ascontiguousarray(part, dtype=np.uint8).tobytes()
        for part in (
            world.block,
            world.item,
            world.light,
            world.down_ladders,
            world.up_ladders,
        )
    )


def read_world(data: bytes) -> World:
    """Return the world ``world.bin`` holds.

    Args:
      data: The decompressed file.

    Returns:
      world: Its grids and ladders.

    Raises:
      ValueError: The file is not 3 grids and 2 ladder tables long.

    """
    cells = NUM_LEVELS * MAP_SIZE * MAP_SIZE
    ladders = NUM_LEVELS * 2
    if len(data) != 3 * cells + 2 * ladders:
        raise ValueError(
            f"world.bin is {3 * cells + 2 * ladders} bytes, not {len(data)}.",
        )
    raw = np.frombuffer(data, np.uint8)
    grids = raw[: 3 * cells].reshape(3, *_GRID)
    tail = raw[3 * cells :].astype(np.int64).reshape(2, NUM_LEVELS, 2)
    return World(
        block=grids[0, ...],
        item=grids[1, ...],
        light=grids[2, ...],
        down_ladders=tail[0, ...],
        up_ladders=tail[1, ...],
    )


def encode_table(rows: NDArray[np.integer], *, columns: Sequence[str]) -> bytes:
    """Return a table: its row count, then each column, each padded to 4 bytes.

    Args:
      rows: int ``[n, len(columns)]``.
      columns: Each column's numpy dtype string, little-endian.

    Returns:
      table: ``u32 n`` then the columns.

    """
    parts = [_U32.pack(len(rows))]
    for k, dtype in enumerate(columns):
        column = rows[:, k].astype(dtype).tobytes()
        parts.append(column + bytes(-len(column) % 4))
    return b"".join(parts)


def read_table(
    data: bytes,
    *,
    offset: int,
    columns: Sequence[str],
) -> tuple[NDArray[np.int64], int]:
    """Return the table at ``offset`` as int64 rows, and the offset after it.

    Args:
      data: A decompressed file.
      offset: Where the table starts, a multiple of 4.
      columns: Each column's numpy dtype string.

    Returns:
      rows: int64 ``[n, len(columns)]``.
      end: The offset of the next table.

    """
    count = int.from_bytes(data[offset : offset + _U32.size], "little")
    offset += _U32.size
    rows = np.zeros((count, len(columns)), np.int64)
    for k, dtype in enumerate(columns):
        width = np.dtype(dtype).itemsize * count
        rows[:, k] = np.frombuffer(data, dtype, count, offset)
        offset += width + -width % 4
    return rows, offset


def encode_events(events: Events) -> bytes:
    """Return ``events.bin``: the map, achievement and escape tables, in that order."""
    return (
        encode_table(events.map, columns=MAP_COLUMNS)
        + encode_table(events.achievements, columns=ACHIEVEMENT_COLUMNS)
        + encode_table(events.escapes, columns=ESCAPE_COLUMNS)
    )


def read_events(data: bytes) -> Events:
    """Return the tables of an ``events.bin``.

    Args:
      data: The decompressed file.

    Returns:
      events: Its three tables.

    Raises:
      ValueError: Bytes follow the last table.

    """
    map_rows, offset = read_table(data, offset=0, columns=MAP_COLUMNS)
    achievements, offset = read_table(data, offset=offset, columns=ACHIEVEMENT_COLUMNS)
    escapes, offset = read_table(data, offset=offset, columns=ESCAPE_COLUMNS)
    if offset != len(data):
        raise ValueError(f"events.bin has {len(data) - offset} bytes after its tables.")
    return Events(map=map_rows, achievements=achievements, escapes=escapes)


def encode_timeline(
    activity: NDArray[np.integer],
    *,
    segments: NDArray[np.integer],
) -> bytes:
    """Return a timeline file: the activity table, then the kept segments table.

    Args:
      activity: int ``[D]``, the shown episodes active at each decision.
      segments: int ``[n, 2]``, the kept ``[start, stop)`` ranges, in order.

    Returns:
      data: The file's bytes before compression.

    """
    return encode_table(activity[:, None], columns=ACTIVITY_COLUMNS) + encode_table(
        segments,
        columns=SEGMENT_COLUMNS,
    )


def read_timeline(data: bytes) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return a timeline file's activity ``[D]`` and kept segments ``[n, 2]``.

    Args:
      data: The decompressed ``timeline-n<count>.bin``.

    Returns:
      activity: int64 ``[D]``.
      segments: int64 ``[n, 2]``.

    Raises:
      ValueError: Bytes follow the segments.

    """
    activity, offset = read_table(data, offset=0, columns=ACTIVITY_COLUMNS)
    segments, offset = read_table(data, offset=offset, columns=SEGMENT_COLUMNS)
    if offset != len(data):
        raise ValueError(f"A timeline has {len(data) - offset} bytes after its tables.")
    return activity[:, 0], segments


def encode_keeps(runs: Sequence[NDArray[np.int64]]) -> bytes:
    """Return a group's time map file: each episode's run count, then every run.

    Args:
      runs: Per episode in index order, its kept ``[start, stop)`` runs ``[n, 2]``.

    Returns:
      data: The file's bytes before compression.

    """
    counts = np.array([len(r) for r in runs], np.int64)[:, None]
    rows = np.concatenate([*runs, np.zeros((0, 2), np.int64)])
    return encode_table(counts, columns=KEEP_COUNT_COLUMNS) + encode_table(
        rows,
        columns=KEEP_COLUMNS,
    )


def read_keeps(data: bytes) -> list[NDArray[np.int64]]:
    """Return each episode's kept runs ``[n, 2]`` from a group's time map file.

    Args:
      data: The decompressed ``keeps.bin``.

    Returns:
      runs: One int64 ``[n, 2]`` per episode of the group, in index order.

    Raises:
      ValueError: The run counts do not add up to the runs, or bytes follow them.

    """
    counts, offset = read_table(data, offset=0, columns=KEEP_COUNT_COLUMNS)
    rows, offset = read_table(data, offset=offset, columns=KEEP_COLUMNS)
    if offset != len(data) or int(np.sum(counts)) != len(rows):
        raise ValueError("A time map's run counts do not match its runs.")
    ends = np.cumsum([0, *counts[:, 0]])
    return [rows[ends[i] : ends[i + 1]] for i in range(len(counts))]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Sleeps:
    """One episode's sleeps as ``sleep.bin`` holds them (``extract.Ghost``'s fields).

    Attributes:
      sleeps: int64 ``[n, 2]``: each sleep's decision and the ticks its step played.
      samples: int64 ``[m + 1]``: where each sample's creature run starts in
        ``creatures``, then where the last ends; a sleep of ``k`` ticks has
        ``(k - 1) // stride`` samples, in order.
      creatures: The samples' creature runs, as creature samples.
      changes: int64 ``[c, 6]``: each changed tile's sample, floor, row,
        column, block and item.

    """

    sleeps: NDArray[np.int64]
    samples: NDArray[np.int64]
    creatures: bytes
    changes: NDArray[np.int64]


def encode_sleep(episodes: Sequence[Sleeps]) -> bytes:
    """Return a group's sleep file: five tables, as FORMAT.md's ``sleep.bin``.

    Args:
      episodes: Per episode in index order, its sleeps.

    Returns:
      data: The file's bytes before compression.

    """
    counts = np.array([[len(e.sleeps)] for e in episodes], np.int64).reshape(-1, 1)
    sleeps = np.concatenate([*(e.sleeps for e in episodes), np.zeros((0, 2), np.int64)])
    runs = b"".join(e.creatures for e in episodes)
    starts = np.cumsum([0, *(len(e.creatures) for e in episodes)])
    samples = np.concatenate(
        [
            *(
                np.stack(
                    [
                        e.samples[:-1] + start,
                        np.bincount(e.changes[:, 0], minlength=len(e.samples) - 1),
                    ],
                    axis=1,
                )
                for e, start in zip(episodes, starts, strict=False)
            ),
            np.zeros((0, 2), np.int64),
        ],
    )
    changes = np.concatenate(
        [*(e.changes[:, 1:] for e in episodes), np.zeros((0, 5), np.int64)],
    )
    return b"".join(
        [
            encode_table(counts, columns=SLEEP_COUNT_COLUMNS),
            encode_table(sleeps, columns=SLEEP_COLUMNS),
            encode_table(samples, columns=SLEEP_SAMPLE_COLUMNS),
            encode_table(changes, columns=SLEEP_CHANGE_COLUMNS),
            encode_table(
                np.frombuffer(runs, np.uint8).astype(np.int64)[:, None],
                columns=SLEEP_RUN_COLUMNS,
            ),
        ],
    )


def read_sleep(data: bytes, *, stride: int) -> list[Sleeps]:
    """Return each episode's sleeps from a group's sleep file.

    Args:
      data: The decompressed ``sleep.bin``.
      stride: Ticks between a sleep's samples (``manifest.sleep_stride``).

    Returns:
      episodes: One per episode of the group, in index order.

    Raises:
      ValueError: The tables do not add up: sleeps, samples, changes or runs.

    """
    counts, offset = read_table(data, offset=0, columns=SLEEP_COUNT_COLUMNS)
    sleeps, offset = read_table(data, offset=offset, columns=SLEEP_COLUMNS)
    samples, offset = read_table(data, offset=offset, columns=SLEEP_SAMPLE_COLUMNS)
    changes, offset = read_table(data, offset=offset, columns=SLEEP_CHANGE_COLUMNS)
    runs, offset = read_table(data, offset=offset, columns=SLEEP_RUN_COLUMNS)
    held = (sleeps[:, 1] - 1) // stride
    if (
        offset != len(data)
        or int(np.sum(counts)) != len(sleeps)
        or int(np.sum(held)) != len(samples)
        or int(np.sum(samples[:, 1])) != len(changes)
    ):
        raise ValueError("A sleep file's tables do not add up.")
    blob = runs[:, 0].astype(np.uint8).tobytes()
    out: list[Sleeps] = []
    first_sleep, first_sample, first_change = 0, 0, 0
    for i in range(len(counts)):
        count = counts.item(i, 0)
        rows = sleeps[first_sleep : first_sleep + count]
        n = int(np.sum(held[first_sleep : first_sleep + count]))
        mine = samples[first_sample : first_sample + n]
        c = int(np.sum(mine[:, 1]))
        start = mine.item(0, 0) if n else 0
        stop = (
            samples.item(first_sample + n, 0)
            if first_sample + n < len(samples)
            else len(blob)
        )
        if not n:
            start = stop = 0
        out.append(
            Sleeps(
                sleeps=rows,
                samples=np.append(mine[:, 0], stop) - start,
                creatures=blob[start:stop],
                changes=np.column_stack(
                    [
                        np.repeat(np.arange(n), mine[:, 1]),
                        changes[first_change : first_change + c],
                    ],
                ).astype(np.int64)
                if c
                else np.zeros((0, 6), np.int64),
            ),
        )
        first_sleep += count
        first_sample += n
        first_change += c
    return out


def displayed_sleep(
    runs: NDArray[np.int64],
    sleeps: NDArray[np.int64],
    *,
    stride: int,
) -> NDArray[np.int64]:
    """Return what each display step of the view that shows sleep shows.

    Each kept decision is one step, and a kept sleep's decision is followed by
    one step per sample of it, the sleep's ticks ``stride``, ``2 stride``, ...

    Args:
      runs: The view's kept runs (``keeps-sleep.bin``), which keep every sleep.
      sleeps: The episode's sleeps ``[n, 2]``: decision and ticks.
      stride: Ticks between samples (``manifest.sleep_stride``).

    Returns:
      steps: int64 ``[L, 2]``: per display step its decision and its sample,
        1-based within the sleep, 0 for the state before the decision.

    """
    samples = {
        sleeps.item(i, 0): (sleeps.item(i, 1) - 1) // stride for i in range(len(sleeps))
    }
    steps: list[tuple[int, int]] = []
    shown = displayed(runs)
    for decision in (shown.item(i) for i in range(len(shown))):
        steps += [(decision, k) for k in range(samples.get(decision, 0) + 1)]
    return np.array(steps, np.int64).reshape(-1, 2)


def displayed(runs: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the decisions kept runs show, in display-step order."""
    spans = [
        np.arange(runs.item(i, 0), runs.item(i, 1), dtype=np.int64)
        for i in range(len(runs))
    ]
    return np.concatenate(spans) if spans else np.zeros(0, np.int64)


def encode_window(runs: Sequence[bytes]) -> bytes:
    """Return a creature window: the episode count, the runs' offsets, then the runs."""
    ends = np.cumsum([0, *(len(run) for run in runs)], dtype=np.int64)
    header = _U32.pack(len(runs)) + ends.astype("<u4").tobytes()
    return header + b"".join(runs)


def read_window(data: bytes) -> list[bytes]:
    """Return each episode's run of creature samples in one window file.

    Args:
      data: The decompressed ``creatures-w<j>.bin``.

    Returns:
      runs: One per episode of the group, in index order.

    Raises:
      ValueError: The offsets do not end at the file's end.

    """
    count = int.from_bytes(data[: _U32.size], "little")
    ends = np.frombuffer(data, "<u4", count + 1, _U32.size).astype(np.int64)
    body = _U32.size * (count + 2)
    if body + ends[-1] != len(data):
        raise ValueError("A creature window's offsets do not end at its end.")
    return [data[body + ends[i] : body + ends[i + 1]] for i in range(count)]


def decode_samples(run: bytes) -> list[list[tuple[int, int, int, int, int]]]:
    """Return each sample of a run as its creatures' (class, species, row, column, facing).

    Args:
      run: One episode's run of a creature window.

    Returns:
      samples: In decision order; each lists the creatures as the run does.

    """
    samples: list[list[tuple[int, int, int, int, int]]] = []
    at = 0
    while at < len(run):
        count = run[at]
        records = run[at + 1 : at + 1 + CREATURE_BYTES * count]
        samples.append(
            [
                (kind >> 4, kind & 0xF, row, col, facing)
                for kind, row, col, facing in zip(
                    records[::CREATURE_BYTES],
                    records[1::CREATURE_BYTES],
                    records[2::CREATURE_BYTES],
                    records[3::CREATURE_BYTES],
                    strict=True,
                )
            ],
        )
        at += 1 + CREATURE_BYTES * count
    return samples


def decode_path(
    players: bytes,
    *,
    escapes: NDArray[np.int64],
    world: World,
    start: Sequence[int],
) -> NDArray[np.int64]:
    """Return where the player stands and faces before each decision and at the end.

    Args:
      players: One episode's player bytes, one per decision.
      escapes: The episode's escape rows, ``[n, 5]``.
      world: The shared world, whose ladders a floor change lands on.
      start: Floor, row, column and facing before decision 0.

    Returns:
      path: int64 ``[T + 1, 4]``: floor, row, column, facing before each
        decision, then after the last.

    """
    escaped = {escapes.item(i, 0): escapes[i, 1:] for i in range(len(escapes))}
    path = np.zeros((len(players) + 1, 4), np.int64)
    path[0] = start
    for t, byte in enumerate(players):
        floor, row, col, facing = (path.item(t, j) for j in range(4))
        action = byte & ACTION_MASK
        if t in escaped:
            path[t + 1] = escaped[t]
            continue
        if action in STEPS:
            facing = action
        if byte & MOVED:
            row += STEPS[Action(action)][0]
            col += STEPS[Action(action)][1]
        elif byte & FLOOR_CHANGE and action == Action.DESCEND:
            floor += 1
            row, col = world.up_ladders.item(floor, 0), world.up_ladders.item(floor, 1)
        elif byte & FLOOR_CHANGE:
            floor -= 1
            row, col = (
                world.down_ladders.item(floor, 0),
                world.down_ladders.item(floor, 1),
            )
        path[t + 1] = floor, row, col, facing
    return path


def interaction_targets(
    players: bytes,
    *,
    path: NDArray[np.int64],
) -> list[tuple[int, ...]]:
    """Return ``(decision, floor, row, column)`` of the tile each interaction faced.

    Args:
      players: One episode's player bytes.
      path: Its :func:`decode_path`.

    Returns:
      targets: One per decision whose action is in ``INTERACTIONS``, in order:
        the tile beside the player in its facing before the decision.

    """
    targets: list[tuple[int, ...]] = []
    for t, byte in enumerate(players):
        if (byte & ACTION_MASK) in INTERACTIONS:
            floor, row, col, facing = (path.item(t, j) for j in range(4))
            step = STEPS[Action(facing)]
            targets.append((t, floor, row + step[0], col + step[1]))
    return targets


def maps_before(
    world: World,
    *,
    map_events: NDArray[np.int64],
    decision: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return one episode's blocks and items before ``decision``.

    Args:
      world: The shared world.
      map_events: The episode's map event rows, ``[n, 6]``.
      decision: The decision whose state is wanted; the episode's length
        gives its end.

    Returns:
      block: uint8 ``[9, 48, 48]``.
      item: uint8 ``[9, 48, 48]``.

    """
    block = world.block.copy()
    item = world.item.copy()
    for i in range(len(map_events)):
        t, floor, row, col, new_block, new_item = (
            map_events.item(i, j) for j in range(6)
        )
        if t >= decision:
            break
        block[floor, row, col] = new_block
        item[floor, row, col] = new_item
    return block, item
