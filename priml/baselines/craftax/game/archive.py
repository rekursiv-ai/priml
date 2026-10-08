"""Practice's archive: worlds saved by achievement return, to restore rows from.

A training option, which ``practice.FrontierPractice`` one directory up builds
and drives. The last ``donors`` environments play natural episodes and watch
their achievement return in levels ``level_width`` points wide (return levels,
not floors). On the step an episode first crosses into a level above every one
it has reached, its donor saves the whole environment into that level: the
world, its stream, the accumulators and stall clocks, and the action and
reward that led there. Level 0 is never saved, since no step crosses into it.
Between rollouts, :func:`restore_rows_numba` puts entries into the first rows,
which then play practice branches from them.

A level holds ``per_level`` entries, at most ``per_world`` of them from one
world: the 64-bit FNV-1a hash of the map at its episode's first step. A world
at that cap replaces one of its own entries, and a full level any entry, both
drawn from the archive's own ``rand_r`` stream. ``reach[level]`` counts the
donor episodes that reached each level, every count multiplied by
``reach_decay`` at each donor episode start, so it remembers about
``1 / (1 - reach_decay)`` episodes. A restore draws a level with weight
``1 / max(1, reach)`` among the populated levels, then an entry uniformly, from
the same stream.

The order is fixed. The donors lie in one buffer, and :func:`after_step_numba` runs
on that buffer's lead thread once every share of a step has joined, observing
the donors in ascending order; restores run between rollouts on the calling
thread. So the stream's draws, the archive and ``reach`` follow the steps
alone, whatever the thread count.

These are kernels in ``game/`` because the env's ``nogil`` buffer loop calls
:func:`after_step_numba`, and a kernel outside ``game/`` may call only ``game/``'s
kernels and its own file's (``jit.py`` says why), so they could not live beside
``FrontierPractice``. They are scalar Numba loops over numpy records for the
reason the step is (``game``'s package docstring).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, SupportsIndex

import numpy as np

from priml.baselines.craftax.game.jit import clampi_numba, jit
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.step import (
    observe_numba,
    write_previous_action_numba,
)


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import ArchiveView, Array3
    from priml.baselines.craftax.game.step import BatchView


FNV_BASIS: Final = np.uint64(0xCBF2_9CE4_8422_2325)
FNV_PRIME: Final = np.uint64(0x0000_0100_0000_01B3)
"""64-bit FNV-1a's offset basis and prime."""

DRAWS: Final = 2_147_483_648.0
"""``RAND_MAX + 1`` for glibc's ``rand_r``: a draw over it is uniform on ``[0, 1)``."""


@jit
def save_numba(batch: BatchView, row: int, archive: ArchiveView, slot: int) -> None:
    """Save environment ``row`` into entry ``slot``: world, stream, stats, action, reward.

    Args:
      batch: The environments, with ``TRAINING_STATS_DTYPE`` stats.
      row: The environment, between steps.
      archive: The archive.
      slot: The entry to overwrite.

    """
    archive.states[slot] = batch.states[row]
    archive.rngs[slot] = batch.rngs[row]
    archive.stats[slot] = batch.stats[row]
    archive.actions[slot] = batch.actions[row, 0]
    archive.rewards[slot] = batch.rewards[row]


@jit
def restore_numba(
    batch: BatchView,
    row: int,
    archive: ArchiveView,
    slot: SupportsIndex,
) -> None:
    """Put entry ``slot`` into environment ``row`` as a practice branch.

    The world, its stream, the accumulators and the stall clocks become the
    saved ones, so the row plays on exactly as the donor would have; the row
    keeps its own episode log and escape stream. The observation and mask are
    computed afresh, the previous-action field is the saved action, the reward
    the saved step's, the terminal 0, and the row plays a branch, whose end is
    not logged. The natural episode it replaces never ends, so it is never
    logged either.

    Args:
      batch: The environments, with ``TRAINING_STATS_DTYPE`` stats.
      row: The environment, between steps.
      archive: The archive.
      slot: The entry to restore.

    """
    batch.states[row] = archive.states[slot]
    batch.rngs[row] = archive.rngs[slot]
    stats = batch.stats[row]
    saved = archive.stats[slot]
    stats.episode_return_accum = saved.episode_return_accum
    stats.episode_length_accum = saved.episode_length_accum
    stats.max_floor_accum = saved.max_floor_accum
    stats.steps = saved.steps
    stats.last_ticks = saved.last_ticks
    stats.undefined_spawns = saved.undefined_spawns
    stats.first_undefined_step = saved.first_undefined_step
    stats.stall_limit = saved.stall_limit
    stats.last_gain = saved.last_gain
    stats.branch = 1
    obs = batch.observations[row]
    observe_numba(batch.states[row], obs, batch.masks[row], batch.rules)
    write_previous_action_numba(obs, int(archive.actions[slot]), batch.rules)
    batch.rewards[row] = archive.rewards[slot]
    batch.terminals[row] = np.float32(0.0)


@jit
def after_step_numba(
    batch: BatchView,
    archive: ArchiveView | None,
    start: int,
    stop: int,
) -> None:
    """Close a buffer step of rows ``start..stop-1`` for practice; nothing without it.

    Run once per buffer step, after every share has joined. It counts the
    buffer's steps, clears the rows' restore slots (valid for one step), and
    observes the buffer's donors in order. Each row counted its own branch
    steps as it stepped.

    Args:
      batch: The environments.
      archive: The archive; None prunes everything.
      start: The buffer's first row.
      stop: One past its last.

    """
    if archive is not None:
        archive.restore_slots[start:stop] = -1
        archive.counts[start // archive.envs_per_buffer] += stop - start
        observe_donors_numba(batch, archive, start, stop)


@jit
def observe_donors_numba(
    batch: BatchView,
    archive: ArchiveView,
    start: int,
    stop: int,
) -> None:
    """Observe every donor among rows ``start..stop-1``, in ascending order.

    A donor whose last step ended its episode, or that has not been observed
    since :func:`restart_donors_numba`, starts an episode: its world is hashed and
    every reach decays. Otherwise a step that crossed into a new return level
    is saved, and the donor's ``save_slots`` entry names where; it is -1 on
    every other step.

    Args:
      batch: The environments, between steps.
      archive: The archive.
      start: The first row to consider.
      stop: One past the last.

    """
    for row in range(max(start, archive.first_donor), stop):
        _observe_donor_numba(batch, archive, row)


@jit
def restart_donors_numba(batch: BatchView, archive: ArchiveView) -> None:
    """Start every donor's episode afresh after a reset, and observe it.

    Args:
      batch: The environments, just reset.
      archive: The archive.

    """
    for donor in range(archive.donor_steps.shape[0]):
        archive.donor_steps[donor] = 0
    observe_donors_numba(batch, archive, archive.first_donor, batch.states.shape[0])


@jit
def restore_rows_numba(batch: BatchView, archive: ArchiveView, selected: int) -> int:
    """Restore rows ``0..selected-1`` from entries drawn from the archive's stream.

    Each row draws a level (:func:`draw_level_numba`), then one of its entries
    uniformly, and is restored from it (:func:`restore_numba`); ``restore_slots``
    names each row's entry, and -1 for every other row.

    Args:
      batch: The environments, between rollouts.
      archive: The archive, holding at least one entry when ``selected``.
      selected: Rows to restore.

    Returns:
      levels: The sum of the levels drawn.

    """
    archive.restore_slots[:] = -1
    if selected == 0:
        return 0
    level_weights_numba(archive)
    per_level = archive.per_level
    total = 0
    for row in range(selected):
        level = draw_level_numba(archive)
        slot = level * per_level + rand_r_numba(archive.stream) % archive.sizes[level]
        restore_numba(batch, row, archive, slot)
        archive.restore_slots[row] = slot
        total += level
    return total


@jit
def level_weights_numba(archive: ArchiveView) -> None:
    """Set ``weights`` to ``1 / max(1, reach)`` over the populated levels, normalized.

    Args:
      archive: The archive, holding at least one entry.

    """
    total = 0.0
    for level in range(archive.sizes.shape[0]):
        archive.weights[level] = 0.0
        if archive.sizes[level]:
            archive.weights[level] = 1.0 / max(1.0, archive.reach[level])
            total += archive.weights[level]
    for level in range(archive.sizes.shape[0]):
        archive.weights[level] /= total


@jit
def draw_level_numba(archive: ArchiveView) -> int:
    """Draw a level by ``weights``: the first whose cumulative weight exceeds the draw.

    Rounding can leave the cumulative sum below a draw near 1; the highest
    populated level catches it.

    Args:
      archive: The archive, its ``weights`` set by :func:`level_weights_numba`.

    Returns:
      level: The level drawn.

    """
    draw = rand_r_numba(archive.stream) / DRAWS
    cumulative = 0.0
    highest = 0
    for level in range(archive.weights.shape[0]):
        cumulative += archive.weights[level]
        if draw < cumulative:
            return level
        if archive.sizes[level]:
            highest = level
    return highest


@jit
def return_level_numba(episode_return: np.float32, archive: ArchiveView) -> int:
    """Return the level of an achievement return: ``return / level_width``, truncated.

    Args:
      episode_return: The episode's achievement return so far.
      archive: For the width and the level count; a return past the last
        level counts as the last.

    Returns:
      level: The level.

    """
    level: int = clampi_numba(
        int(episode_return / archive.level_width),
        0,
        archive.sizes.shape[0] - 1,
    )
    return level


@jit
def record_level_numba(
    archive: ArchiveView,
    donor: int,
    level: int,
    start: bool,
) -> bool:
    """Count a donor's episode into ``reach`` and return whether it crossed a new level.

    An episode start decays every reach and counts the episode at level 0; each
    level the episode reaches for the first time counts it again.

    Args:
      archive: The archive.
      donor: Which donor.
      level: Its episode's level now.
      start: Whether this observation starts the episode.

    Returns:
      crossed: Whether ``level`` is above every level the episode reached before.

    """
    if start:
        for index in range(archive.reach.shape[0]):
            archive.reach[index] *= archive.reach_decay
        archive.reach[0] += 1.0
        archive.donor_levels[donor] = 0
    highest = archive.donor_levels[donor]
    for index in range(highest + 1, level + 1):
        archive.reach[index] += 1.0
    archive.donor_levels[donor] = max(highest, level)
    return level > highest


@jit
def insert_numba(archive: ArchiveView, level: int, world: np.uint64) -> int:
    """Choose the entry of ``level`` a new save of ``world`` takes, and claim it.

    A world with ``per_world`` entries in the level replaces one of them; else
    a level with room appends; else a full level replaces any entry. Both
    replacements draw from the archive's stream.

    Args:
      archive: The archive.
      level: The level saved into.
      world: The saved episode's world hash.

    Returns:
      slot: The entry to save into.

    """
    per_level = archive.per_level
    base = level * per_level
    size = int(archive.sizes[level])
    same = 0
    for index in range(size):
        if archive.worlds[base + index] == world:
            same += 1
    slot = base + size
    if same >= archive.per_world:
        pick = int(rand_r_numba(archive.stream)) % same
        for index in range(size):
            if archive.worlds[base + index] == world:
                if pick == 0:
                    slot = base + index
                    break
                pick -= 1
    elif size < per_level:
        archive.sizes[level] = size + 1
    else:
        slot = base + int(rand_r_numba(archive.stream)) % size
    archive.worlds[slot] = world
    return slot


@jit
def world_hash_numba(grid: Array3[int]) -> np.uint64:
    """Return the 64-bit FNV-1a hash of a ``uint8`` grid's bytes in memory order.

    Args:
      grid: ``uint8 [levels, rows, columns]``, a world's map.

    Returns:
      world: The hash; worlds with equal maps have equal hashes.

    """
    value = FNV_BASIS
    for level in range(grid.shape[0]):
        for row in range(grid.shape[1]):
            for col in range(grid.shape[2]):
                value = (value ^ np.uint64(grid[level, row, col])) * FNV_PRIME
    return value


@jit
def _observe_donor_numba(batch: BatchView, archive: ArchiveView, row: int) -> None:
    """Observe donor ``row`` between steps; save its world if it crossed a new level."""
    donor = row - archive.first_donor
    archive.save_slots[row] = -1
    start = batch.terminals[row] != 0 or archive.donor_steps[donor] == 0
    if start:
        archive.donor_worlds[donor] = world_hash_numba(batch.states[row].map)
        archive.donor_steps[donor] = 0
    level = return_level_numba(batch.stats[row].episode_return_accum, archive)
    crossed = record_level_numba(archive, donor, level, start)
    archive.donor_steps[donor] += 1
    if crossed and not start:
        slot = insert_numba(archive, level, archive.donor_worlds[donor])
        # An earlier donor of this step whose new entry this one replaced saved
        # nothing that survives: two carries scattered into one slot would race.
        for other in range(archive.first_donor, row):
            if archive.save_slots[other] == slot:
                archive.save_slots[other] = -1
        save_numba(batch, row, archive, slot)
        archive.save_slots[row] = slot
