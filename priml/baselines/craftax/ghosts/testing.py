"""Test support for the ghost site: ghosts built by hand, without a replay.

The site's builders read a ghost's arrays and bytes, not the game, so their
tests take ghosts made here, whose lengths and outcomes each test chooses.
"""

from __future__ import annotations

import numpy as np

from priml.baselines.craftax.game.state import ACHIEVEMENT_REWARD_MAP, Achievement
from priml.baselines.craftax.ghosts.extract import ENDED, NEW_TILE, Ghost
from priml.baselines.craftax.ghosts.layout import Events


def made_ghost(
    ordinal: int,
    *,
    decisions: int,
    outcome: str,
    world_seed: int = 1,
    sampling_seed: int | None = None,
    sleeps: tuple[tuple[int, int], ...] = (),
) -> Ghost:
    """Return a ghost of its own bytes, each field telling its ordinal apart.

    It reaches a new tile every 70th decision from its ordinal, so ghosts
    shown together leave quiet runs of 67 decisions, long enough for a
    timeline to collapse. Every 4th decision samples one creature, standing in
    row ``ordinal``, so groups and windows that mix ghosts up show. An odd
    ordinal has an escape row, reaches floor 1 halfway and ends there. Each
    sleep of ``k`` ticks has ``(k - 1) // 4`` samples of one creature, the
    first of them a ripe plant.

    Args:
      ordinal: Its capture ordinal.
      decisions: Its length.
      outcome: How it ends.
      world_seed: Its world.
      sampling_seed: Its receipt's sampling seed; ``100 + ordinal`` if None.
      sleeps: ``(decision, ticks)`` of each sleep.

    Returns:
      ghost: The ghost.

    """
    samples = -(-decisions // 4)
    active = bytearray(NEW_TILE if t % 70 == ordinal else 0 for t in range(decisions))
    active[-1] |= ENDED
    held = [(ticks - 1) // 4 for _, ticks in sleeps]
    changes = [
        (sum(held[:i]), 0, 5, 5 + i, 16, 0) for i, count in enumerate(held) if count
    ]
    return Ghost(
        ordinal=ordinal,
        world_seed=world_seed,
        sampling_seed=100 + ordinal if sampling_seed is None else sampling_seed,
        decisions=decisions,
        outcome=outcome,
        end=(ordinal % 2, 24, 24 + ordinal, 2),
        floor_first=(0, *((decisions // 2,) if ordinal % 2 else (-1,)), *[-1] * 7),
        achievement_return=int(ACHIEVEMENT_REWARD_MAP.item(Achievement.COLLECT_WOOD)),
        players=bytes((ordinal + t) % 43 for t in range(decisions)),
        creatures=b"".join(
            bytes([1, 0x10, ordinal, k % 48, 0]) for k in range(samples)
        ),
        samples=np.arange(samples + 1, dtype=np.int64) * 5,
        events=Events(
            map=np.array([[5, 0, 1, ordinal, 6, 0]], np.int64),
            achievements=np.array([[3, Achievement.COLLECT_WOOD]], np.int64),
            escapes=np.array([[7, 0, 2, 2, 3]] * (ordinal % 2), np.int64).reshape(
                -1,
                5,
            ),
        ),
        active=bytes(active),
        sleeps=np.array(sleeps, np.int64).reshape(-1, 2),
        sleep_samples=np.arange(sum(held) + 1, dtype=np.int64) * 5,
        sleep_creatures=b"".join(bytes([1, 0x10, 7, k, 0]) for k in range(sum(held))),
        sleep_changes=np.array(changes, np.int64).reshape(-1, 6),
    )
