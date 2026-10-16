"""Test support for the viewer's bundles: a few decisions of the game, a sleep among them.

The episode plays the game's kernels as Python (``eager``) on the tiny world:
the player turns against a cow penned at its left, sleeps nine ticks, turns
right and waits, and the clock runs out at the last decision. Its tests run
inside ``eager(world=world)``, where a reset of any seed builds that world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from priml.baselines.craftax.eager import scripted, tiny_world
from priml.baselines.craftax.game import rules
from priml.baselines.craftax.game.state import (
    DEFAULT_MAX_TIMESTEPS,
    MAP_SIZE,
    Action,
    BlockType,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import Episode, Record


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import Array1, EnvState


SCRIPT: Final = (Action.LEFT, Action.SLEEP, Action.RIGHT, Action.NOOP)
"""The episode's actions."""

SLEEP_TICKS: Final = 9
"""Ticks the sleep plays: two frames at a stride of 4."""

COW: Final = (MAP_SIZE // 2, MAP_SIZE // 2 - 1)
"""Where the penned cow stands, left of the start."""


def world(state: EnvState, rng: Array1[np.uint32]) -> None:
    """Fill the tiny world, a cow penned in view, the player one energy short.

    The player's fatigue makes the sleep wake after :data:`SLEEP_TICKS`
    ticks, and the clock runs out at the episode's last decision.

    Args:
      state: A zeroed world record, filled in place.
      rng: The world's stream.

    """
    tiny_world(
        state,
        rng,
        timestep=DEFAULT_MAX_TIMESTEPS - (len(SCRIPT) - 1 + SLEEP_TICKS),
    )
    row, col = COW
    for dr, dc in ((-1, 0), (1, 0), (0, -1)):
        rules.set_block_numba(state, 0, row + dr, col + dc, BlockType.STONE)
    cows = state.passive_mobs[0]
    cows.mask[0], cows.type_id[0], cows.position[0, 0], cows.position[0, 1] = (
        1,
        0,
        row,
        col,
    )
    rules.set_mob_bit_numba(state, 0, row, col, True)
    # Asleep, the fatigue drops a point a tick and, past -10, makes the energy
    # whole: a sleep from fatigue f ends at its tick f + 12. The turn before it
    # adds a point.
    state.player_energy = 8
    state.player_fatigue = np.float32(SLEEP_TICKS - 13)


def record() -> Record:
    """Return the episode as capture records it, of world 4 and sampling seed 2.

    Call it inside ``eager(world=world)``.

    Returns:
      record: The scripted episode's record.

    """
    return scripted([int(action) for action in SCRIPT], world_seed=4, sampling_seed=2)


def episode() -> Episode:
    """Return the episode with its token frames: :func:`record`, replayed.

    Call it inside ``eager(world=world)``.

    Returns:
      episode: The scripted episode, its frames replayed.

    """
    played = record()
    empty = torch.empty(0)
    return replay.replay(
        Episode(
            receipt=played.receipt,
            actions=played.actions,
            hashes=played.hashes,
            cells=empty,
            aux=empty,
            reward=empty,
            done=empty,
            summary={},
        ),
    )
