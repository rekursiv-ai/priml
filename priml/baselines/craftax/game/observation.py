"""The symbolic observation: what the agent actually sees.

The view is a 9x11 window centered on the player, one-hot encoded per channel
-- block, item, and one plane per creature class and species -- followed by the
inventory and the player's condition as scalars. Everything outside the lit
region is zeroed, so darkness genuinely hides the world rather than merely
dimming it.

Counts are compressed with a square root before being scaled: the difference
between one and two logs matters far more than between ninety and ninety-one,
and the root gives the early differences room without letting a full stack
saturate.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import BlockType, ItemType
from priml.baselines.craftax.game.indexing import local_view
from priml.baselines.craftax.game.state import EnvState


OBSERVATION_SIZE: int = (
    constants.OBS_DIM[0]
    * constants.OBS_DIM[1]
    * (
        len(BlockType)
        + len(ItemType)
        + 5 * 8  # five creature classes, eight species each
        + 1  # whether the tile is lit
    )
    + constants.INVENTORY_OBS_SIZE
)
"""Width of one flattened observation."""


def render(state: EnvState) -> Tensor:
    """Encode the world as the agent sees it.

    Args:
      state: The current world.

    Returns:
      observation: One row per environment, ``[envs, OBSERVATION_SIZE]``.

    """
    return torch.cat((_render_view(state), _render_player(state)), dim=-1)


def _render_view(state: EnvState) -> Tensor:
    """Encode the lit window around the player as one-hot planes."""
    view = constants.OBS_DIM
    blocks = local_view(
        mechanics.current_map(state),
        state.player_position,
        view,
        outside=int(BlockType.OUT_OF_BOUNDS),
    )
    items = local_view(
        mechanics.current_items(state),
        state.player_position,
        view,
        outside=int(ItemType.NONE),
    )
    planes = torch.cat(
        (
            functional.one_hot(blocks.long(), len(BlockType)).float(),
            functional.one_hot(items.long(), len(ItemType)).float(),
            _render_mobs(state),
        ),
        dim=-1,
    )

    # A tile below the light threshold is not shown at all: this is what makes
    # a torch worth carrying rather than a convenience.
    lit = (
        local_view(
            mechanics.current_light(state),
            state.player_position,
            view,
            outside=0.0,
        )
        > 0.05
    ).float()
    planes = planes * lit[..., None]
    return torch.cat((planes, lit[..., None]), dim=-1).flatten(1)


def _render_mobs(state: EnvState) -> Tensor:
    """Mark each visible creature on the plane for its class and species."""
    rows, columns = constants.OBS_DIM
    planes = torch.zeros(
        (state.num_envs, rows, columns, 5 * 8),
        device=state.device,
    )
    corner = state.player_position - torch.tensor(
        [rows // 2, columns // 2],
        device=state.device,
    )
    classes = (
        state.melee_mobs,
        state.passive_mobs,
        state.ranged_mobs,
        state.mob_projectiles,
        state.player_projectiles,
    )
    # The reference stores melee first but encodes passive first, so the plane
    # order here is the ENCODING order, not the state's field order.
    encoded_class = (1, 0, 2, 3, 4)
    index = torch.arange(state.num_envs, device=state.device)
    for mobs, plane in zip(classes, encoded_class, strict=True):
        level = state.player_level.long()
        for slot in range(mobs.mask.shape[-1]):
            alive = mobs.mask[index, level, slot]
            local = mobs.position[index, level, slot] - corner
            visible = (
                alive
                & (local[:, 0] >= 0)
                & (local[:, 0] < rows)
                & (local[:, 1] >= 0)
                & (local[:, 1] < columns)
            )
            channel = plane * 8 + mobs.type_id[index, level, slot].long()
            # The write is masked as well as clamped, and accumulates rather
            # than assigns: a clamped-only assignment would paint an
            # off-screen creature onto the edge of the view, and a plain
            # assignment would erase one already marked on that tile.
            row = local[:, 0].clamp(0, rows - 1)
            column = local[:, 1].clamp(0, columns - 1)
            planes[index, row, column, channel] = torch.maximum(
                planes[index, row, column, channel],
                visible.float(),
            )
    return planes


def _render_player(state: EnvState) -> Tensor:
    """Encode the inventory, meters, and condition as scalars."""
    inventory = state.inventory
    counts = torch.stack(
        (
            inventory.wood,
            inventory.stone,
            inventory.coal,
            inventory.iron,
            inventory.diamond,
            inventory.sapphire,
            inventory.ruby,
            inventory.sapling,
            inventory.torches,
            inventory.arrows,
        ),
        dim=-1,
    ).float()
    tools = torch.stack(
        (
            inventory.books.float() / 2.0,
            inventory.pickaxe.float() / 4.0,
            inventory.sword.float() / 4.0,
            state.sword_enchantment.float(),
            state.bow_enchantment.float(),
            inventory.bow.float(),
        ),
        dim=-1,
    )
    meters = torch.stack(
        (
            state.player_health,
            state.player_food.float(),
            state.player_drink.float(),
            state.player_energy.float(),
            state.player_mana.float(),
            state.player_xp.float(),
            state.player_dexterity.float(),
            state.player_strength.float(),
            state.player_intelligence.float(),
        ),
        dim=-1,
    )
    rows = torch.arange(state.num_envs, device=state.device)
    condition = torch.stack(
        (
            state.light_level,
            state.is_sleeping.float(),
            state.is_resting.float(),
            state.learned_spells[:, 0].float(),
            state.learned_spells[:, 1].float(),
            state.player_level.float() / 10.0,
            (
                state.monsters_killed[rows, state.player_level.long()]
                >= constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
            ).float(),
            mechanics.is_boss_vulnerable(state).float(),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            counts.sqrt() / 10.0,
            tools,
            inventory.potions.float().sqrt() / 10.0,
            meters / 10.0,
            functional.one_hot(
                (state.player_direction.long() - 1).clamp(min=0),
                4,
            ).float(),
            inventory.armour.float() / 2.0,
            state.armour_enchantments.float(),
            condition,
        ),
        dim=-1,
    )
