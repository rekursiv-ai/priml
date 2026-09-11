"""Crafting and block placement.

Every recipe is the same shape -- spend some materials, gain a tool or a
stock, subject to standing near the right station -- so they are written as a
table of :class:`Recipe` rather than as one branch each. That makes the
progression legible in one screen: what each tier costs, and what it needs to
be built beside.

Recipes are applied in tier order within a step, and each checks the stock
left by the previous one. The player only ever takes one action per step, so
at most one recipe can fire; the ordering matters only in that a recipe reads
the running inventory rather than the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from torch import Tensor

import torch

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import (
    Achievement,
    Action,
    BlockType,
    ItemType,
)
from priml.baselines.craftax.game.indexing import (
    scatter_tiles_where,
)
from priml.baselines.craftax.game.state import EnvState


@dataclass(frozen=True, slots=True, kw_only=True)
class Recipe:
    """One thing the player can make.

    Attributes:
      action: The action that attempts this recipe.
      costs: Materials spent, as inventory field to amount.
      tool: The tool field this upgrades, if any, and the tier it reaches. A
        tool recipe is refused when the player already has that tier.
      stock: A stock field this adds to, and how much.
      needs_table: Whether a crafting table must be adjacent.
      needs_furnace: Whether a furnace must be adjacent.
      achievement: The achievement making this unlocks.

    """

    action: Action
    costs: dict[str, int] = field(default_factory=dict)
    tool: tuple[str, int] | None = None
    stock: tuple[str, int] | None = None
    needs_table: bool = True
    needs_furnace: bool = False
    achievement: Achievement | None = None


RECIPES: Final = (
    Recipe(
        action=Action.MAKE_WOOD_PICKAXE,
        costs={"wood": 1},
        tool=("pickaxe", 1),
        achievement=Achievement.MAKE_WOOD_PICKAXE,
    ),
    Recipe(
        action=Action.MAKE_STONE_PICKAXE,
        costs={"wood": 1, "stone": 1},
        tool=("pickaxe", 2),
        achievement=Achievement.MAKE_STONE_PICKAXE,
    ),
    Recipe(
        action=Action.MAKE_IRON_PICKAXE,
        costs={"wood": 1, "stone": 1, "iron": 1, "coal": 1},
        tool=("pickaxe", 3),
        needs_furnace=True,
        achievement=Achievement.MAKE_IRON_PICKAXE,
    ),
    Recipe(
        action=Action.MAKE_DIAMOND_PICKAXE,
        costs={"wood": 1, "diamond": 3},
        tool=("pickaxe", 4),
        achievement=Achievement.MAKE_DIAMOND_PICKAXE,
    ),
    Recipe(
        action=Action.MAKE_WOOD_SWORD,
        costs={"wood": 1},
        tool=("sword", 1),
        achievement=Achievement.MAKE_WOOD_SWORD,
    ),
    Recipe(
        action=Action.MAKE_STONE_SWORD,
        costs={"wood": 1, "stone": 1},
        tool=("sword", 2),
        achievement=Achievement.MAKE_STONE_SWORD,
    ),
    Recipe(
        action=Action.MAKE_IRON_SWORD,
        costs={"wood": 1, "stone": 1, "iron": 1, "coal": 1},
        tool=("sword", 3),
        needs_furnace=True,
        achievement=Achievement.MAKE_IRON_SWORD,
    ),
    Recipe(
        action=Action.MAKE_DIAMOND_SWORD,
        costs={"wood": 1, "diamond": 2},
        tool=("sword", 4),
        achievement=Achievement.MAKE_DIAMOND_SWORD,
    ),
    Recipe(
        action=Action.MAKE_ARROW,
        costs={"wood": 1, "stone": 1},
        stock=("arrows", 2),
        achievement=Achievement.MAKE_ARROW,
    ),
    Recipe(
        action=Action.MAKE_TORCH,
        costs={"wood": 1, "coal": 1},
        stock=("torches", 4),
        achievement=Achievement.MAKE_TORCH,
    ),
)
"""Every recipe with a scalar output, in tier order."""


def craft(state: EnvState, action: Tensor) -> EnvState:
    """Apply whichever recipe the action names, if it can be afforded.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with materials spent and the product gained.

    """
    near_table = mechanics.is_near_block(state, int(BlockType.CRAFTING_TABLE))
    near_furnace = mechanics.is_near_block(state, int(BlockType.FURNACE))

    for recipe in RECIPES:
        making = action == int(recipe.action)
        if recipe.needs_table:
            making = making & near_table
        if recipe.needs_furnace:
            making = making & near_furnace
        for material, amount in recipe.costs.items():
            making = making & (getattr(state.inventory, material) >= amount)
        if recipe.tool is not None:
            # A tier already held cannot be re-crafted, which stops the
            # player from spending materials to downgrade.
            name, tier = recipe.tool
            making = making & (getattr(state.inventory, name) < tier)

        for material, amount in recipe.costs.items():
            setattr(
                state.inventory,
                material,
                getattr(state.inventory, material) - amount * making.int(),
            )
        if recipe.tool is not None:
            name, tier = recipe.tool
            setattr(
                state.inventory,
                name,
                torch.where(
                    making,
                    torch.full_like(getattr(state.inventory, name), tier),
                    getattr(state.inventory, name),
                ),
            )
        if recipe.stock is not None:
            name, amount = recipe.stock
            setattr(
                state.inventory,
                name,
                getattr(state.inventory, name) + amount * making.int(),
            )
        if recipe.achievement is not None:
            state.achievements = mechanics.unlock_achievement(
                state,
                torch.full(
                    (state.num_envs,),
                    int(recipe.achievement),
                    device=state.device,
                ),
                making,
            )
    return _craft_armour(
        state, action, near_table=near_table, near_furnace=near_furnace
    )


def place(state: EnvState, action: Tensor) -> EnvState:
    """Put a block or torch from the inventory onto the faced tile.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with the block placed and the stock spent.

    """
    target = (
        state.player_position
        + constants.DIRECTIONS.to(state.device)[state.player_direction.long()]
    )
    block = mechanics.block_at(state, target)
    # Only loose ground accepts a block, and never on top of a creature.
    free = (
        mechanics.in_bounds(target)
        & constants.CAN_PLACE_ITEM_ON.to(state.device)[block.long()]
        & ~mechanics.is_occupied(state, target)
    )

    placements = (
        (Action.PLACE_STONE, "stone", 1, BlockType.STONE, Achievement.PLACE_STONE),
        (
            Action.PLACE_TABLE,
            "wood",
            # A table is the one placement that costs more than a single
            # unit, which is what makes the first one a real decision.
            1,
            BlockType.CRAFTING_TABLE,
            Achievement.PLACE_TABLE,
        ),
        (
            Action.PLACE_FURNACE,
            "stone",
            1,
            BlockType.FURNACE,
            Achievement.PLACE_FURNACE,
        ),
        (
            Action.PLACE_PLANT,
            "sapling",
            1,
            BlockType.PLANT,
            Achievement.PLACE_PLANT,
        ),
    )
    for action_kind, material, cost, block_kind, achievement in placements:
        placing = (
            (action == int(action_kind))
            & free
            & (getattr(state.inventory, material) >= cost)
        )
        state = _write_block(state, target, int(block_kind), placing)
        setattr(
            state.inventory,
            material,
            getattr(state.inventory, material) - cost * placing.int(),
        )
        state.achievements = mechanics.unlock_achievement(
            state,
            torch.full((state.num_envs,), int(achievement), device=state.device),
            placing,
        )
        if action_kind == Action.PLACE_PLANT:
            state = _sow_plant(state, target, placing)

    return _place_torch(state, target, action)


def _craft_armour(
    state: EnvState,
    action: Tensor,
    *,
    near_table: Tensor,
    near_furnace: Tensor,
) -> EnvState:
    """Make a full set of armour, which fills all four body slots at once."""
    recipes = (
        (
            Action.MAKE_IRON_ARMOUR,
            {"iron": 3, "coal": 3},
            1,
            True,
            Achievement.MAKE_IRON_ARMOUR,
        ),
        (
            Action.MAKE_DIAMOND_ARMOUR,
            {"diamond": 3},
            2,
            False,
            Achievement.MAKE_DIAMOND_ARMOUR,
        ),
    )
    for action_kind, costs, tier, needs_furnace, achievement in recipes:
        making = (action == int(action_kind)) & near_table
        if needs_furnace:
            making = making & near_furnace
        for material, amount in costs.items():
            making = making & (getattr(state.inventory, material) >= amount)
        # Armour is made a piece at a time: the recipe fills the first slot
        # that is not already at this tier.
        upgradeable = (state.inventory.armour < tier).any(-1)
        making = making & upgradeable
        slot = (state.inventory.armour < tier).int().argmax(-1)

        for material, amount in costs.items():
            setattr(
                state.inventory,
                material,
                getattr(state.inventory, material) - amount * making.int(),
            )
        rows = torch.arange(state.num_envs, device=state.device)
        current = state.inventory.armour[rows, slot]
        state.inventory.armour[rows, slot] = torch.where(making, tier, current)
        state.achievements = mechanics.unlock_achievement(
            state,
            torch.full((state.num_envs,), int(achievement), device=state.device),
            making,
        )
    return state


def _place_torch(state: EnvState, target: Tensor, action: Tensor) -> EnvState:
    """Set a torch down and light the tiles around it."""
    placing = (
        (action == int(Action.PLACE_TORCH))
        & (state.inventory.torches >= 1)
        & mechanics.in_bounds(target)
        & ~constants.SOLID_BLOCK.to(state.device)[
            mechanics.block_at(state, target).long()
        ]
    )
    state.inventory.torches = state.inventory.torches - placing.int()

    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    state.item_map[rows, level] = scatter_tiles_where(
        state.item_map[rows, level],
        target,
        torch.full((state.num_envs,), int(ItemType.TORCH), device=state.device),
        placing,
    )

    # The 9x9 glow is written in ONE scatter, not eighty-one. Iterating the
    # patch in Python cost 162 tensor dispatches for a tile nobody usually
    # places, and this step runs a few thousand dispatches already.
    #
    # A torch brightens a tile, never dims one already brighter, so the write
    # takes a maximum against what is there -- and because two patch cells
    # never address the same tile, the maxima do not need sequencing.
    glow = constants.TORCH_LIGHT_MAP.to(state.device)
    offsets = torch.arange(9, device=state.device) - 4
    light = state.light_map[rows, level]

    patch_rows = (target[:, 0, None, None] + offsets[None, :, None]).clamp(
        0,
        light.shape[-2] - 1,
    )
    patch_columns = (target[:, 1, None, None] + offsets[None, None, :]).clamp(
        0,
        light.shape[-1] - 1,
    )
    inside = (
        (target[:, 0, None, None] + offsets[None, :, None] >= 0)
        & (target[:, 0, None, None] + offsets[None, :, None] < light.shape[-2])
        & (target[:, 1, None, None] + offsets[None, None, :] >= 0)
        & (target[:, 1, None, None] + offsets[None, None, :] < light.shape[-1])
    )
    env = rows[:, None, None].expand_as(patch_rows)
    brightened = torch.maximum(light[env, patch_rows, patch_columns], glow)
    light = light.clone()
    light[env, patch_rows, patch_columns] = torch.where(
        placing[:, None, None] & inside,
        brightened,
        light[env, patch_rows, patch_columns],
    )
    state.light_map[rows, level] = light
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.PLACE_TORCH),
            device=state.device,
        ),
        placing,
    )
    return state


def _sow_plant(state: EnvState, target: Tensor, placing: Tensor) -> EnvState:
    """Record a sown plant so it can ripen over the coming steps."""
    free = ~state.growing_plants_mask
    slot = free.int().argmax(-1)
    rows = torch.arange(state.num_envs, device=state.device)
    sowing = placing & free.any(-1)
    state.growing_plants_positions[rows, slot] = torch.where(
        sowing[:, None],
        target.int(),
        state.growing_plants_positions[rows, slot],
    )
    state.growing_plants_age[rows, slot] = torch.where(
        sowing,
        torch.zeros_like(slot, dtype=torch.int32),
        state.growing_plants_age[rows, slot],
    )
    state.growing_plants_mask[rows, slot] = (
        state.growing_plants_mask[rows, slot] | sowing
    )
    return state


def _write_block(
    state: EnvState,
    target: Tensor,
    block: int,
    applies: Tensor,
) -> EnvState:
    """Write one block on the player's floor where ``applies``."""
    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    state.map[rows, level] = scatter_tiles_where(
        state.map[rows, level],
        target,
        torch.full((state.num_envs,), block, device=state.device),
        applies,
    )
    return state
