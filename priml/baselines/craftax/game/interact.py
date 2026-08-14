"""The single interact action: mine, gather, drink, eat, open, or strike.

One action covers everything the player can do to the tile they face, and what
happens is decided by what is there. Attacking takes precedence: if a creature
occupies the tile, the blow lands on it and the block is left alone.

Mining is gated on the pickaxe tier, which is the game's whole progression
spine -- stone needs a wood pickaxe, iron needs stone, diamond needs iron, and
the gems need diamond. Nothing else enforces the order in which the world
opens up.
"""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import (
    Achievement,
    BlockType,
    ItemType,
)
from priml.baselines.craftax.game.indexing import (
    gather_tiles,
    scatter_tiles_where,
)
from priml.baselines.craftax.game.state import EnvState


def interact(
    state: EnvState,
    *,
    doing: Tensor,
    generator: torch.Generator | None = None,
) -> EnvState:
    """Apply the interact action to the tile the player faces.

    Args:
      state: The current world.
      doing: Which environments chose to interact, ``[envs]``.
      generator: Source of randomness for sapling and chest draws.

    Returns:
      state: The world after the interaction.

    """
    target = (
        state.player_position
        + constants.DIRECTIONS.to(state.device)[state.player_direction.long()]
    )
    state, struck = _strike_whatever_stands_there(state, target=target, doing=doing)
    # A blow that lands on a creature does not also mine the tile behind it.
    acting = doing & mechanics.in_bounds(target) & ~struck

    block = mechanics.block_at(state, target)
    state = _mine_blocks(state, target=target, block=block, acting=acting)
    state = _gather_ground(
        state,
        target=target,
        block=block,
        acting=acting,
        generator=generator,
    )
    state = _open_chest(state, target=target, block=block, acting=acting)
    return _damage_boss(state, block=block, acting=acting)


def _strike_whatever_stands_there(
    state: EnvState,
    *,
    target: Tensor,
    doing: Tensor,
) -> tuple[EnvState, Tensor]:
    """Hit any creature on the faced tile, across all three classes."""
    damage = mechanics.player_damage(state)
    yes = torch.ones(state.num_envs, dtype=torch.bool, device=state.device)
    struck = torch.zeros(state.num_envs, dtype=torch.bool, device=state.device)
    killed_monster = struck.clone()

    for field, mob_class, can_unlock in (
        ("melee_mobs", 1, yes),
        ("passive_mobs", 0, yes),
        ("ranged_mobs", 2, yes),
    ):
        mobs, killed, hit, achievements = mechanics.attack_mob_class(
            state,
            getattr(state, field),
            position=target,
            damage=damage,
            mob_class=mob_class,
            can_unlock=can_unlock & doing,
        )
        setattr(state, field, mobs)
        state.achievements = torch.where(
            doing[:, None],
            achievements,
            state.achievements,
        )
        struck = struck | (hit & doing)
        if field == "passive_mobs":
            # Killing a cow is eating it, which is the early game's food.
            fed = killed & doing
            state.player_food = torch.where(
                fed,
                (state.player_food + 6).minimum(mechanics.max_food(state)),
                state.player_food,
            )
            state.player_hunger = torch.where(
                fed,
                torch.zeros_like(state.player_hunger),
                state.player_hunger,
            )
        else:
            killed_monster = killed_monster | (killed & doing)

    rows = torch.arange(state.num_envs, device=state.device)
    state.monsters_killed[rows, state.player_level.long()] += killed_monster.int()
    return state, struck


def _mine_blocks(
    state: EnvState,
    *,
    target: Tensor,
    block: Tensor,
    acting: Tensor,
) -> EnvState:
    """Break the faced block if the player's pickaxe is good enough."""
    pickaxe = state.inventory.pickaxe
    # Trees need no tool, which is why wood is the first resource. Each
    # remaining tier is gated by the pickaxe that opens it.
    quarries = (
        (BlockType.TREE, BlockType.GRASS, "wood", 0),
        (BlockType.FIRE_TREE, BlockType.FIRE_GRASS, "wood", 0),
        (BlockType.ICE_SHRUB, BlockType.ICE_GRASS, "wood", 0),
        (BlockType.STONE, BlockType.PATH, "stone", 1),
        (BlockType.STALAGMITE, BlockType.PATH, "stone", 1),
        (BlockType.COAL, BlockType.PATH, "coal", 1),
        (BlockType.IRON, BlockType.PATH, "iron", 2),
        (BlockType.DIAMOND, BlockType.PATH, "diamond", 3),
        (BlockType.SAPPHIRE, BlockType.PATH, "sapphire", 4),
        (BlockType.RUBY, BlockType.PATH, "ruby", 4),
    )
    achievements = {
        BlockType.TREE: Achievement.COLLECT_WOOD,
        BlockType.STONE: Achievement.COLLECT_STONE,
        BlockType.COAL: Achievement.COLLECT_COAL,
        BlockType.IRON: Achievement.COLLECT_IRON,
        BlockType.DIAMOND: Achievement.COLLECT_DIAMOND,
        BlockType.SAPPHIRE: Achievement.COLLECT_SAPPHIRE,
        BlockType.RUBY: Achievement.COLLECT_RUBY,
    }

    for source, leaves, resource, tier in quarries:
        mining = acting & (block == int(source)) & (pickaxe >= tier)
        state = _replace_block(state, target, int(leaves), mining)
        setattr(
            state.inventory,
            resource,
            getattr(state.inventory, resource) + mining.int(),
        )
        if source in achievements:
            state.achievements = mechanics.unlock_achievement(
                state,
                torch.full(
                    (state.num_envs,),
                    int(achievements[source]),
                    device=state.device,
                ),
                mining,
            )

    # A table or furnace can always be reclaimed, but yields nothing: it is
    # removal rather than mining.
    for furniture in (BlockType.CRAFTING_TABLE, BlockType.FURNACE):
        state = _replace_block(
            state,
            target,
            int(BlockType.PATH),
            acting & (block == int(furniture)),
        )
    return state


def _gather_ground(
    state: EnvState,
    *,
    target: Tensor,
    block: Tensor,
    acting: Tensor,
    generator: torch.Generator | None,
) -> EnvState:
    """Take a sapling, a drink, or a ripe plant from the faced tile."""
    # Grass sometimes yields a sapling, which is the only way to farm.
    sapling = (
        acting
        & (block == int(BlockType.GRASS))
        & (torch.rand(state.num_envs, generator=generator, device=state.device) < 0.1)
    )
    state.inventory.sapling = state.inventory.sapling + sapling.int()
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.COLLECT_SAPLING),
            device=state.device,
        ),
        sapling,
    )

    drinking = acting & (
        (block == int(BlockType.WATER)) | (block == int(BlockType.FOUNTAIN))
    )
    state.player_drink = torch.where(
        drinking,
        (state.player_drink + 1).minimum(mechanics.max_drink(state)),
        state.player_drink,
    )
    state.player_thirst = torch.where(
        drinking,
        torch.zeros_like(state.player_thirst),
        state.player_thirst,
    )
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.COLLECT_DRINK),
            device=state.device,
        ),
        drinking,
    )

    eating = acting & (block == int(BlockType.RIPE_PLANT))
    state = _replace_block(state, target, int(BlockType.PLANT), eating)
    state.player_food = torch.where(
        eating,
        (state.player_food + 4).minimum(mechanics.max_food(state)),
        state.player_food,
    )
    state.player_hunger = torch.where(
        eating,
        torch.zeros_like(state.player_hunger),
        state.player_hunger,
    )
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.EAT_PLANT),
            device=state.device,
        ),
        eating,
    )
    # An eaten plant restarts its growth rather than vanishing.
    eaten_here = (state.growing_plants_positions == target[:, None, :]).all(-1)
    state.growing_plants_age = torch.where(
        eaten_here & eating[:, None],
        torch.zeros_like(state.growing_plants_age),
        state.growing_plants_age,
    )
    return state


def _open_chest(
    state: EnvState,
    *,
    target: Tensor,
    block: Tensor,
    acting: Tensor,
) -> EnvState:
    """Empty a chest into the inventory and leave bare path behind."""
    opening = acting & (block == int(BlockType.CHEST))
    state = _replace_block(state, target, int(BlockType.PATH), opening)
    rows = torch.arange(state.num_envs, device=state.device)
    state.chests_opened[rows, state.player_level.long()] |= opening
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.OPEN_CHEST),
            device=state.device,
        ),
        opening,
    )
    return state


def _damage_boss(state: EnvState, *, block: Tensor, acting: Tensor) -> EnvState:
    """Wound the necromancer, but only while its summons are dead."""
    hitting = (
        acting
        & (block == int(BlockType.NECROMANCER))
        & mechanics.is_boss_vulnerable(state)
        & mechanics.is_fighting_boss(state)
    )
    state.boss_progress = state.boss_progress + hitting.int()
    # Each wound starts the next wave, so the fight is a race against the
    # summons rather than a duel.
    state.boss_timesteps_to_spawn_this_round = torch.where(
        hitting,
        torch.full_like(
            state.boss_timesteps_to_spawn_this_round,
            constants.BOSS_FIGHT_SPAWN_TURNS,
        ),
        state.boss_timesteps_to_spawn_this_round,
    )
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.DAMAGE_NECROMANCER),
            device=state.device,
        ),
        hitting,
    )
    return state


def _replace_block(
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


def item_at(state: EnvState, position: Tensor) -> Tensor:
    """The item lying on ``position`` of the player's floor, ``[envs]``."""
    return gather_tiles(mechanics.current_items(state), position)


def is_ladder(item: Tensor, kind: ItemType) -> Tensor:
    """Whether ``item`` is the named ladder."""
    return item == int(kind)
