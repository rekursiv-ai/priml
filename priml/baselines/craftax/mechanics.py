"""Shared rules the game's actions are built from.

These are the pieces used in more than one place: how much a hit lands for,
what the player's meters cap at, whether a tile can be stood on, and how a
creature is struck. Keeping them here means an action module states what it
does rather than restating arithmetic, and each rule is checkable on its own.

Every function is batched over a leading environment axis.
"""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.craftax import constants
from priml.baselines.craftax.constants import BlockType
from priml.baselines.craftax.indexing import gather_tiles
from priml.baselines.craftax.state import EnvState, Mobs


def max_health(state: EnvState) -> Tensor:
    """Health cap, which strength raises."""
    return 8 + state.player_strength


def max_food(state: EnvState) -> Tensor:
    """Food cap, which dexterity raises."""
    return 7 + 2 * state.player_dexterity


def max_drink(state: EnvState) -> Tensor:
    """Water cap, which dexterity raises."""
    return 7 + 2 * state.player_dexterity


def max_energy(state: EnvState) -> Tensor:
    """Energy cap, which dexterity raises."""
    return 7 + 2 * state.player_dexterity


def max_mana(state: EnvState) -> Tensor:
    """Mana cap, which intelligence raises."""
    return 6 + 3 * state.player_intelligence


def player_damage(state: EnvState) -> Tensor:
    """Return the player's outgoing damage as physical, fire, and ice.

    Damage comes from the sword tier; strength scales the physical component
    up to double at the cap, while an enchantment adds half the physical
    damage again in its element, which intelligence scales more gently.

    Args:
      state: The current world.

    Returns:
      damage: Damage by element, ``[envs, 3]``.

    """
    physical = torch.tensor(
        [1.0, 2.0, 3.0, 5.0, 8.0],
        device=state.device,
    )[state.inventory.sword.long()]
    fire = physical * (state.sword_enchantment == 1) * 0.5
    ice = physical * (state.sword_enchantment == 2) * 0.5
    physical = physical * (1 + 0.25 * (state.player_strength - 1))
    scale = 1 + 0.05 * (state.player_intelligence - 1)
    return torch.stack((physical, fire * scale, ice * scale), dim=-1)


def damage_to_player(state: EnvState, damage: Tensor) -> Tensor:
    """Return the damage a hit lands on the player after their armour.

    Each armour piece blocks a tenth of physical damage, and an enchanted
    piece blocks a fifth of its element. The boss floor multiplies incoming
    damage, which is what makes the final fight lethal rather than long.

    Args:
      state: The current world.
      damage: Incoming damage by element, ``[envs, 3]``.

    Returns:
      total: Damage actually taken, ``[envs]``.

    """
    defense = torch.stack(
        (
            state.inventory.armour * 0.1,
            (state.armour_enchantments == 1) * 0.2,
            (state.armour_enchantments == 2) * 0.2,
        ),
        dim=1,
    ).sum(-1)
    scaled = damage * (
        1 + is_fighting_boss(state).float()[:, None] * constants.BOSS_FIGHT_EXTRA_DAMAGE
    )
    return apply_defense(scaled, defense)


def apply_defense(damage: Tensor, defense: Tensor) -> Tensor:
    """Reduce each element of ``damage`` by its defense and sum the result.

    Args:
      damage: Incoming damage by element, ``[envs, 3]``.
      defense: Fraction of each element resisted, ``[envs, 3]``.

    Returns:
      total: Damage that lands, ``[envs]``.

    """
    return ((1.0 - defense) * damage).sum(-1)


def is_fighting_boss(state: EnvState) -> Tensor:
    """Whether the player stands on the final floor."""
    return state.player_level == constants.NUM_LEVELS - 1


def is_boss_vulnerable(state: EnvState) -> Tensor:
    """Whether the boss can currently be struck.

    The necromancer is shielded while any of its summons live, so the fight is
    a matter of clearing waves before the next one spawns.

    Args:
      state: The current world.

    Returns:
      vulnerable: Whether the boss is exposed, ``[envs]``.

    """
    return (
        (_on_level(state.melee_mobs.mask, state.player_level).sum(-1) == 0)
        & (_on_level(state.ranged_mobs.mask, state.player_level).sum(-1) == 0)
        & (state.boss_timesteps_to_spawn_this_round <= 0)
    )


def has_beaten_boss(state: EnvState) -> Tensor:
    """Whether the boss has been defeated, which ends the episode in victory."""
    return state.boss_progress >= constants.NUM_LEVELS - 1


def current_map(state: EnvState) -> Tensor:
    """The block grid of the floor the player is on, ``[envs, rows, columns]``."""
    return _on_level(state.map, state.player_level)


def current_items(state: EnvState) -> Tensor:
    """The item grid of the floor the player is on."""
    return _on_level(state.item_map, state.player_level)


def current_mobs(state: EnvState) -> Tensor:
    """The creature-occupancy grid of the floor the player is on."""
    return _on_level(state.mob_map, state.player_level)


def current_light(state: EnvState) -> Tensor:
    """The light grid of the floor the player is on."""
    return _on_level(state.light_map, state.player_level)


def block_at(state: EnvState, position: Tensor) -> Tensor:
    """The block standing at ``position`` on the player's floor, ``[envs]``."""
    return gather_tiles(current_map(state), position)


def in_bounds(position: Tensor) -> Tensor:
    """Whether ``position`` addresses a real tile, ``[envs]``.

    Negative coordinates are out of bounds here even though the indexing
    helpers would wrap them: walking off the top edge must be refused, not
    teleport the player to the bottom.
    """
    rows, columns = constants.MAP_SIZE
    return (
        (position[..., 0] >= 0)
        & (position[..., 0] < rows)
        & (position[..., 1] >= 0)
        & (position[..., 1] < columns)
    )


def is_occupied(state: EnvState, position: Tensor) -> Tensor:
    """Whether a creature or the player already stands at ``position``."""
    return gather_tiles(current_mobs(state), position) | (
        state.player_position == position
    ).all(-1)


def can_walk_on(
    state: EnvState,
    position: Tensor,
    collides_with: Tensor,
) -> Tensor:
    """Whether a creature with the given collisions may enter ``position``.

    Args:
      state: The current world.
      position: Destination tile, ``[envs, 2]``.
      collides_with: Whether ``(ground, water, lava)`` blocks this creature,
        ``[envs, 3]``.

    Returns:
      allowed: Whether the move is legal, ``[envs]``.

    """
    block = block_at(state, position)
    solid = constants.SOLID_BLOCK.to(state.device)[block.long()]
    water = block == int(BlockType.WATER)
    lava = block == int(BlockType.LAVA)
    ground = ~solid & ~water & ~lava
    return (
        in_bounds(position)
        & ~is_occupied(state, position)
        & ~solid
        & ~(collides_with[..., 0] & ground)
        & ~(collides_with[..., 1] & water)
        & ~(collides_with[..., 2] & lava)
    )


def is_near_block(state: EnvState, block: int) -> Tensor:
    """Whether one of the player's eight neighbours is ``block``.

    Crafting needs a table or furnace nearby rather than underfoot, which is
    what this answers.

    Args:
      state: The current world.
      block: The block to look for.

    Returns:
      near: Whether it is adjacent, ``[envs]``.

    """
    neighbours = state.player_position[:, None, :] + constants.CLOSE_BLOCKS.to(
        state.device,
    )
    grid = current_map(state)
    found = torch.zeros(state.num_envs, dtype=torch.bool, device=state.device)
    for offset in range(neighbours.shape[1]):
        tile = neighbours[:, offset]
        found = found | (in_bounds(tile) & (gather_tiles(grid, tile) == block))
    return found


def clip_meters(state: EnvState) -> EnvState:
    """Hold every stock and meter inside its legal range.

    Called once at the end of a step so the many places that add or subtract
    do not each have to know the caps.

    Args:
      state: The world after an action has been applied.

    Returns:
      state: The same world with its meters clipped.

    """
    for name in (
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapling",
        "pickaxe",
        "sword",
        "bow",
        "arrows",
        "torches",
        "ruby",
        "sapphire",
        "books",
        "armour",
        "potions",
    ):
        setattr(
            state.inventory,
            name,
            getattr(state.inventory, name).clamp(max=99),
        )
    state.player_health = state.player_health.clamp(min=0).minimum(
        max_health(state).float(),
    )
    state.player_food = state.player_food.clamp(min=0).minimum(max_food(state))
    state.player_drink = state.player_drink.clamp(min=0).minimum(max_drink(state))
    state.player_energy = state.player_energy.clamp(min=0).minimum(max_energy(state))
    state.player_mana = state.player_mana.clamp(min=0).minimum(max_mana(state))
    return state


def unlock_achievement(state: EnvState, achievement: Tensor, earned: Tensor) -> Tensor:
    """Mark an achievement unlocked where ``earned``, leaving others alone.

    Args:
      state: The current world.
      achievement: Which achievement each environment would unlock, ``[envs]``.
      earned: Whether it was in fact earned, ``[envs]``.

    Returns:
      achievements: The updated unlock table, ``[envs, achievements]``.

    """
    index = achievement.long()
    rows = torch.arange(state.num_envs, device=state.device)
    updated = state.achievements.clone()
    updated[rows, index] = updated[rows, index] | earned
    return updated


def attack_mob_class(
    state: EnvState,
    mobs: Mobs,
    *,
    position: Tensor,
    damage: Tensor,
    mob_class: int,
    can_unlock: Tensor,
) -> tuple[Mobs, Tensor, Tensor, Tensor]:
    """Strike whichever creature of one class stands at ``position``.

    Args:
      state: The current world.
      mobs: The creature class being struck.
      position: The tile being attacked, ``[envs, 2]``.
      damage: The player's damage by element, ``[envs, 3]``.
      mob_class: Which class this is, indexing the defense and achievement
        tables.
      can_unlock: Whether a kill may unlock its achievement, ``[envs]``.

    Returns:
      mobs: The class with damage and deaths applied.
      killed: Whether a creature died, ``[envs]``.
      struck: Whether a creature was there to hit, ``[envs]``.
      achievements: The updated unlock table.

    """
    rows = torch.arange(state.num_envs, device=state.device)
    positions = _on_level(mobs.position, state.player_level)
    alive = _on_level(mobs.mask, state.player_level)
    present = (positions == position[:, None, :]).all(-1) & alive
    struck = present.any(-1)
    target = present.int().argmax(-1)

    species = _on_level(mobs.type_id, state.player_level)[rows, target]
    defense = constants.MOB_DEFENSE.to(state.device)[
        state.player_level.long(),
        mob_class,
    ]
    landed = apply_defense(damage, defense) * struck

    health = mobs.health.clone()
    was_alive = mobs.mask[rows, state.player_level.long(), target]
    health[rows, state.player_level.long(), target] -= landed
    mask = mobs.mask & (health > 0)
    killed = was_alive & ~mask[rows, state.player_level.long(), target]

    achievements = unlock_achievement(
        state,
        constants.MOB_ACHIEVEMENT.to(state.device)[mob_class, species.long()],
        killed & can_unlock,
    )
    return (
        Mobs(
            position=mobs.position,
            health=health,
            mask=mask,
            attack_cooldown=mobs.attack_cooldown,
            type_id=mobs.type_id,
        ),
        killed,
        struck,
        achievements,
    )


def _on_level(field: Tensor, level: Tensor) -> Tensor:
    """Select each environment's current floor from a per-level field."""
    return field[torch.arange(field.shape[0], device=field.device), level.long()]
