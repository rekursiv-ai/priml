"""The late-game actions: potions, spells, enchanting, and levelling up.

These are what the deeper floors are for. Potions are the game's hidden
variable -- which colour heals and which poisons is shuffled each episode, so
the only way to learn the mapping is to drink one and find out. Enchanting
spends gems and mana at a table to add an element to a weapon or armour, and
levelling spends experience on the attributes that raise the meters.
"""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import (
    Achievement,
    Action,
    BlockType,
    ProjectileType,
)
from priml.baselines.craftax.game.indexing import scatter_tiles_where
from priml.baselines.craftax.game.state import EnvState


def drink_potion(state: EnvState, action: Tensor) -> EnvState:
    """Drink a potion, whose effect this episode's mapping decides.

    Six colours map onto six effects -- generous or harmful doses of health,
    mana, and energy -- and the mapping is reshuffled every episode. A colour
    therefore carries no fixed meaning, which is the one thing the agent
    cannot learn from the observation alone.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with the potion drunk and its effect applied.

    """
    colours = (
        Action.DRINK_POTION_RED,
        Action.DRINK_POTION_GREEN,
        Action.DRINK_POTION_BLUE,
        Action.DRINK_POTION_PINK,
        Action.DRINK_POTION_CYAN,
        Action.DRINK_POTION_YELLOW,
    )
    rows = torch.arange(state.num_envs, device=state.device)
    drinking = torch.zeros(state.num_envs, dtype=torch.bool, device=state.device)
    colour = torch.zeros(state.num_envs, dtype=torch.long, device=state.device)
    for index, potion in enumerate(colours):
        chosen = (action == int(potion)) & (state.inventory.potions[:, index] > 0)
        colour = torch.where(chosen, index, colour)
        drinking = drinking | chosen

    effect = state.potion_mapping[rows, colour]
    for target, benefit, harm in (
        ("player_health", 0, 1),
        ("player_mana", 2, 3),
        ("player_energy", 4, 5),
    ):
        delta = 8 * (effect == benefit).int() - 3 * (effect == harm).int()
        current = getattr(state, target)
        setattr(
            state,
            target,
            current
            + torch.where(drinking, delta, torch.zeros_like(delta)).to(
                current.dtype,
            ),
        )

    state.inventory.potions[rows, colour] -= drinking.int()
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.DRINK_POTION),
            device=state.device,
        ),
        drinking,
    )
    return state


def shoot_arrow(state: EnvState, action: Tensor) -> EnvState:
    """Loose an arrow in the faced direction, spending one from the quiver."""
    firing = (
        (action == int(Action.SHOOT_ARROW))
        & (state.inventory.bow >= 1)
        & (state.inventory.arrows >= 1)
    )
    state = _launch(
        state,
        firing=firing,
        kind=torch.full(
            (state.num_envs,),
            int(ProjectileType.ARROW2),
            device=state.device,
        ),
    )
    state.inventory.arrows = state.inventory.arrows - firing.int()
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full((state.num_envs,), int(Achievement.FIRE_BOW), device=state.device),
        firing,
    )
    return state


def cast_spell(state: EnvState, action: Tensor) -> EnvState:
    """Cast a known spell, spending mana.

    A spell must be learned from a book first, which is what makes books
    worth carrying out of a dungeon.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with the spell cast.

    """
    fire = (
        (action == int(Action.CAST_FIREBALL))
        & (state.player_mana >= 2)
        & state.learned_spells[:, 0]
    )
    ice = (
        (action == int(Action.CAST_ICEBALL))
        & (state.player_mana >= 2)
        & state.learned_spells[:, 1]
    )
    casting = fire | ice
    kind = torch.where(
        fire,
        int(ProjectileType.FIREBALL),
        int(ProjectileType.ICEBALL),
    )
    state = _launch(state, firing=casting, kind=kind)
    state.player_mana = state.player_mana - 2 * casting.int()
    for chosen, achievement in (
        (fire, Achievement.CAST_FIREBALL),
        (ice, Achievement.CAST_ICEBALL),
    ):
        state.achievements = mechanics.unlock_achievement(
            state,
            torch.full((state.num_envs,), int(achievement), device=state.device),
            chosen,
        )
    return state


def read_book(
    state: EnvState,
    action: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> EnvState:
    """Learn one unknown spell from a book, spending it.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.
      generator: Source of randomness for which spell is learned.

    Returns:
      state: The world with the spell learned.

    """
    reading = (action == int(Action.READ_BOOK)) & (state.inventory.books > 0)
    unknown = ~state.learned_spells
    # A book teaches something new where it can; with both known it is spent
    # on the first slot, as the reference does.
    weights = torch.where(
        unknown.any(-1, keepdim=True),
        unknown.float(),
        torch.ones_like(unknown, dtype=torch.float32),
    )
    spell = torch.multinomial(weights, 1, generator=generator).squeeze(-1)

    rows = torch.arange(state.num_envs, device=state.device)
    state.learned_spells[rows, spell] = state.learned_spells[rows, spell] | reading
    state.inventory.books = state.inventory.books - reading.int()
    for index, achievement in (
        (0, Achievement.LEARN_FIREBALL),
        (1, Achievement.LEARN_ICEBALL),
    ):
        state.achievements = mechanics.unlock_achievement(
            state,
            torch.full((state.num_envs,), int(achievement), device=state.device),
            reading & (spell == index),
        )
    return state


def enchant(
    state: EnvState,
    action: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> EnvState:
    """Bind an element to a weapon or armour at an enchantment table.

    The table's element decides which gem is spent: fire tables burn rubies,
    ice tables sapphires. Both cost most of a full mana bar, so enchanting is
    something done between fights rather than during one.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.
      generator: Source of randomness for which armour piece is chosen.

    Returns:
      state: The world with the enchantment applied.

    """
    target = (
        state.player_position
        + constants.DIRECTIONS.to(state.device)[state.player_direction.long()]
    )
    block = mechanics.block_at(state, target)
    fire_table = block == int(BlockType.ENCHANTMENT_TABLE_FIRE)
    at_table = fire_table | (block == int(BlockType.ENCHANTMENT_TABLE_ICE))
    element = torch.where(fire_table, 1, 2)
    gems = torch.where(fire_table, state.inventory.ruby, state.inventory.sapphire)

    ready = (state.player_mana >= 9) & at_table & (gems >= 1)
    on_sword = (
        ready & (action == int(Action.ENCHANT_SWORD)) & (state.inventory.sword > 0)
    )
    on_bow = ready & (action == int(Action.ENCHANT_BOW)) & (state.inventory.bow > 0)
    on_armour = (
        ready
        & (action == int(Action.ENCHANT_ARMOUR))
        & (state.inventory.armour.sum(-1) > 0)
    )
    enchanting = on_sword | on_bow | on_armour

    state.sword_enchantment = torch.where(
        on_sword,
        element.int(),
        state.sword_enchantment,
    )
    state.bow_enchantment = torch.where(on_bow, element.int(), state.bow_enchantment)

    # Prefer a bare piece; failing that, overwrite one carrying the other
    # element, so a second enchantment is never wasted.
    bare = state.armour_enchantments == 0
    opposite = (state.armour_enchantments != 0) & (
        state.armour_enchantments != element[:, None]
    )
    candidates = torch.where(bare.any(-1, keepdim=True), bare, opposite).float()
    candidates = torch.where(
        candidates.sum(-1, keepdim=True) > 0,
        candidates,
        torch.ones_like(candidates),
    )
    piece = torch.multinomial(candidates, 1, generator=generator).squeeze(-1)
    rows = torch.arange(state.num_envs, device=state.device)
    state.armour_enchantments[rows, piece] = torch.where(
        on_armour,
        element.int(),
        state.armour_enchantments[rows, piece],
    )

    state.inventory.ruby = state.inventory.ruby - (enchanting & (element == 1)).int()
    state.inventory.sapphire = (
        state.inventory.sapphire - (enchanting & (element == 2)).int()
    )
    state.player_mana = state.player_mana - 9 * enchanting.int()
    for chosen, achievement in (
        (on_sword, Achievement.ENCHANT_SWORD),
        (on_armour, Achievement.ENCHANT_ARMOUR),
    ):
        state.achievements = mechanics.unlock_achievement(
            state,
            torch.full((state.num_envs,), int(achievement), device=state.device),
            chosen,
        )
    return state


def level_up(state: EnvState, action: Tensor) -> EnvState:
    """Spend a point of experience to raise one attribute.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with the attribute raised.

    """
    for chosen_action, attribute in (
        (Action.LEVEL_UP_DEXTERITY, "player_dexterity"),
        (Action.LEVEL_UP_STRENGTH, "player_strength"),
        (Action.LEVEL_UP_INTELLIGENCE, "player_intelligence"),
    ):
        current = getattr(state, attribute)
        raising = (
            (action == int(chosen_action))
            & (state.player_xp >= 1)
            & (current < constants.MAX_ATTRIBUTE)
        )
        setattr(state, attribute, current + raising.int())
        state.player_xp = state.player_xp - raising.int()
    return state


def grow_plants(state: EnvState) -> EnvState:
    """Age every sown plant and ripen the ones that are ready.

    Args:
      state: The current world.

    Returns:
      state: The world with plants aged and any ripened.

    """
    state.growing_plants_age = (
        state.growing_plants_age + state.growing_plants_mask.int()
    )
    ripe = state.growing_plants_mask & (state.growing_plants_age > 600)

    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    grid = state.map[rows, level]
    for slot in range(state.growing_plants_mask.shape[-1]):
        grid = scatter_tiles_where(
            grid,
            state.growing_plants_positions[:, slot],
            torch.full(
                (state.num_envs,),
                int(BlockType.RIPE_PLANT),
                device=state.device,
            ),
            ripe[:, slot],
        )
    state.map[rows, level] = grid
    return state


def _launch(state: EnvState, *, firing: Tensor, kind: Tensor) -> EnvState:
    """Put one of the player's projectiles into a free slot."""
    projectiles = state.player_projectiles
    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    free = projectiles.mask[rows, level]
    slot = (~free).int().argmax(-1)
    firing = firing & (~free).any(-1)

    heading = constants.DIRECTIONS.to(state.device)[state.player_direction.long()]
    projectiles.position[rows, level, slot] = torch.where(
        firing[:, None],
        (state.player_position + heading).int(),
        projectiles.position[rows, level, slot],
    )
    projectiles.mask[rows, level, slot] = projectiles.mask[rows, level, slot] | firing
    projectiles.type_id[rows, level, slot] = torch.where(
        firing,
        kind.int(),
        projectiles.type_id[rows, level, slot],
    )
    state.player_projectile_directions[rows, level, slot] = torch.where(
        firing[:, None],
        heading.int(),
        state.player_projectile_directions[rows, level, slot],
    )
    return state
