"""One game step: the order in which everything the world does happens.

The order is itself a rule. The player acts first and the world reacts, so a
creature killed this step cannot also strike back, and a block placed this step
already blocks the creature that moves onto it. Survival is ticked last, so a
player who ate this step is not also starved by it.

A sleeping or resting player has their action replaced with doing nothing: they
must be woken -- by choice or by a blow -- before they can act again, which is
what makes sleeping a real commitment.
"""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.craftax import (
    abilities,
    constants,
    crafting,
    interact,
    mechanics,
    mobs,
    survival,
    world_gen,
)
from priml.baselines.craftax.constants import Achievement, Action, ItemType
from priml.baselines.craftax.indexing import gather_tiles
from priml.baselines.craftax.state import EnvState


def step(
    state: EnvState,
    action: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[EnvState, Tensor]:
    """Advance the world one step and return the reward it earned.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.
      generator: Source of randomness for every draw this step.

    Returns:
      state: The world after the step.
      reward: Reward earned, ``[envs]``.

    """
    unlocked_before = state.achievements.clone()
    health_before = state.player_health.clone()

    # A sleeping or resting player cannot act until something wakes them.
    action = torch.where(
        state.is_sleeping | state.is_resting,
        torch.full_like(action, int(Action.NOOP)),
        action,
    )

    state = change_floor(state, action)
    state = crafting.craft(state, action)
    state = interact.interact(
        state,
        doing=action == int(Action.DO),
        generator=generator,
    )
    state = crafting.place(state, action)
    state = abilities.shoot_arrow(state, action)
    state = abilities.cast_spell(state, action)
    state = abilities.drink_potion(state, action)
    state = abilities.read_book(state, action, generator=generator)
    state = abilities.enchant(state, action, generator=generator)
    state = _advance_boss(state)
    state = abilities.level_up(state, action)
    state = survival.move_player(state, action)

    state = mobs.update_mobs(state, generator=generator)
    state = mobs.spawn_mobs(state, generator=generator)
    state = abilities.grow_plants(state)

    state = survival.update_intrinsics(state, action)
    state = mechanics.clip_meters(state)
    state = _unlock_from_inventory(state)

    reward = _reward(
        state,
        unlocked_before=unlocked_before,
        health_before=health_before,
    )
    state.timestep = state.timestep + 1
    state.light_level = world_gen.daylight(state.timestep)
    return state, reward


def is_done(state: EnvState) -> Tensor:
    """Whether each episode has ended, ``[envs]``.

    An episode ends when the player dies, when the boss falls, or when the
    step limit is reached.

    Args:
      state: The current world.

    Returns:
      done: Whether the episode is over, ``[envs]``.

    """
    return (
        (state.player_health <= 0)
        | mechanics.has_beaten_boss(state)
        | (state.timestep >= constants.MAX_TIMESTEPS)
    )


def change_floor(state: EnvState, action: Tensor) -> EnvState:
    """Take a ladder up or down, if the player is standing on one.

    Descending is gated on having cleared the floor, which is what turns each
    level into an objective rather than a corridor. Arriving somewhere new
    grants a point of experience.

    Args:
      state: The current world.
      action: The chosen action per environment, ``[envs]``.

    Returns:
      state: The world with the player moved between floors.

    """
    rows = torch.arange(state.num_envs, device=state.device)
    standing_on = gather_tiles(
        mechanics.current_items(state),
        state.player_position,
    )
    cleared = (
        state.monsters_killed[rows, state.player_level.long()]
        >= constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    descending = (
        (action == int(Action.DESCEND))
        & (standing_on == int(ItemType.LADDER_DOWN))
        & cleared
        & (state.player_level < constants.NUM_LEVELS - 1)
    )
    ascending = (
        (action == int(Action.ASCEND))
        & (standing_on == int(ItemType.LADDER_UP))
        & (state.player_level > 0)
    )

    delta = descending.int() - ascending.int()
    arrival = state.player_level.long() + delta
    below = state.up_ladders[rows, arrival.clamp(max=constants.NUM_LEVELS - 1)]
    above = state.down_ladders[rows, arrival.clamp(min=0)]
    state.player_position = torch.where(
        descending[:, None],
        below,
        torch.where(ascending[:, None], above, state.player_position),
    )
    state.player_level = state.player_level + delta

    achievement = constants.LEVEL_ACHIEVEMENT.to(state.device)[arrival]
    # Only the first arrival on a floor pays: the experience point is for
    # getting there, not for using the ladder.
    first_visit = (arrival != 0) & ~state.achievements[rows, achievement.long()]
    state.achievements = mechanics.unlock_achievement(state, achievement, arrival != 0)
    state.player_xp = state.player_xp + first_visit.int()
    return state


def _advance_boss(state: EnvState) -> EnvState:
    """Count down to the next summoning wave and record a victory."""
    state.boss_timesteps_to_spawn_this_round = (
        state.boss_timesteps_to_spawn_this_round
        - mechanics.is_fighting_boss(state).int()
    )
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full(
            (state.num_envs,),
            int(Achievement.DEFEAT_NECROMANCER),
            device=state.device,
        ),
        mechanics.has_beaten_boss(state),
    )
    return state


def _unlock_from_inventory(state: EnvState) -> EnvState:
    """Grant the achievements that merely require holding something.

    A diamond pickaxe can be crafted or looted from a chest, and both should
    count. Checking the inventory once at the end of the step covers every
    route without each route having to remember.
    """
    thresholds = (
        ("wood", 1, Achievement.COLLECT_WOOD),
        ("stone", 1, Achievement.COLLECT_STONE),
        ("coal", 1, Achievement.COLLECT_COAL),
        ("iron", 1, Achievement.COLLECT_IRON),
        ("diamond", 1, Achievement.COLLECT_DIAMOND),
        ("ruby", 1, Achievement.COLLECT_RUBY),
        ("sapphire", 1, Achievement.COLLECT_SAPPHIRE),
        ("sapling", 1, Achievement.COLLECT_SAPLING),
        ("bow", 1, Achievement.FIND_BOW),
        ("arrows", 1, Achievement.MAKE_ARROW),
        ("torches", 1, Achievement.MAKE_TORCH),
        ("pickaxe", 1, Achievement.MAKE_WOOD_PICKAXE),
        ("pickaxe", 2, Achievement.MAKE_STONE_PICKAXE),
        ("pickaxe", 3, Achievement.MAKE_IRON_PICKAXE),
        ("pickaxe", 4, Achievement.MAKE_DIAMOND_PICKAXE),
        ("sword", 1, Achievement.MAKE_WOOD_SWORD),
        ("sword", 2, Achievement.MAKE_STONE_SWORD),
        ("sword", 3, Achievement.MAKE_IRON_SWORD),
        ("sword", 4, Achievement.MAKE_DIAMOND_SWORD),
    )
    for field, amount, achievement in thresholds:
        state.achievements = mechanics.unlock_achievement(
            state,
            torch.full((state.num_envs,), int(achievement), device=state.device),
            getattr(state.inventory, field) >= amount,
        )
    return state


def _reward(
    state: EnvState,
    *,
    unlocked_before: Tensor,
    health_before: Tensor,
) -> Tensor:
    """Score the step: what was achieved, plus a tenth of health gained.

    Achievements are one-time, so the reward is the difference in the unlock
    table rather than its total. The health term is small and signed, which
    nudges toward staying alive without paying for it directly.
    """
    newly = (state.achievements.int() - unlocked_before.int()).float()
    earned = (newly * constants.ACHIEVEMENT_REWARD.to(state.device)).sum(-1)
    return earned + (state.player_health - health_before) * 0.1
