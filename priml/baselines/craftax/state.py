"""The mutable world state, batched across parallel environments.

Every field carries a leading environment axis, so one state object IS the
whole batch and a step is one tensor program rather than a loop over games.
The shapes in each docstring omit that axis: ``[levels, rows, columns]`` means
``[envs, levels, rows, columns]`` on the real tensor.

The state is a dataclass rather than a dict so a typo is an error at the point
it is written, and it is mutable because a step rewrites most of it -- copying
the whole world to change one tile would dominate the step cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Self

from torch import Tensor

import torch

from priml.baselines.craftax import constants


@dataclass(slots=True, kw_only=True)
class Mobs:
    """One class of creature, stored as fixed-width slots per level.

    A slot is live only where ``mask`` is set. Fixed width is what keeps the
    state rectangular: spawning fills a free slot rather than growing an array,
    so the shape never depends on what happened in the episode.

    Attributes:
      position: Tile coordinates, ``[levels, slots, 2]``.
      health: Remaining health, ``[levels, slots]``.
      mask: Whether each slot holds a live creature, ``[levels, slots]``.
      attack_cooldown: Steps until the creature may attack again.
      type_id: Which species occupies the slot.

    """

    position: Tensor
    health: Tensor
    mask: Tensor
    attack_cooldown: Tensor
    type_id: Tensor

    @classmethod
    def empty(
        cls,
        *,
        num_envs: int,
        num_levels: int,
        num_slots: int,
        device: torch.device,
    ) -> Self:
        """Return a batch with every slot free.

        Args:
          num_envs: Parallel environments.
          num_levels: Floors of the world.
          num_slots: Creatures of this class each floor may hold at once.
          device: Device the tensors are allocated on.

        Returns:
          mobs: A batch of empty creature slots.

        """
        shape = (num_envs, num_levels, num_slots)
        return cls(
            position=torch.zeros((*shape, 2), dtype=torch.int32, device=device),
            health=torch.ones(shape, dtype=torch.float32, device=device),
            mask=torch.zeros(shape, dtype=torch.bool, device=device),
            attack_cooldown=torch.zeros(shape, dtype=torch.int32, device=device),
            type_id=torch.zeros(shape, dtype=torch.int32, device=device),
        )


@dataclass(slots=True, kw_only=True)
class Inventory:
    """What the player is carrying.

    Every field is ``[envs]`` except ``armour`` and ``potions``, which are
    ``[envs, 4]`` and ``[envs, 6]`` -- one entry per body slot and per potion
    colour respectively.
    """

    wood: Tensor
    stone: Tensor
    coal: Tensor
    iron: Tensor
    diamond: Tensor
    sapling: Tensor
    pickaxe: Tensor
    sword: Tensor
    bow: Tensor
    arrows: Tensor
    armour: Tensor
    torches: Tensor
    ruby: Tensor
    sapphire: Tensor
    potions: Tensor
    books: Tensor

    @classmethod
    def empty(cls, *, num_envs: int, device: torch.device) -> Self:
        """Return an inventory holding nothing.

        Args:
          num_envs: Parallel environments.
          device: Device the tensors are allocated on.

        Returns:
          inventory: An empty inventory batch.

        """
        counts = {
            name: torch.zeros(num_envs, dtype=torch.int32, device=device)
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
            )
        }
        return cls(
            armour=torch.zeros((num_envs, 4), dtype=torch.int32, device=device),
            potions=torch.zeros((num_envs, 6), dtype=torch.int32, device=device),
            **counts,
        )


@dataclass(slots=True, kw_only=True)
class EnvState:
    """The complete world, batched across environments.

    Attributes:
      map: Block at each tile, ``[levels, rows, columns]``.
      item_map: Item lying on each tile, same shape.
      light_map: How lit each tile is, same shape.
      mob_map: Whether a creature occupies each tile, same shape.
      down_ladders: Row/column of each floor's descent, ``[levels, 2]``.
      up_ladders: Row/column of each floor's ascent, ``[levels, 2]``.
      chests_opened: Whether this floor's chest achievement fired, ``[levels]``.
      monsters_killed: Kills on each floor, ``[levels]``.
      player_position: Tile the player stands on, ``[2]``.
      player_level: Which floor the player is on.
      player_direction: Which way the player faces, as an ``Action`` value.
      player_health: Remaining health; zero ends the episode.
      player_food: Food meter.
      player_drink: Water meter.
      player_energy: Energy meter, restored by sleeping.
      player_mana: Mana meter, spent on spells.
      is_sleeping: Whether the player is asleep.
      is_resting: Whether the player is resting.
      player_recover: Fractional progress toward the next health tick.
      player_hunger: Fractional progress toward the next food loss.
      player_thirst: Fractional progress toward the next drink loss.
      player_fatigue: Fractional progress toward the next energy loss.
      player_recover_mana: Fractional progress toward the next mana tick.
      player_xp: Unspent experience.
      player_dexterity: Dexterity attribute.
      player_strength: Strength attribute.
      player_intelligence: Intelligence attribute.
      inventory: What the player carries.
      melee_mobs: Creatures that attack in contact.
      passive_mobs: Creatures that flee and can be eaten.
      ranged_mobs: Creatures that shoot.
      mob_projectiles: Projectiles fired at the player.
      mob_projectile_directions: Their travel directions, ``[levels, slots, 2]``.
      player_projectiles: Projectiles the player fired.
      player_projectile_directions: Their travel directions.
      growing_plants_positions: Where sown plants sit, ``[plants, 2]``.
      growing_plants_age: How long each has grown, ``[plants]``.
      growing_plants_mask: Which plant slots are in use, ``[plants]``.
      potion_mapping: Which effect each potion colour has this episode,
        ``[6]``. Randomized per episode, which is what makes the game
        partially observable.
      learned_spells: Whether fireball and iceball are known, ``[2]``.
      sword_enchantment: Element the sword is enchanted with.
      bow_enchantment: Element the bow is enchanted with.
      armour_enchantments: Element each armour piece carries, ``[4]``.
      boss_progress: How far the final fight has advanced.
      boss_timesteps_to_spawn_this_round: Countdown to the next boss wave.
      light_level: Daylight outside, on ``[0, 1]``.
      achievements: Which achievements have fired, ``[achievements]``.
      timestep: Steps taken this episode.

    """

    map: Tensor
    item_map: Tensor
    light_map: Tensor
    mob_map: Tensor
    down_ladders: Tensor
    up_ladders: Tensor
    chests_opened: Tensor
    monsters_killed: Tensor

    player_position: Tensor
    player_level: Tensor
    player_direction: Tensor

    player_health: Tensor
    player_food: Tensor
    player_drink: Tensor
    player_energy: Tensor
    player_mana: Tensor
    is_sleeping: Tensor
    is_resting: Tensor

    player_recover: Tensor
    player_hunger: Tensor
    player_thirst: Tensor
    player_fatigue: Tensor
    player_recover_mana: Tensor

    player_xp: Tensor
    player_dexterity: Tensor
    player_strength: Tensor
    player_intelligence: Tensor

    inventory: Inventory

    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs
    mob_projectiles: Mobs
    mob_projectile_directions: Tensor
    player_projectiles: Mobs
    player_projectile_directions: Tensor

    growing_plants_positions: Tensor
    growing_plants_age: Tensor
    growing_plants_mask: Tensor

    potion_mapping: Tensor
    learned_spells: Tensor

    sword_enchantment: Tensor
    bow_enchantment: Tensor
    armour_enchantments: Tensor

    boss_progress: Tensor
    boss_timesteps_to_spawn_this_round: Tensor

    light_level: Tensor
    achievements: Tensor
    timestep: Tensor

    @property
    def num_envs(self) -> int:
        """Environments this state carries."""
        return self.timestep.shape[0]

    @property
    def device(self) -> torch.device:
        """Device every tensor in this state lives on."""
        return self.timestep.device

    def select(self, keep: Tensor, other: Self) -> Self:
        """Take rows from ``other`` where ``keep`` is set, else from ``self``.

        This is how a terminated worker is swapped for a freshly generated one
        without disturbing the workers still mid-episode: both states are full
        batches, and the choice is per row.

        Args:
          keep: Which environments take their row from ``other``, ``[envs]``.
          other: A state batch of the same shape to draw those rows from.

        Returns:
          state: A new state, row-selected between the two.

        """

        def choose(mine: Tensor, theirs: Tensor) -> Tensor:
            return torch.where(
                keep.reshape(keep.shape[0], *(1,) * (mine.dim() - 1)),
                theirs,
                mine,
            )

        return _map_state(choose, self, other)

    def state_dict(self) -> dict[str, Any]:
        """Return every tensor by dotted name, for checkpointing."""
        flat: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                flat[field.name] = value
                continue
            for sub in fields(value):
                flat[f"{field.name}.{sub.name}"] = getattr(value, sub.name)
        return flat

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore tensors saved by :meth:`state_dict`, in place."""
        for name, value in state_dict.items():
            head, _, tail = name.partition(".")
            target = self if not tail else getattr(self, head)
            setattr(target, tail or head, value)


def empty_state(*, num_envs: int, device: torch.device) -> EnvState:
    """Allocate a state batch with every field at its zero value.

    The world it describes is unplayable -- no terrain, no player position --
    so it exists to be filled by world generation, not to be stepped.

    Args:
      num_envs: Parallel environments.
      device: Device the tensors are allocated on.

    Returns:
      state: A zeroed state batch.

    """
    levels, (rows, columns) = constants.NUM_LEVELS, constants.MAP_SIZE
    grid = (num_envs, levels, rows, columns)
    per_level = (num_envs, levels)
    scalar_int = torch.zeros(num_envs, dtype=torch.int32, device=device)
    scalar_float = torch.zeros(num_envs, dtype=torch.float32, device=device)
    return EnvState(
        map=torch.zeros(grid, dtype=torch.int32, device=device),
        item_map=torch.zeros(grid, dtype=torch.int32, device=device),
        light_map=torch.zeros(grid, dtype=torch.float32, device=device),
        mob_map=torch.zeros(grid, dtype=torch.bool, device=device),
        down_ladders=torch.zeros((*per_level, 2), dtype=torch.int32, device=device),
        up_ladders=torch.zeros((*per_level, 2), dtype=torch.int32, device=device),
        chests_opened=torch.zeros(per_level, dtype=torch.bool, device=device),
        monsters_killed=torch.zeros(per_level, dtype=torch.int32, device=device),
        player_position=torch.zeros((num_envs, 2), dtype=torch.int32, device=device),
        player_level=scalar_int.clone(),
        player_direction=scalar_int.clone(),
        player_health=scalar_float.clone(),
        player_food=scalar_int.clone(),
        player_drink=scalar_int.clone(),
        player_energy=scalar_int.clone(),
        player_mana=scalar_int.clone(),
        is_sleeping=torch.zeros(num_envs, dtype=torch.bool, device=device),
        is_resting=torch.zeros(num_envs, dtype=torch.bool, device=device),
        player_recover=scalar_float.clone(),
        player_hunger=scalar_float.clone(),
        player_thirst=scalar_float.clone(),
        player_fatigue=scalar_float.clone(),
        player_recover_mana=scalar_float.clone(),
        player_xp=scalar_int.clone(),
        player_dexterity=scalar_int.clone(),
        player_strength=scalar_int.clone(),
        player_intelligence=scalar_int.clone(),
        inventory=Inventory.empty(num_envs=num_envs, device=device),
        melee_mobs=Mobs.empty(
            num_envs=num_envs,
            num_levels=levels,
            num_slots=constants.MAX_MELEE_MOBS,
            device=device,
        ),
        passive_mobs=Mobs.empty(
            num_envs=num_envs,
            num_levels=levels,
            num_slots=constants.MAX_PASSIVE_MOBS,
            device=device,
        ),
        ranged_mobs=Mobs.empty(
            num_envs=num_envs,
            num_levels=levels,
            num_slots=constants.MAX_RANGED_MOBS,
            device=device,
        ),
        mob_projectiles=Mobs.empty(
            num_envs=num_envs,
            num_levels=levels,
            num_slots=constants.MAX_MOB_PROJECTILES,
            device=device,
        ),
        mob_projectile_directions=torch.zeros(
            (*per_level, constants.MAX_MOB_PROJECTILES, 2),
            dtype=torch.int32,
            device=device,
        ),
        player_projectiles=Mobs.empty(
            num_envs=num_envs,
            num_levels=levels,
            num_slots=constants.MAX_PLAYER_PROJECTILES,
            device=device,
        ),
        player_projectile_directions=torch.zeros(
            (*per_level, constants.MAX_PLAYER_PROJECTILES, 2),
            dtype=torch.int32,
            device=device,
        ),
        growing_plants_positions=torch.zeros(
            (num_envs, constants.MAX_GROWING_PLANTS, 2),
            dtype=torch.int32,
            device=device,
        ),
        growing_plants_age=torch.zeros(
            (num_envs, constants.MAX_GROWING_PLANTS),
            dtype=torch.int32,
            device=device,
        ),
        growing_plants_mask=torch.zeros(
            (num_envs, constants.MAX_GROWING_PLANTS),
            dtype=torch.bool,
            device=device,
        ),
        potion_mapping=torch.zeros((num_envs, 6), dtype=torch.int32, device=device),
        learned_spells=torch.zeros((num_envs, 2), dtype=torch.bool, device=device),
        sword_enchantment=scalar_int.clone(),
        bow_enchantment=scalar_int.clone(),
        armour_enchantments=torch.zeros(
            (num_envs, 4),
            dtype=torch.int32,
            device=device,
        ),
        boss_progress=scalar_int.clone(),
        boss_timesteps_to_spawn_this_round=scalar_int.clone(),
        light_level=scalar_float.clone(),
        achievements=torch.zeros(
            (num_envs, len(constants.Achievement)),
            dtype=torch.bool,
            device=device,
        ),
        timestep=scalar_int.clone(),
    )


def _map_state[StateT: EnvState](
    combine: Callable[[Tensor, Tensor], Tensor],
    left: StateT,
    right: StateT,
) -> StateT:
    """Apply ``combine`` to every matching tensor pair of two states."""
    merged: dict[str, Any] = {}
    for field in fields(left):
        mine = getattr(left, field.name)
        theirs = getattr(right, field.name)
        if isinstance(mine, Tensor):
            merged[field.name] = combine(mine, theirs)
            continue
        merged[field.name] = type(mine)(
            **{
                sub.name: combine(getattr(mine, sub.name), getattr(theirs, sub.name))
                for sub in fields(mine)
            },
        )
    return type(left)(**merged)
