"""Creature behaviour: hunting, fleeing, shooting, and spawning.

Each creature class acts once per step, one slot at a time, because a slot's
decision depends on where the earlier ones just moved -- two creatures must not
step onto the same tile. The slot loop is therefore sequential, but every
environment inside it is handled at once.

Melee creatures hunt: within ten tiles they mostly step toward the player, and
otherwise wander, which is what makes a distant one look aimless and a close
one dangerous. Passive creatures only wander. Ranged creatures keep their
distance and fire.
"""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.craftax import constants, mechanics
from priml.baselines.craftax.constants import Achievement, BlockType
from priml.baselines.craftax.indexing import (
    scatter_tiles_where,
)
from priml.baselines.craftax.state import EnvState, Mobs


def update_mobs(
    state: EnvState,
    *,
    generator: torch.Generator | None = None,
) -> EnvState:
    """Advance every creature and projectile by one step.

    Args:
      state: The current world.
      generator: Source of randomness for movement and attacks.

    Returns:
      state: The world with creatures moved and attacks resolved.

    """
    state = _update_melee(state, generator=generator)
    state = _update_passive(state, generator=generator)
    state = _update_ranged(state, generator=generator)
    return _update_projectiles(state)


def spawn_mobs(
    state: EnvState,
    *,
    generator: torch.Generator | None = None,
) -> EnvState:
    """Populate the player's floor, at a rate the floor and night set.

    An uncleared floor spawns three times as fast, which is what makes
    clearing one a real objective rather than a formality. Creatures appear
    outside the player's immediate surroundings but within sight, so they
    arrive rather than materialize on top of them.

    Args:
      state: The current world.
      generator: Source of randomness for the spawn draws.

    Returns:
      state: The world with any new creatures placed.

    """
    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    uncleared = (
        state.monsters_killed[rows, level] < constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    rate = 1 + 2 * uncleared.int()

    grid = mechanics.current_map(state)
    distance = _distance_to_player(state)
    # Far enough that they do not appear underfoot, near enough to matter.
    room = (
        (distance > 3)
        & (distance < constants.MOB_DESPAWN_DISTANCE)
        & ~mechanics.current_mobs(state)
    )
    walkable = (
        (grid == int(BlockType.GRASS))
        | (grid == int(BlockType.PATH))
        | (grid == int(BlockType.FIRE_GRASS))
        | (grid == int(BlockType.ICE_GRASS))
    )
    room = room & walkable

    chances = constants.FLOOR_MOB_SPAWN_CHANCE.to(state.device)[level]
    species = constants.FLOOR_MOB_TYPE.to(state.device)[level]
    for field, column, mob_class in (
        ("passive_mobs", 0, 0),
        ("melee_mobs", 1, 1),
        ("ranged_mobs", 2, 2),
    ):
        mobs: Mobs = getattr(state, field)
        alive = _on_level(mobs.mask, state.player_level)
        # Night is when the surface becomes dangerous: the fourth column is
        # the extra melee chance, weighted by how dark it is.
        chance = chances[:, column]
        if column == 1:
            chance = chance + chances[:, 3] * (1.0 - state.light_level) ** 2
        spawning = (
            (alive.sum(-1) < alive.shape[-1])
            & (
                torch.rand(state.num_envs, generator=generator, device=state.device)
                < chance * rate
            )
            & room.flatten(1).any(-1)
        )
        if column == 0:
            # The boss floor spawns no cattle; it is not a place to graze.
            spawning = spawning & ~mechanics.is_fighting_boss(state)

        place = _sample_position(room, generator=generator)
        slot = (~alive).int().argmax(-1)
        health = constants.MOB_HEALTH.to(state.device)[level, mob_class]
        state = _place_mob(
            state,
            field=field,
            slot=slot,
            position=place,
            species=species[:, column],
            health=health,
            spawning=spawning,
        )
    return state


def _update_melee(
    state: EnvState,
    *,
    generator: torch.Generator | None,
) -> EnvState:
    """Step every hunting creature: approach, strike, or wander."""
    mobs = state.melee_mobs
    for slot in range(mobs.mask.shape[-1]):
        alive = _slot(mobs.mask, state, slot)
        position = _slot(mobs.position, state, slot)
        toward = _step_toward_player(state, position, generator=generator)
        wander = _random_step(state.num_envs, state.device, generator, moves=4)

        gap = (position - state.player_position).abs().sum(-1)
        hunting = ((gap < 10) | mechanics.is_fighting_boss(state)) & (
            torch.rand(state.num_envs, generator=generator, device=state.device) < 0.75
        )
        proposed = position + torch.where(hunting[:, None], toward, wander)

        striking = (gap == 1) & (_slot(mobs.attack_cooldown, state, slot) <= 0) & alive
        proposed = torch.where(striking[:, None], position, proposed)
        state = _strike_player(
            state,
            species=_slot(mobs.type_id, state, slot),
            mob_class=1,
            striking=striking,
        )

        collides = constants.MOB_COLLIDES_WITH.to(state.device)[
            state.player_level.long(),
            1,
        ]
        moved = torch.where(
            mechanics.can_walk_on(state, proposed, collides)[:, None],
            proposed,
            position,
        )
        cooldown = torch.where(
            striking,
            torch.full_like(_slot(mobs.attack_cooldown, state, slot), 5),
            _slot(mobs.attack_cooldown, state, slot) - 1,
        )
        state = _relocate(
            state,
            field="melee_mobs",
            slot=slot,
            old=position,
            new=moved,
            cooldown=cooldown,
            despawns=~mechanics.is_fighting_boss(state),
        )
        mobs = state.melee_mobs
    return state


def _update_passive(
    state: EnvState,
    *,
    generator: torch.Generator | None,
) -> EnvState:
    """Step every grazing creature, which only ever wanders."""
    mobs = state.passive_mobs
    for slot in range(mobs.mask.shape[-1]):
        position = _slot(mobs.position, state, slot)
        # Eight directions rather than four, so a cow stands still half the
        # time and drifts rather than pacing.
        proposed = position + _random_step(
            state.num_envs,
            state.device,
            generator,
            moves=8,
        )
        collides = constants.MOB_COLLIDES_WITH.to(state.device)[
            state.player_level.long(),
            0,
        ]
        moved = torch.where(
            mechanics.can_walk_on(state, proposed, collides)[:, None],
            proposed,
            position,
        )
        state = _relocate(
            state,
            field="passive_mobs",
            slot=slot,
            old=position,
            new=moved,
            cooldown=_slot(mobs.attack_cooldown, state, slot),
            despawns=torch.ones(state.num_envs, dtype=torch.bool, device=state.device),
        )
        mobs = state.passive_mobs
    return state


def _update_ranged(
    state: EnvState,
    *,
    generator: torch.Generator | None,
) -> EnvState:
    """Step every shooting creature: keep distance and fire down a line."""
    mobs = state.ranged_mobs
    for slot in range(mobs.mask.shape[-1]):
        alive = _slot(mobs.mask, state, slot)
        position = _slot(mobs.position, state, slot)
        offset = state.player_position - position
        gap = offset.abs().sum(-1)

        # Archers back away when the player closes, which is what makes them
        # awkward to fight without a bow of your own.
        toward = _step_toward_player(state, position, generator=generator)
        proposed = position + torch.where((gap < 5)[:, None], -toward, toward)
        wander = position + _random_step(
            state.num_envs,
            state.device,
            generator,
            moves=4,
        )
        proposed = torch.where((gap < 10)[:, None], proposed, wander)

        aligned = ((offset[:, 0] == 0) | (offset[:, 1] == 0)) & (gap < 10)
        firing = aligned & alive & (_slot(mobs.attack_cooldown, state, slot) <= 0)
        state = _fire_projectile(
            state,
            source=position,
            toward=offset,
            species=_slot(mobs.type_id, state, slot),
            firing=firing,
        )

        collides = constants.MOB_COLLIDES_WITH.to(state.device)[
            state.player_level.long(),
            2,
        ]
        moved = torch.where(
            mechanics.can_walk_on(state, proposed, collides)[:, None],
            proposed,
            position,
        )
        cooldown = torch.where(
            firing,
            torch.full_like(_slot(mobs.attack_cooldown, state, slot), 4),
            _slot(mobs.attack_cooldown, state, slot) - 1,
        )
        state = _relocate(
            state,
            field="ranged_mobs",
            slot=slot,
            old=position,
            new=moved,
            cooldown=cooldown,
            despawns=~mechanics.is_fighting_boss(state),
        )
        mobs = state.ranged_mobs
    return state


def _update_projectiles(state: EnvState) -> EnvState:
    """Fly every projectile one tile and resolve what it hits."""
    for field, directions_field, hurts_player in (
        ("mob_projectiles", "mob_projectile_directions", True),
        ("player_projectiles", "player_projectile_directions", False),
    ):
        mobs: Mobs = getattr(state, field)
        directions = getattr(state, directions_field)
        for slot in range(mobs.mask.shape[-1]):
            alive = _slot(mobs.mask, state, slot)
            position = _slot(mobs.position, state, slot)
            heading = _slot(directions, state, slot)
            flown = position + heading

            hits_player = alive & (flown == state.player_position).all(-1)
            if hurts_player:
                damage = constants.MOB_DAMAGE.to(state.device)[
                    _slot(mobs.type_id, state, slot).long(),
                    3,
                ]
                state.player_health = state.player_health - torch.where(
                    hits_player,
                    mechanics.damage_to_player(state, damage),
                    torch.zeros_like(state.player_health),
                )
                state.is_sleeping = state.is_sleeping & ~hits_player

            # A projectile stops at the first solid thing it meets.
            blocked = constants.SOLID_BLOCK.to(state.device)[
                mechanics.block_at(state, flown).long()
            ]
            survives = alive & ~hits_player & ~blocked & mechanics.in_bounds(flown)

            rows = torch.arange(state.num_envs, device=state.device)
            level = state.player_level.long()
            mobs.position[rows, level, slot] = torch.where(
                survives[:, None],
                flown.int(),
                position.int(),
            )
            mobs.mask[rows, level, slot] = survives
    return state


def _step_toward_player(
    state: EnvState,
    position: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """One axis-aligned step that closes the larger gap to the player."""
    offset = state.player_position - position
    magnitude = offset.abs()
    # Move along whichever axis is further away; ties break at random, which
    # keeps a diagonal approach from locking into a staircase.
    prefer_rows = magnitude[:, 0] > magnitude[:, 1]
    tied = magnitude[:, 0] == magnitude[:, 1]
    coin = torch.rand(state.num_envs, generator=generator, device=state.device) < 0.5
    use_rows = torch.where(tied, coin, prefer_rows)
    step = torch.zeros_like(position)
    step[:, 0] = torch.where(
        use_rows, offset[:, 0].sign(), torch.zeros_like(step[:, 0])
    )
    step[:, 1] = torch.where(
        use_rows, torch.zeros_like(step[:, 1]), offset[:, 1].sign()
    )
    return step


def _random_step(
    num_envs: int,
    device: torch.device,
    generator: torch.Generator | None,
    *,
    moves: int,
) -> Tensor:
    """Draw one step from the first ``moves`` neighbour offsets."""
    choice = torch.randint(0, moves, (num_envs,), generator=generator, device=device)
    return constants.CLOSE_BLOCKS.to(device)[choice]


def _strike_player(
    state: EnvState,
    *,
    species: Tensor,
    mob_class: int,
    striking: Tensor,
) -> EnvState:
    """Land a creature's blow, waking the player if they were asleep."""
    base = constants.MOB_DAMAGE.to(state.device)[species.long(), mob_class]
    # A sleeping player takes far more: sleeping is a gamble, not a rest stop.
    damage = mechanics.damage_to_player(
        state,
        base * (1 + 2.5 * state.is_sleeping.float())[:, None],
    )
    state.player_health = state.player_health - torch.where(
        striking,
        damage,
        torch.zeros_like(damage),
    )
    woken = state.is_sleeping & striking
    state.is_sleeping = state.is_sleeping & ~striking
    state.is_resting = state.is_resting & ~striking
    state.achievements = mechanics.unlock_achievement(
        state,
        torch.full((state.num_envs,), int(Achievement.WAKE_UP), device=state.device),
        woken,
    )
    return state


def _fire_projectile(
    state: EnvState,
    *,
    source: Tensor,
    toward: Tensor,
    species: Tensor,
    firing: Tensor,
) -> EnvState:
    """Launch a creature's projectile toward the player."""
    heading = torch.zeros_like(source)
    along_rows = toward[:, 1] == 0
    heading[:, 0] = torch.where(
        along_rows,
        toward[:, 0].sign(),
        torch.zeros_like(heading[:, 0]),
    )
    heading[:, 1] = torch.where(
        along_rows,
        torch.zeros_like(heading[:, 1]),
        toward[:, 1].sign(),
    )

    projectiles = state.mob_projectiles
    free = ~_on_level(projectiles.mask, state.player_level)
    slot = free.int().argmax(-1)
    firing = firing & free.any(-1)

    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    projectiles.position[rows, level, slot] = torch.where(
        firing[:, None],
        (source + heading).int(),
        projectiles.position[rows, level, slot],
    )
    projectiles.mask[rows, level, slot] = projectiles.mask[rows, level, slot] | firing
    projectiles.type_id[rows, level, slot] = torch.where(
        firing,
        constants.RANGED_MOB_PROJECTILE.to(state.device)[species.long()],
        projectiles.type_id[rows, level, slot],
    )
    state.mob_projectile_directions[rows, level, slot] = torch.where(
        firing[:, None],
        heading.int(),
        state.mob_projectile_directions[rows, level, slot],
    )
    return state


def _relocate(
    state: EnvState,
    *,
    field: str,
    slot: int,
    old: Tensor,
    new: Tensor,
    cooldown: Tensor,
    despawns: Tensor,
) -> EnvState:
    """Move one creature slot and keep the occupancy grid in step with it."""
    mobs: Mobs = getattr(state, field)
    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    alive = mobs.mask[rows, level, slot]

    # A creature that has wandered too far is forgotten, which is what keeps
    # the fixed slots available for creatures near the player.
    stays = (
        (new - state.player_position).abs().sum(-1) < constants.MOB_DESPAWN_DISTANCE
    ) | ~despawns
    remains = alive & stays

    occupancy = state.mob_map[rows, level]
    occupancy = scatter_tiles_where(
        occupancy,
        old,
        torch.zeros(state.num_envs, dtype=torch.bool, device=state.device),
        alive,
    )
    occupancy = scatter_tiles_where(
        occupancy,
        new,
        torch.ones(state.num_envs, dtype=torch.bool, device=state.device),
        remains,
    )
    state.mob_map[rows, level] = occupancy

    mobs.position[rows, level, slot] = torch.where(alive[:, None], new.int(), old.int())
    mobs.attack_cooldown[rows, level, slot] = cooldown
    mobs.mask[rows, level, slot] = remains
    return state


def _place_mob(
    state: EnvState,
    *,
    field: str,
    slot: Tensor,
    position: Tensor,
    species: Tensor,
    health: Tensor,
    spawning: Tensor,
) -> EnvState:
    """Fill one free slot with a new creature."""
    mobs: Mobs = getattr(state, field)
    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    mobs.position[rows, level, slot] = torch.where(
        spawning[:, None],
        position.int(),
        mobs.position[rows, level, slot],
    )
    mobs.health[rows, level, slot] = torch.where(
        spawning,
        health,
        mobs.health[rows, level, slot],
    )
    mobs.type_id[rows, level, slot] = torch.where(
        spawning,
        species.int(),
        mobs.type_id[rows, level, slot],
    )
    mobs.attack_cooldown[rows, level, slot] = torch.where(
        spawning,
        torch.zeros_like(species, dtype=torch.int32),
        mobs.attack_cooldown[rows, level, slot],
    )
    mobs.mask[rows, level, slot] = mobs.mask[rows, level, slot] | spawning
    state.mob_map[rows, level] = scatter_tiles_where(
        state.mob_map[rows, level],
        position,
        torch.ones(state.num_envs, dtype=torch.bool, device=state.device),
        spawning,
    )
    return state


def _sample_position(
    room: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Draw one eligible tile per environment from a boolean map."""
    weights = room.flatten(1).float()
    safe = torch.where(
        weights.sum(-1, keepdim=True) > 0,
        weights,
        torch.ones_like(weights),
    )
    flat = torch.multinomial(safe, 1, generator=generator).squeeze(-1)
    columns = room.shape[-1]
    return torch.stack((flat // columns, flat % columns), dim=-1).int()


def _distance_to_player(state: EnvState) -> Tensor:
    """Euclidean distance from the player to every tile of their floor."""
    rows, columns = constants.MAP_SIZE
    row_gap = (
        torch.arange(rows, device=state.device)[None, :]
        - state.player_position[:, 0, None]
    ).abs()
    column_gap = (
        torch.arange(columns, device=state.device)[None, :]
        - state.player_position[:, 1, None]
    ).abs()
    return (row_gap[:, :, None] ** 2 + column_gap[:, None, :] ** 2).float().sqrt()


def _on_level(field: Tensor, level: Tensor) -> Tensor:
    """Select each environment's current floor from a per-level field."""
    return field[torch.arange(field.shape[0], device=field.device), level.long()]


def _slot(field: Tensor, state: EnvState, slot: int) -> Tensor:
    """Select one creature slot on the player's floor, for every environment."""
    return _on_level(field, state.player_level)[:, slot]
