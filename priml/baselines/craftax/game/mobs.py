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

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import (
    Achievement,
    BlockType,
    ProjectileType,
)
from priml.baselines.craftax.game.indexing import (
    scatter_tiles_where,
)
from priml.baselines.craftax.game.state import EnvState, Mobs


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
    fighting_boss = mechanics.is_fighting_boss(state)
    boss_wave = state.boss_timesteps_to_spawn_this_round >= 1
    monster_rate = (1 + 2 * uncleared.int()) * torch.where(
        fighting_boss,
        boss_wave.int() * 1000,
        torch.ones_like(level),
    )

    grid = mechanics.current_map(state)
    distance = _distance_to_player(state)
    unoccupied = ~mechanics.current_mobs(state)
    walkable = (
        (grid == int(BlockType.GRASS))
        | (grid == int(BlockType.PATH))
        | (grid == int(BlockType.FIRE_GRASS))
        | (grid == int(BlockType.ICE_GRASS))
    )
    passive_room = (
        (distance > 3)
        & (distance < constants.MOB_DESPAWN_DISTANCE)
        & unoccupied
        & walkable
    )
    monster_distance = torch.where(
        fighting_boss[:, None, None],
        distance <= 6,
        distance > 9,
    )
    grave = (
        (grid == int(BlockType.GRAVE))
        | (grid == int(BlockType.GRAVE2))
        | (grid == int(BlockType.GRAVE3))
    )
    monster_tiles = torch.where(fighting_boss[:, None, None], grave, walkable)
    monster_room = (
        monster_distance
        & (distance < constants.MOB_DESPAWN_DISTANCE)
        & unoccupied
        & monster_tiles
    )

    chances = constants.FLOOR_MOB_SPAWN_CHANCE.to(state.device)[level]
    floor_species = constants.FLOOR_MOB_TYPE.to(state.device)[level]
    boss_species = constants.FLOOR_MOB_TYPE.to(state.device)[state.boss_progress.long()]
    species = torch.where(fighting_boss[:, None], boss_species, floor_species)
    for field, column, mob_class in (
        ("passive_mobs", 0, 0),
        ("melee_mobs", 1, 1),
        ("ranged_mobs", 2, 2),
    ):
        mobs: object = getattr(state, field)  # pyright: ignore[reportAny] -- Dynamic state fields are narrowed below.
        assert isinstance(mobs, Mobs)
        alive = _on_level(mobs.mask, state.player_level)
        chance = chances[:, column]
        if column == 1:
            chance = chance + chances[:, 3] * (1.0 - state.light_level) ** 2
        rate = torch.ones_like(chance) if column == 0 else monster_rate
        room = passive_room if column == 0 else monster_room
        spawning = (
            (alive.sum(-1) < alive.shape[-1])
            & (
                torch.rand(state.num_envs, generator=generator, device=state.device)
                < chance * rate
            )
            & room.flatten(1).any(-1)
        )
        if column == 0:
            spawning &= ~fighting_boss
        if column == 2:
            water_only = species[:, column] == 5
            room = torch.where(
                water_only[:, None, None],
                grid == int(BlockType.WATER),
                room,
            )
            room = torch.where(fighting_boss[:, None, None], grave, room)
            room &= (
                unoccupied
                & monster_distance
                & (distance < constants.MOB_DESPAWN_DISTANCE)
            )
            spawning &= room.flatten(1).any(-1)

        place = _sample_position(room, generator=generator)
        slot = (~alive).int().argmax(-1)
        health = constants.MOB_HEALTH.to(state.device)[
            species[:, column].long(),
            mob_class,
        ]
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

        toward = _step_toward_player(state, position, generator=generator)
        wander = position + _random_step(
            state.num_envs,
            state.device,
            generator,
            moves=4,
        )
        proposed = torch.where((gap >= 6)[:, None], position + toward, wander)
        proposed = torch.where((gap <= 3)[:, None], position - toward, proposed)
        use_wander = (
            torch.rand(state.num_envs, generator=generator, device=state.device) <= 0.85
        )
        proposed = torch.where(use_wander[:, None], wander, proposed)

        collides = constants.MOB_COLLIDES_WITH.to(state.device)[
            state.player_level.long(),
            2,
        ]
        can_retreat = mechanics.can_walk_on(state, proposed, collides)
        firing = ((gap >= 4) & (gap <= 5)) | ((gap <= 3) & ~can_retreat)
        firing &= alive & (_slot(mobs.attack_cooldown, state, slot) <= 0)
        state = _fire_projectile(
            state,
            source=position,
            toward=offset,
            species=_slot(mobs.type_id, state, slot),
            firing=firing,
        )
        proposed = torch.where(firing[:, None], position, proposed)

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
        mobs: object = getattr(state, field)  # pyright: ignore[reportAny] -- Dynamic state fields are narrowed below.
        assert isinstance(mobs, Mobs)
        directions: object = getattr(state, directions_field)  # pyright: ignore[reportAny] -- Dynamic state fields are narrowed below.
        assert isinstance(directions, Tensor)
        for slot in range(mobs.mask.shape[-1]):
            alive = _slot(mobs.mask, state, slot)
            position = _slot(mobs.position, state, slot)
            heading = _slot(directions, state, slot)
            flown = position + heading
            species = _slot(mobs.type_id, state, slot)

            if hurts_player:
                hits = alive & (flown == state.player_position).all(-1)
                damage = constants.MOB_DAMAGE.to(state.device)[species.long(), 3]
                state.player_health = state.player_health - torch.where(
                    hits,
                    mechanics.damage_to_player(state, damage),
                    torch.zeros_like(state.player_health),
                )
                state.is_sleeping = state.is_sleeping & ~hits
                state.is_resting = state.is_resting & ~hits
            else:
                state, hits = _strike_with_projectile(
                    state,
                    species=species,
                    at=(position, flown),
                    alive=alive,
                )

            # A projectile stops at the first solid thing it meets.
            blocked = constants.SOLID_BLOCK.to(state.device)[
                mechanics.block_at(state, flown).long()
            ]
            if hurts_player:
                blocked |= mechanics.is_occupied(state, flown)
            survives = alive & ~hits & ~blocked & mechanics.in_bounds(flown)

            rows = torch.arange(state.num_envs, device=state.device)
            level = state.player_level.long()
            mobs.position[rows, level, slot] = torch.where(
                survives[:, None],
                flown.int(),
                position.int(),
            )
            mobs.mask[rows, level, slot] = survives
    return state


# Arrows add half their physical damage in the bow's element and scale with dexterity;
# spells scale with intelligence. The projectile checks its current tile first, so a
# creature that stepped into its path is not skipped.
def _strike_with_projectile(
    state: EnvState,
    *,
    species: Tensor,
    at: tuple[Tensor, Tensor],
    alive: Tensor,
) -> tuple[EnvState, Tensor]:
    """Land a player projectile on a creature at its tile or the next one."""
    damage = constants.MOB_DAMAGE.to(state.device)[species.long(), 3] * alive[:, None]
    arrow = (species == int(ProjectileType.ARROW)) | (
        species == int(ProjectileType.ARROW2)
    )
    spell = (species == int(ProjectileType.FIREBALL)) | (
        species == int(ProjectileType.ICEBALL)
    )
    element = torch.zeros_like(damage).scatter_(
        -1,
        state.bow_enchantment.long()[:, None],
        damage[:, :1] / 2,
    )
    element[:, 0] = 0.0
    damage = damage + element * arrow[:, None]
    damage = (
        damage
        * torch.where(
            arrow,
            1 + 0.2 * (state.player_dexterity - 1),
            torch.where(spell, 1 + 0.5 * (state.player_intelligence - 1), 1.0),
        )[:, None]
    )

    hits = torch.zeros_like(alive)
    for target in at:
        remaining = damage * ~hits[:, None]
        state, struck = _projectile_hits_tile(state, target=target, damage=remaining)
        hits = hits | (struck & alive)
    return state, hits


def _projectile_hits_tile(
    state: EnvState,
    *,
    target: Tensor,
    damage: Tensor,
) -> tuple[EnvState, Tensor]:
    """Apply ``damage`` to any creature of any class standing on ``target``."""
    struck = torch.zeros(state.num_envs, dtype=torch.bool, device=state.device)
    killed_monster = struck.clone()
    killed_any = struck.clone()
    # A projectile kill unlocks the monster achievement, but shooting a cow
    # is not eating it: passive kills neither feed nor unlock.
    for field, mob_class, can_unlock in (
        ("melee_mobs", 1, True),
        ("passive_mobs", 0, False),
        ("ranged_mobs", 2, True),
    ):
        mobs: object = getattr(state, field)  # pyright: ignore[reportAny] -- Dynamic state fields are narrowed below.
        assert isinstance(mobs, Mobs)
        mobs, killed, hit, achievements = mechanics.attack_mob_class(
            state,
            mobs,
            position=target,
            damage=damage,
            mob_class=mob_class,
            can_unlock=torch.full_like(struck, can_unlock),
        )
        setattr(state, field, mobs)
        state.achievements = achievements
        struck = struck | hit
        killed_any = killed_any | killed
        if mob_class:
            killed_monster = killed_monster | killed

    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    state.monsters_killed[rows, level] += killed_monster.int()
    state.mob_map[rows, level] = scatter_tiles_where(
        state.mob_map[rows, level],
        target,
        torch.zeros(state.num_envs, dtype=torch.bool, device=state.device),
        killed_any,
    )
    return state, struck


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
        use_rows,
        offset[:, 0].sign(),
        torch.zeros_like(step[:, 0]),
    )
    step[:, 1] = torch.where(
        use_rows,
        torch.zeros_like(step[:, 1]),
        offset[:, 1].sign(),
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
    mobs: object = getattr(state, field)  # pyright: ignore[reportAny] -- Dynamic state fields are narrowed below.
    assert isinstance(mobs, Mobs)
    rows = torch.arange(state.num_envs, device=state.device)
    level = state.player_level.long()
    alive = mobs.mask[rows, level, slot]

    # A creature that has wandered too far is forgotten, which is what keeps
    # the fixed slots available for creatures near the player.
    stays = (
        (old - state.player_position).abs().sum(-1) < constants.MOB_DESPAWN_DISTANCE
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
    mobs: object = getattr(state, field)  # pyright: ignore[reportAny] -- Dynamic state fields are narrowed below.
    assert isinstance(mobs, Mobs)
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
