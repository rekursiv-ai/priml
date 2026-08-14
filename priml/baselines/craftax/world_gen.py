"""Procedural generation of the nine floors, batched across environments.

Two procedures build the world. Open floors -- the surface, the mines, the
elemental realms -- are grown from terrain noise: sea below one threshold,
coast between two, mountains above a second noise field, then ores and trees
sprinkled where the rules allow. Dungeons are built instead: eight rooms
scattered over a chunk grid, corridors cut between them, torches in the
corners, and a chest and fountain inside.

Everything is batched over a leading environment axis, so one call generates a
whole batch of distinct worlds. Where the reference draws one scalar per
environment and vectorizes, this draws the whole batch at once -- which is the
same distribution, since every draw here is independent per environment.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax import constants
from priml.baselines.craftax.constants import BlockType, ItemType
from priml.baselines.craftax.indexing import (
    scatter_tiles,
    scatter_tiles_where,
)
from priml.baselines.craftax.noise import fractal_noise
from priml.baselines.craftax.state import EnvState, empty_state
from priml.baselines.craftax.world_config import (
    LEVEL_CONFIGS,
    DungeonConfig,
    SmoothWorldConfig,
)


def generate_world(
    *,
    num_envs: int,
    generator: torch.Generator | None = None,
    device: torch.device,
) -> EnvState:
    """Build a fresh batch of worlds with the player standing at the center.

    Args:
      num_envs: Independent worlds to build.
      generator: Source of randomness; ``None`` uses the global stream.
      device: Device the world is built on.

    Returns:
      state: A playable state batch at timestep zero.

    """
    state = empty_state(num_envs=num_envs, device=device)
    rows, columns = constants.MAP_SIZE
    player_position = torch.tensor([rows // 2, columns // 2], device=device)
    state.player_position = player_position.expand(num_envs, 2).contiguous().int()

    for level, config in enumerate(LEVEL_CONFIGS):
        if isinstance(config, SmoothWorldConfig):
            floor = generate_smooth_world(
                num_envs=num_envs,
                config=config,
                player_position=player_position,
                generator=generator,
                device=device,
            )
        else:
            floor = generate_dungeon(
                num_envs=num_envs,
                config=config,
                generator=generator,
                device=device,
            )
        blocks, items, light, down_ladder, up_ladder = floor
        state.map[:, level] = blocks
        state.item_map[:, level] = items
        state.light_map[:, level] = light
        state.down_ladders[:, level] = down_ladder
        state.up_ladders[:, level] = up_ladder

    # The surface ladder starts open: the player has nothing to kill up there,
    # so requiring the usual clearing count would seal the world shut.
    state.monsters_killed[:, 0] = 10
    state.player_direction[:] = int(constants.Action.UP)
    state.player_health[:] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        getattr(state, meter)[:] = 9
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    state.boss_timesteps_to_spawn_this_round[:] = constants.BOSS_FIGHT_SPAWN_TURNS
    state.light_level[:] = daylight(
        torch.zeros(num_envs, dtype=torch.int32, device=device),
    )
    # Each episode shuffles what the six potion colours do, which is the
    # game's one genuinely hidden variable.
    state.potion_mapping = torch.stack(
        [
            torch.randperm(6, generator=generator, device=device)
            for _ in range(num_envs)
        ],
    ).int()
    return state


def generate_smooth_world(
    *,
    num_envs: int,
    config: SmoothWorldConfig,
    player_position: Tensor,
    generator: torch.Generator | None = None,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Grow one open floor from terrain noise.

    Args:
      num_envs: Independent floors to build.
      config: This floor's blocks and thresholds.
      player_position: Where the player will stand, ``[2]``.
      generator: Source of randomness; ``None`` uses the global stream.
      device: Device the floor is built on.

    Returns:
      blocks: The block at each tile, ``[envs, rows, columns]``.
      items: The item at each tile, same shape.
      light: Ambient light at each tile, same shape.
      down_ladder: Where the descent sits, ``[envs, 2]``.
      up_ladder: Where the ascent sits, ``[envs, 2]``.

    """
    shape = constants.MAP_SIZE
    coarse = (shape[0] // 16, shape[1] // 16)
    stretched = (shape[0] // 8, shape[1] // 2)
    detailed = (shape[0] // 4, shape[1] // 4)

    # Water and mountains are pushed away from the spawn so the player never
    # starts walled in or in the sea.
    distance = _distance_from(player_position, shape, device=device)
    water_clearance = (distance / config.player_proximity_map_water_strength).clamp(
        max=config.player_proximity_map_water_max
    )
    mountain_clearance = (
        distance / config.player_proximity_map_mountain_strength
    ).clamp(max=config.player_proximity_map_mountain_max)

    def noise(resolution: tuple[int, int]) -> Tensor:
        return fractal_noise(
            num_envs=num_envs,
            shape=shape,
            resolution=resolution,
            generator=generator,
            device=device,
        )

    water = noise(coarse) + water_clearance - 1.0
    blocks = torch.where(
        water > config.water_threshold,
        config.sea_block,
        config.default_block,
    )
    blocks = torch.where(
        (water > config.sand_threshold) & (blocks != config.sea_block),
        config.coast_block,
        blocks,
    )

    mountain = noise(coarse) + 0.05 + mountain_clearance - 1.0
    is_mountain = mountain > 0.7
    blocks = torch.where(is_mountain, config.mountain_block, blocks)

    # One noise field, used along both axes, cuts the passes through the
    # ranges; sharing it is what makes the paths meet at right angles.
    ridge = noise(stretched)
    blocks = torch.where(is_mountain & (ridge > 0.8), config.path_block, blocks)
    blocks = torch.where(
        is_mountain & (ridge.transpose(-2, -1) > 0.8),
        config.path_block,
        blocks,
    )
    blocks = torch.where(
        (mountain > 0.85) & (water > 0.4),
        config.inner_mountain_block,
        blocks,
    )

    tree_noise = noise(detailed)
    grows = (tree_noise > config.tree_threshold_perlin) * torch.rand(
        (num_envs, *shape),
        generator=generator,
        device=device,
    ) > config.tree_threshold_uniform
    blocks = torch.where(
        grows & (blocks == config.tree_requirement_block),
        config.tree,
        blocks,
    )

    for slot in range(5):
        seam = (blocks == config.ore_requirement_blocks[slot]) & (
            torch.rand((num_envs, *shape), generator=generator, device=device)
            < config.ore_chances[slot]
        )
        blocks = torch.where(seam, config.ores[slot], blocks)

    lava = (mountain > 0.85) & (tree_noise > 0.7)
    blocks = torch.where(lava, config.lava, blocks)
    blocks = scatter_tiles(
        blocks,
        player_position.expand(num_envs, 2),
        torch.full((num_envs,), config.player_spawn, device=device),
    )

    light = torch.full((num_envs, *shape), config.default_light, device=device)
    items = torch.zeros((num_envs, *shape), dtype=torch.int32, device=device)

    eligible = (blocks == config.valid_ladder).flatten(1).float()
    down_ladder = _sample_tile(eligible, shape, generator=generator, device=device)
    up_ladder = _sample_tile(eligible, shape, generator=generator, device=device)
    if config.ladder_down:
        items = scatter_tiles(
            items,
            down_ladder,
            torch.full((num_envs,), int(ItemType.LADDER_DOWN), device=device),
        )
    if config.ladder_up:
        items = scatter_tiles(
            items,
            up_ladder,
            torch.full((num_envs,), int(ItemType.LADDER_UP), device=device),
        )
        light = _brighten_around(light, up_ladder, ambient=config.default_light)

    if config.lava == BlockType.LAVA:
        # Lava lights its surroundings. The kernel is symmetric, so a
        # cross-correlation is the same as the convolution the reference uses.
        glow = torch.tensor(
            [[0.2, 0.7, 0.2], [0.7, 1.0, 0.7], [0.2, 0.7, 0.2]],
            device=device,
        )
        light = light + functional.conv2d(
            lava.float().unsqueeze(1),
            glow[None, None],
            padding=1,
        ).squeeze(1)
    return blocks.int(), items.int(), light.clamp(0.0, 1.0), down_ladder, up_ladder


def generate_dungeon(
    *,
    num_envs: int,
    config: DungeonConfig,
    generator: torch.Generator | None = None,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build one floor of rooms joined by corridors.

    Rooms are placed one per chunk of a coarse grid, without replacement, so
    they cannot overlap. Each new room is then joined to one already-connected
    room by an L-shaped corridor, which guarantees the whole floor is
    reachable rather than leaving isolated pockets.

    Args:
      num_envs: Independent floors to build.
      config: This floor's special, fountain, and texture blocks.
      generator: Source of randomness; ``None`` uses the global stream.
      device: Device the floor is built on.

    Returns:
      blocks: The block at each tile, ``[envs, rows, columns]``.
      items: The item at each tile, same shape.
      light: Ambient light at each tile, same shape.
      down_ladder: Where the descent sits, ``[envs, 2]``.
      up_ladder: Where the ascent sits, ``[envs, 2]``.

    """
    shape = constants.MAP_SIZE
    chunk, num_rooms = 16, 8
    smallest, largest = 5, 10
    chunks_down, chunks_across = shape[0] // chunk, shape[1] // chunk

    blocks = torch.full(
        (num_envs, *shape),
        int(BlockType.WALL),
        dtype=torch.int32,
        device=device,
    )
    items = torch.zeros((num_envs, *shape), dtype=torch.int32, device=device)

    sizes = torch.randint(
        smallest,
        largest,
        (num_envs, num_rooms, 2),
        generator=generator,
        device=device,
    )
    # Rooms take distinct chunks, so a permutation prefix is exactly the
    # "choose without replacement" the reference performs by zeroing weights.
    chunks = torch.stack(
        [
            torch.randperm(
                chunks_down * chunks_across,
                generator=generator,
                device=device,
            )[:num_rooms]
            for _ in range(num_envs)
        ],
    )
    offsets = torch.randint(
        0,
        chunk - smallest,
        (num_envs, num_rooms, 2),
        generator=generator,
        device=device,
    )
    corners = (
        torch.stack(
            (chunks % chunks_across, chunks // chunks_across),
            dim=-1,
        )
        * chunk
        + offsets
    )

    rows = torch.arange(shape[0], device=device)[None, :, None]
    columns = torch.arange(shape[1], device=device)[None, None, :]
    for room in range(num_rooms):
        top, left = corners[:, room, 0, None, None], corners[:, room, 1, None, None]
        height = sizes[:, room, 0, None, None]
        width = sizes[:, room, 1, None, None]
        inside = (
            (rows >= top)
            & (rows < top + height)
            & (columns >= left)
            & (columns < left + width)
        )
        blocks = torch.where(inside, int(BlockType.PATH), blocks)
        items = _place_room_torches(
            items,
            corner=corners[:, room],
            size=sizes[:, room],
        )
        blocks = _place_room_features(
            blocks,
            corner=corners[:, room],
            size=sizes[:, room],
            config=config,
            generator=generator,
            device=device,
        )

    for room in range(1, num_rooms):
        blocks = _carve_corridor(
            blocks,
            source=corners[:, room],
            sink=corners[:, room - 1],
            device=device,
        )

    # The special block sits just inside the first room, which is where the
    # enchantment tables live on the floors that have them.
    blocks = scatter_tiles(
        blocks,
        corners[:, 0] + 2,
        torch.full((num_envs,), config.special_block, device=device),
    )

    # A wall touching a corridor is masonry; one buried behind other walls is
    # never seen, so it reads as darkness instead.
    adjacency = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
    )
    near_path = (
        functional.conv2d(
            (blocks != int(BlockType.WALL)).float().unsqueeze(1),
            adjacency[None, None],
            padding=1,
        ).squeeze(1)
        > 0.5
    )
    speckle = torch.rand((num_envs, *shape), generator=generator, device=device) < 0.1
    walls = torch.where(speckle, int(BlockType.WALL_MOSS), int(BlockType.WALL))
    textured = torch.where(
        speckle & (blocks == int(BlockType.PATH)) & (items == int(ItemType.NONE)),
        config.rare_path_replacement_block,
        blocks,
    )
    is_wall = (blocks == int(BlockType.WALL)) & near_path
    blocks = torch.where(
        is_wall,
        walls,
        torch.where(near_path, textured, int(BlockType.DARKNESS)),
    )

    light = torch.ones((num_envs, *shape), device=device)
    eligible = (blocks == int(BlockType.PATH)).flatten(1).float()
    down_ladder = _sample_tile(eligible, shape, generator=generator, device=device)
    up_ladder = _sample_tile(eligible, shape, generator=generator, device=device)
    items = scatter_tiles(
        items,
        down_ladder,
        torch.full((num_envs,), int(ItemType.LADDER_DOWN), device=device),
    )
    items = scatter_tiles(
        items,
        up_ladder,
        torch.full((num_envs,), int(ItemType.LADDER_UP), device=device),
    )
    return blocks.int(), items.int(), light, down_ladder, up_ladder


def daylight(timestep: Tensor) -> Tensor:
    """Return how bright the surface is, on ``[0, 1]``.

    The cube is what makes the day feel like a day: a plain cosine would
    spend half the cycle in twilight, while cubing flattens the peak into a
    long bright afternoon and compresses dusk and dawn. The episode starts a
    third of the way through the cycle rather than at midnight.

    Args:
      timestep: Steps elapsed this episode, any shape.

    Returns:
      light: Ambient surface light, same shape.

    """
    phase = (timestep.float() / constants.DAY_LENGTH) % 1.0 + 0.3
    return 1.0 - (torch.pi * phase).cos().abs() ** 3


def _distance_from(
    position: Tensor,
    shape: tuple[int, int],
    *,
    device: torch.device,
) -> Tensor:
    """Euclidean distance from one tile to every other, ``[rows, columns]``."""
    rows = (torch.arange(shape[0], device=device) - position[0]).abs()
    columns = (torch.arange(shape[1], device=device) - position[1]).abs()
    return (rows[:, None] ** 2 + columns[None, :] ** 2).float().sqrt()


def _sample_tile(
    weights: Tensor,
    shape: tuple[int, int],
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> Tensor:
    """Draw one tile per environment, proportional to ``weights``.

    A floor with no eligible tile would make the draw undefined, so an
    all-zero row falls back to a uniform choice rather than raising.
    """
    safe = torch.where(
        weights.sum(-1, keepdim=True) > 0,
        weights,
        torch.ones_like(weights),
    )
    flat = torch.multinomial(safe, 1, generator=generator).squeeze(-1)
    return torch.stack((flat // shape[1], flat % shape[1]), dim=-1).int().to(device)


def _place_room_torches(items: Tensor, *, corner: Tensor, size: Tensor) -> Tensor:
    """Light each room from its four corners."""
    for down, across in ((0, 0), (1, 0), (0, 1), (1, 1)):
        offset = torch.stack(
            (
                (size[:, 0] - 1) * down,
                (size[:, 1] - 1) * across,
            ),
            dim=-1,
        )
        items = scatter_tiles(
            items,
            corner + offset,
            torch.full((items.shape[0],), int(ItemType.TORCH), device=items.device),
        )
    return items


def _place_room_features(
    blocks: Tensor,
    *,
    corner: Tensor,
    size: Tensor,
    config: DungeonConfig,
    generator: torch.Generator | None,
    device: torch.device,
) -> Tensor:
    """Put a chest in every room and a fountain in about half of them."""
    num_envs = blocks.shape[0]
    # Both sit strictly inside the room so a corridor cannot open onto them.
    inner = (size - 2).clamp(min=1)
    chest = (
        corner
        + 1
        + (torch.rand((num_envs, 2), generator=generator, device=device) * inner).long()
    )
    blocks = scatter_tiles(
        blocks,
        chest,
        torch.full((num_envs,), int(BlockType.CHEST), device=device),
    )
    fountain = (
        corner
        + 1
        + (torch.rand((num_envs, 2), generator=generator, device=device) * inner).long()
    )
    has_fountain = torch.rand(num_envs, generator=generator, device=device) > 0.5
    return scatter_tiles_where(
        blocks,
        fountain,
        torch.full((num_envs,), config.fountain_block, device=device),
        has_fountain,
    )


def _carve_corridor(
    blocks: Tensor,
    *,
    source: Tensor,
    sink: Tensor,
    device: torch.device,
) -> Tensor:
    """Cut an L-shaped passage between two rooms through solid wall only.

    Only wall is replaced: running the corridor over an existing room would
    erase its chest or fountain.
    """
    rows = torch.arange(blocks.shape[-2], device=device)[None, :, None]
    columns = torch.arange(blocks.shape[-1], device=device)[None, None, :]
    source_row, source_column = source[:, 0, None, None], source[:, 1, None, None]
    sink_row, sink_column = sink[:, 0, None, None], sink[:, 1, None, None]

    horizontal = (
        (rows == source_row)
        & (columns >= torch.minimum(source_column, sink_column))
        & (columns <= torch.maximum(source_column, sink_column))
    )
    vertical = (
        (columns == sink_column)
        & (rows >= torch.minimum(source_row, sink_row))
        & (rows <= torch.maximum(source_row, sink_row))
    )
    return torch.where(
        (horizontal | vertical) & (blocks == int(BlockType.WALL)),
        int(BlockType.PATH),
        blocks,
    )


def _brighten_around(light: Tensor, position: Tensor, *, ambient: float) -> Tensor:
    """Raise the light around an ascent so its tile is never pitch dark."""
    glow = constants.TORCH_LIGHT_MAP.to(light.device) * (1 - ambient) + ambient
    rows = torch.arange(9, device=light.device) - 4
    for row_offset in range(9):
        for column_offset in range(9):
            offset = torch.stack(
                (rows[row_offset], rows[column_offset]),
            ).expand(light.shape[0], 2)
            light = scatter_tiles(
                light,
                position + offset,
                glow[row_offset, column_offset].expand(light.shape[0]),
            )
    return light
