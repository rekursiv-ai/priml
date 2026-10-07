"""Tests for the shared game rules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from priml.baselines.craftax.game import constants, indexing, mechanics
from priml.baselines.craftax.game.constants import BlockType
from priml.baselines.craftax.game.state import EnvState, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.player_strength[:] = 1
    state.player_dexterity[:] = 1
    state.player_intelligence[:] = 1
    state.map[:] = int(BlockType.GRASS)
    return state


def test_meter_caps_rise_with_their_attribute() -> None:
    state = _state()
    assert mechanics.max_health(state).tolist() == [9, 9]
    state.player_strength[:] = 5
    assert mechanics.max_health(state).tolist() == [13, 13]
    state.player_dexterity[:] = 5
    assert mechanics.max_food(state).tolist() == [17, 17]
    assert mechanics.max_drink(state).tolist() == [17, 17]
    assert mechanics.max_energy(state).tolist() == [17, 17]
    state.player_intelligence[:] = 5
    assert mechanics.max_mana(state).tolist() == [21, 21]


def test_bare_hands_do_less_damage_than_a_sword() -> None:
    state = _state()
    unarmed = mechanics.player_damage(state)[:, 0]
    state.inventory.sword[:] = 4
    armed = mechanics.player_damage(state)[:, 0]
    assert unarmed.tolist() == [1.0, 1.0]
    assert armed.tolist() == [8.0, 8.0]


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu_torch_cuda)],
)
def test_damage_sums_in_xlas_order(device: str) -> None:
    """XLA pairs the four armour pieces, then adds the elements left to right.

    Values from upstream ``get_damage_done_to_player`` jitted on CUDA with
    ``--xla_gpu_deterministic_ops=true``; with autotuning on, XLA may round this
    last bit either way from one process to the next. Torch's own reductions
    miss it on about 1.5% of hits: the first two rows catch CPU's armour order,
    the last two CUDA's element order.
    """
    state = empty_state(num_envs=4, device=torch.device(device))
    state.player_level[:] = torch.tensor([0, 8, 2, 4])
    state.inventory.armour[:] = torch.tensor(
        [[1, 2, 2, 2], [1, 2, 2, 2], [1, 1, 0, 1], [0, 0, 2, 0]],
    )
    state.armour_enchantments[:] = torch.tensor(
        [[0, 0, 2, 2], [2, 1, 2, 0], [2, 2, 0, 0], [1, 1, 1, 0]],
    )
    damage = torch.tensor(
        [[6.0, 1.0, 1.0], [17.5, 0.0, 0.0], [21.0, 3.5, 3.5], [14.0, 10.5, 10.5]],
        device=device,
    )
    landed = mechanics.damage_to_player(state, damage)
    assert landed.tolist() == [
        3.3999996185302734,
        7.874998569488525,
        20.30000114440918,
        25.899999618530273,
    ]


def test_strength_doubles_physical_damage_at_the_cap() -> None:
    state = _state()
    state.inventory.sword[:] = 2
    state.player_strength[:] = 5
    assert mechanics.player_damage(state)[:, 0].tolist() == [6.0, 6.0]


def test_an_enchanted_sword_adds_its_element() -> None:
    state = _state()
    state.inventory.sword[:] = 2
    state.sword_enchantment[:] = torch.tensor([1, 2])
    state.player_intelligence[:] = torch.tensor([3, 5])
    damage = mechanics.player_damage(state)
    assert damage[0].tolist() == pytest.approx([3.0, 1.65, 0.0])
    assert damage[1].tolist() == pytest.approx([3.0, 0.0, 1.8])
    assert damage.shape == (2, 3)


def test_armour_reduces_incoming_damage() -> None:
    state = _state()
    incoming = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    bare = mechanics.damage_to_player(state, incoming)
    state.inventory.armour[:] = 2
    armoured = mechanics.damage_to_player(state, incoming)
    assert bare.tolist() == [10.0, 10.0]
    # Four pieces at two points each block eight tenths of the blow.
    assert armoured.tolist() == pytest.approx([2.0, 2.0])


def test_a_tier_two_piece_blocks_a_fifth_of_physical_damage() -> None:
    state = _state()
    state.inventory.armour[:, 0] = 2
    incoming = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert mechanics.damage_to_player(state, incoming).tolist() == pytest.approx(
        [8.0, 8.0],
    )
    assert "per tier" in (mechanics.damage_to_player.__doc__ or "")


def test_docstrings_carry_no_placeholder_sections() -> None:
    source = Path(mechanics.__file__).read_text()
    assert "result: The Tensor." not in source
    assert "      state: State.\n" not in source


def test_each_armour_enchantment_reduces_only_its_element() -> None:
    state = _state()
    state.armour_enchantments[0] = 1
    state.armour_enchantments[1] = 2
    damage = torch.tensor([[0.0, 10.0, 10.0], [0.0, 10.0, 10.0]])
    assert mechanics.damage_to_player(state, damage).tolist() == pytest.approx(
        [12.0, 12.0],
    )


def test_the_boss_floor_amplifies_incoming_damage() -> None:
    state = _state()
    incoming = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    state.player_level[:] = constants.NUM_LEVELS - 1
    assert mechanics.damage_to_player(state, incoming).tolist() == pytest.approx(
        [15.0, 15.0],
    )


def test_the_boss_is_shielded_while_its_summons_live() -> None:
    state = _state()
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.boss_timesteps_to_spawn_this_round[:] = 0
    assert mechanics.is_boss_vulnerable(state).tolist() == [True, True]
    state.melee_mobs.mask[:, constants.NUM_LEVELS - 1, 0] = True
    assert mechanics.is_boss_vulnerable(state).tolist() == [False, False]


def test_the_boss_is_shielded_between_waves() -> None:
    state = _state()
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.boss_timesteps_to_spawn_this_round[:] = 3
    assert mechanics.is_boss_vulnerable(state).tolist() == [False, False]


def test_boss_vulnerability_checks_each_mob_and_spawn_boundary() -> None:
    state = _state(num_envs=4)
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.ranged_mobs.mask[0, constants.NUM_LEVELS - 1, 1] = True
    state.melee_mobs.mask[1, constants.NUM_LEVELS - 1, 0] = True
    state.boss_timesteps_to_spawn_this_round[:] = torch.tensor([0, 0, 1, 0])
    assert mechanics.is_boss_vulnerable(state).tolist() == [False, False, False, True]


def test_walking_is_refused_into_stone_and_allowed_onto_grass() -> None:
    state = _state()
    state.map[:, 0, 5, 5] = int(BlockType.STONE)
    never_collides = torch.zeros(2, 3, dtype=torch.bool)
    assert mechanics.can_walk_on(
        state,
        torch.tensor([[5, 5], [6, 6]]),
        never_collides,
    ).tolist() == [False, True]


def test_walking_off_the_map_is_refused_rather_than_wrapping() -> None:
    # The indexing helpers wrap a negative coordinate, so bounds must be
    # checked here or stepping off the top edge teleports to the bottom.
    state = _state()
    never_collides = torch.zeros(2, 3, dtype=torch.bool)
    assert mechanics.can_walk_on(
        state,
        torch.tensor([[-1, 5], [5, -1]]),
        never_collides,
    ).tolist() == [False, False]


def test_bounds_include_zero_and_the_last_tile_but_not_the_extent() -> None:
    extent = constants.MAP_SIZE[0]
    positions = torch.tensor([[0, 0], [extent - 1, extent - 1], [extent, 0]])
    assert mechanics.in_bounds(positions).tolist() == [True, True, False]


def test_a_land_creature_will_not_enter_water() -> None:
    state = _state()
    state.map[:, 0, 5, 5] = int(BlockType.WATER)
    land = torch.tensor([[False, True, True], [False, True, True]])
    swims = torch.tensor([[True, False, True], [True, False, True]])
    target = torch.tensor([[5, 5], [5, 5]])
    assert mechanics.can_walk_on(state, target, land).tolist() == [False, False]
    assert mechanics.can_walk_on(state, target, swims).tolist() == [True, True]


def test_ground_and_lava_respect_their_collision_flags() -> None:
    state = _state()
    state.map[:, 0, 5, 5] = int(BlockType.LAVA)
    position = torch.tensor([[5, 5], [5, 5]])
    collides_with_ground = torch.tensor([[True, False, False]] * 2)
    collides_with_lava = torch.tensor([[False, False, True]] * 2)
    avoids_ground = torch.tensor([[False, False, False]] * 2)

    state.map[:, 0, 5, 5] = int(BlockType.GRASS)
    assert mechanics.can_walk_on(state, position, collides_with_ground).tolist() == [
        False,
        False,
    ]
    assert mechanics.can_walk_on(state, position, avoids_ground).tolist() == [
        True,
        True,
    ]
    state.map[:, 0, 5, 5] = int(BlockType.LAVA)
    assert mechanics.can_walk_on(state, position, collides_with_lava).tolist() == [
        False,
        False,
    ]
    assert mechanics.can_walk_on(state, position, avoids_ground).tolist() == [
        True,
        True,
    ]


def test_a_tile_holding_a_creature_is_occupied() -> None:
    state = _state()
    state.mob_map[:, 0, 5, 5] = True
    never_collides = torch.zeros(2, 3, dtype=torch.bool)
    assert mechanics.can_walk_on(
        state,
        torch.tensor([[5, 5], [6, 6]]),
        never_collides,
    ).tolist() == [False, True]


def test_the_player_blocks_their_own_tile() -> None:
    state = _state()
    assert mechanics.is_occupied(state, state.player_position).tolist() == [True, True]


def test_player_occupancy_keeps_the_environment_axis() -> None:
    state = _state(num_envs=3)
    state.player_position[:] = torch.tensor([[10, 10], [11, 11], [12, 12]])
    targets = torch.tensor([[10, 10], [11, 10], [12, 12]])
    assert mechanics.is_occupied(state, targets).tolist() == [True, False, True]


def test_adjacency_finds_a_neighbouring_block_but_not_a_distant_one() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.CRAFTING_TABLE)
    assert mechanics.is_near_block(state, int(BlockType.CRAFTING_TABLE)).tolist() == [
        True,
        True,
    ]
    assert mechanics.is_near_block(state, int(BlockType.FURNACE)).tolist() == [
        False,
        False,
    ]


def test_adjacency_checks_all_eight_neighbors() -> None:
    state = _state()
    state.map[:, 0, 9, 10] = int(BlockType.CRAFTING_TABLE)
    assert mechanics.is_near_block(
        state,
        int(BlockType.CRAFTING_TABLE),
    ).tolist() == [True, True]


def test_adjacency_ignores_the_tile_underfoot() -> None:
    # A table the player stands on is not usable; it must be beside them.
    state = _state()
    state.map[:, 0, 10, 10] = int(BlockType.CRAFTING_TABLE)
    assert mechanics.is_near_block(state, int(BlockType.CRAFTING_TABLE)).tolist() == [
        False,
        False,
    ]


def test_clipping_holds_meters_and_stocks_in_range() -> None:
    state = _state()
    state.player_health[:] = 99.0
    state.player_food[:] = -5
    state.inventory.wood[:] = 500
    clipped = mechanics.clip_meters(state)
    assert clipped.player_health.tolist() == [9.0, 9.0]
    assert clipped.player_food.tolist() == [0, 0]
    assert clipped.inventory.wood.tolist() == [99, 99]


def test_clipping_covers_all_player_meters() -> None:
    state = _state()
    state.player_health[:] = -1.0
    state.player_dexterity[:] = 5
    state.player_food[:] = 99
    state.player_drink[:] = -1
    state.player_energy[:] = torch.tensor([-1, 99])
    state.player_mana[:] = -1

    mechanics.clip_meters(state)

    assert state.player_health.tolist() == [0.0, 0.0]
    assert state.player_food.tolist() == [17, 17]
    assert state.player_drink.tolist() == [0, 0]
    assert state.player_energy.tolist() == [0, 17]
    assert state.player_mana.tolist() == [0, 0]


def test_unlocking_an_achievement_touches_only_the_named_one() -> None:
    state = _state()
    updated = mechanics.unlock_achievement(
        state,
        torch.tensor([int(constants.Achievement.COLLECT_WOOD)] * 2),
        torch.tensor([True, False]),
    )
    assert updated[0, int(constants.Achievement.COLLECT_WOOD)]
    assert not updated[1, int(constants.Achievement.COLLECT_WOOD)]
    assert not updated[:, int(constants.Achievement.PLACE_TABLE)].any()


def test_attacking_reduces_health_and_kills_at_zero() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 3.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)

    mobs, killed, struck, _ = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 5], [5, 5]]),
        damage=torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        mob_class=1,
        can_unlock=torch.tensor([True, True]),
    )

    assert struck.tolist() == [True, True]
    assert killed.tolist() == [False, True]
    assert mobs.health[0, 0, 0].item() == pytest.approx(1.0)
    assert mobs.health[1, 0, 0].item() == 0.0
    assert mobs.mask[:, 0, 0].tolist() == [True, False]
    assert torch.equal(mobs.position, state.melee_mobs.position)
    assert torch.equal(mobs.attack_cooldown, state.melee_mobs.attack_cooldown)
    assert torch.equal(mobs.type_id, state.melee_mobs.type_id)


def test_a_summon_resists_as_its_type_not_the_boss_floor() -> None:
    """Upstream reads defense by the target's type (game_logic_utils.py:50-55).

    The boss floor's own row resists nothing, but its summons keep their kind's
    armour: a type-4 creature halves physical blows, a type-6 one ignores fire.
    """
    state = _state()
    state.player_level[:] = 8
    state.melee_mobs.mask[:, 8, 0] = True
    state.melee_mobs.health[:, 8, 0] = 10.0
    state.melee_mobs.type_id[:, 8, 0] = torch.tensor([4, 6], dtype=torch.int32)
    state.melee_mobs.position[:, 8, 0] = torch.tensor([5, 5], dtype=torch.int32)

    mobs, _, struck, _ = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 5], [5, 5]]),
        damage=torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        mob_class=1,
        can_unlock=torch.tensor([True, True]),
    )

    assert struck.tolist() == [True, True]
    assert mobs.health[:, 8, 0].tolist() == [9.0, 10.0]


def test_attacking_an_empty_tile_does_nothing() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 3.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)

    mobs, killed, struck, _ = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[9, 9], [9, 9]]),
        damage=torch.tensor([[5.0, 0.0, 0.0]] * 2),
        mob_class=1,
        can_unlock=torch.tensor([True, True]),
    )

    assert struck.tolist() == [False, False]
    assert killed.tolist() == [False, False]
    assert mobs.health[0, 0, 0].item() == pytest.approx(3.0)


def test_attacks_select_each_environment_target_slot() -> None:
    state = _state(num_envs=3)
    slots = torch.tensor([0, 1, 2])
    rows = torch.arange(3)
    state.melee_mobs.mask[rows, 0, slots] = True
    state.melee_mobs.health[rows, 0, slots] = 4.0
    state.melee_mobs.position[rows, 0, slots] = torch.tensor(
        [[5, 5]] * 3,
        dtype=torch.int32,
    )

    mobs, killed, struck, _ = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 5]] * 3),
        damage=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        mob_class=1,
        can_unlock=torch.zeros(3, dtype=torch.bool),
    )

    assert struck.tolist() == [True, True, True]
    assert killed.tolist() == [False, False, False]
    assert mobs.health[rows, 0, slots].tolist() == [3.0, 2.0, 1.0]


def test_a_kill_unlocks_the_species_achievement() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 1.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)

    _, _, _, achievements = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 5], [5, 5]]),
        damage=torch.tensor([[5.0, 0.0, 0.0]] * 2),
        mob_class=1,
        can_unlock=torch.tensor([True, False]),
    )

    zombie = int(constants.Achievement.DEFEAT_ZOMBIE)
    assert achievements[0, zombie]
    assert not achievements[1, zombie]


def test_current_level_accessors_and_boss_progress_use_each_environment() -> None:
    state = _state(num_envs=3)
    state.player_level[:] = torch.tensor([0, 1, 2])
    rows = torch.arange(3)
    state.item_map[rows, rows, 10, 10] = torch.tensor([4, 5, 6], dtype=torch.int32)
    state.light_map[rows, rows, 10, 10] = torch.tensor([0.2, 0.5, 0.8])
    state.mob_map[rows, rows, 10, 10] = True
    state.map[rows, rows, 10, 10] = torch.tensor([7, 8, 9], dtype=torch.int32)
    state.boss_progress[:] = torch.tensor(
        [constants.NUM_LEVELS - 2, constants.NUM_LEVELS - 1, 0],
    )

    assert mechanics.current_items(state)[:, 10, 10].tolist() == [4, 5, 6]
    assert mechanics.current_light(state)[:, 10, 10].tolist() == pytest.approx(
        [0.2, 0.5, 0.8],
    )
    assert mechanics.current_mobs(state)[:, 10, 10].tolist() == [True, True, True]
    assert mechanics.current_map(state)[:, 10, 10].tolist() == [7, 8, 9]
    assert mechanics.has_beaten_boss(state).tolist() == [False, True, False]


def test_defense_and_occupancy_reduce_only_over_their_element_axis() -> None:
    damage = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
    defense = torch.tensor([[0.0, 0.5, 0.75], [0.5, 0.2, 0.0]])
    assert mechanics.apply_defense(damage, defense).tolist() == pytest.approx(
        [3.0, 12.5],
    )

    state = _state(num_envs=3)
    state.mob_map[0, 0, 10, 11] = True
    positions = torch.tensor([[10, 11], [10, 10], [10, 10]])
    assert mechanics.is_occupied(state, positions).tolist() == [True, True, True]
    assert (
        mechanics.in_bounds(torch.tensor([[0, 1], [1, 0], [1, 1]])).tolist()
        == [True] * 3
    )


def test_adjacent_block_search_covers_each_offset_and_enforces_bounds() -> None:
    state = _state(num_envs=3)
    state.player_position[:] = torch.tensor([[0, 0], [10, 10], [10, 10]])
    state.map[0, 0, 0, 1] = int(BlockType.CRAFTING_TABLE)
    state.map[1, 0, 9, 9] = int(BlockType.CRAFTING_TABLE)
    state.map[2, 0, 10, 10] = int(BlockType.CRAFTING_TABLE)

    assert mechanics.is_near_block(
        state,
        int(BlockType.CRAFTING_TABLE),
    ).tolist() == [True, True, False]


def test_walk_collision_predicates_distinguish_solid_water_lava_and_player() -> None:
    state = _state(num_envs=3)
    state.map[0, 0, 5, 5] = int(BlockType.WATER)
    state.map[1, 0, 5, 5] = int(BlockType.LAVA)
    state.map[2, 0, 5, 5] = int(BlockType.STONE)
    positions = torch.tensor([[5, 5], [5, 5], [5, 5]])
    collides = torch.tensor(
        [[False, True, False], [False, False, True], [True, False, False]],
    )

    assert mechanics.can_walk_on(state, positions, collides).tolist() == [False] * 3
    assert mechanics.can_walk_on(
        state,
        positions,
        torch.zeros_like(collides),
    ).tolist() == [True] * 2 + [False]


def test_mechanics_passes_state_device_to_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=2)
    batch_devices: list[torch.device | None] = []
    table_devices: list[tuple[torch.Tensor, torch.device | None]] = []
    zero_devices: list[torch.device | None] = []
    collides = torch.zeros(2, 3, dtype=torch.bool)
    can_unlock = torch.zeros(2, dtype=torch.bool)
    batch_rows = indexing.batch_rows
    on_device = constants.on_device
    zeros = torch.zeros

    def record_batch_rows(envs: int, device: torch.device) -> torch.Tensor:
        batch_devices.append(device)
        return batch_rows(envs, device)

    def record_on_device(
        table: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        table_devices.append((table, device))
        return on_device(table, device)

    def record_zeros(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        assert isinstance(device, torch.device)
        zero_devices.append(device)
        return zeros(size, dtype=dtype, device=device)

    monkeypatch.setattr(mechanics, "batch_rows", record_batch_rows)
    monkeypatch.setattr(constants, "on_device", record_on_device)
    monkeypatch.setattr(torch, "zeros", record_zeros)

    mechanics._on_level(state.map, state.player_level)
    mechanics.player_damage(state)
    mechanics.can_walk_on(
        state,
        torch.tensor([[10, 11], [11, 10]]),
        collides,
    )
    mechanics.in_bounds(torch.tensor([[0, 1], [1, 0]]))
    mechanics.is_near_block(state, int(BlockType.CRAFTING_TABLE))
    mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[10, 11], [11, 10]]),
        damage=torch.ones(2, 3),
        mob_class=1,
        can_unlock=can_unlock,
    )

    assert batch_devices
    assert all(device == state.device for device in batch_devices)
    assert {id(table) for table, device in table_devices if device == state.device} >= {
        id(constants.SWORD_DAMAGE),
        id(constants.SOLID_BLOCK),
        id(constants.CLOSE_BLOCKS),
        id(constants.MOB_DEFENSE),
        id(constants.MOB_ACHIEVEMENT),
    }
    assert all(
        device
        == (torch.device("cpu") if table is constants.MAP_EXTENT else state.device)
        for table, device in table_devices
    )
    assert zero_devices == [state.device]


def test_attack_uses_level_target_and_updates_only_its_health_and_achievement() -> None:
    state = _state(num_envs=3)
    levels = torch.tensor([0, 1, 2])
    slots = torch.tensor([1, 0, 1])
    rows = torch.arange(3)
    state.player_level[:] = levels
    state.melee_mobs.mask[rows, levels, slots] = True
    state.melee_mobs.health[rows, levels, slots] = torch.tensor([2.0, 3.0, 4.0])
    state.melee_mobs.position[rows, levels, slots] = torch.tensor(
        [[5, 6]] * 3,
        dtype=torch.int32,
    )
    state.melee_mobs.type_id[rows, levels, slots] = torch.tensor(
        [0, 1, 2],
        dtype=torch.int32,
    )

    mobs, killed, struck, achievements = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 6], [5, 6], [8, 8]]),
        damage=torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        mob_class=1,
        can_unlock=torch.tensor([True, True, True]),
    )

    assert struck.tolist() == [True, True, False]
    assert killed.tolist() == [True, False, False]
    assert mobs.health[rows, levels, slots].tolist() == [0.0, 2.0, 4.0]
    assert mobs.mask[rows, levels, slots].tolist() == [False, True, True]
    assert achievements[0].any()
    assert not achievements[1:].any()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
