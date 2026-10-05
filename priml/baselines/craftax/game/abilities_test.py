"""Tests for potions, spells, enchanting, and levelling."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import abilities, constants
from priml.baselines.craftax.game.constants import (
    Achievement,
    Action,
    BlockType,
    ProjectileType,
)
from priml.baselines.craftax.game.indexing import batch_rows
from priml.baselines.craftax.game.state import EnvState, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.player_direction[:] = int(Action.RIGHT)
    state.map[:] = int(BlockType.GRASS)
    state.player_health[:] = 5.0
    state.player_mana[:] = 9
    state.player_energy[:] = 5
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    # Identity mapping, so colour zero heals and colour one poisons.
    state.potion_mapping[:] = torch.arange(6, dtype=torch.int32)
    return state


def _act(action: Action, num_envs: int = 2) -> Tensor:
    return torch.full((num_envs,), int(action), dtype=torch.int32)


def test_a_potion_applies_the_effect_its_colour_maps_to() -> None:
    state = _state()
    state.inventory.potions[:, 0] = 1
    healed = abilities.drink_potion(state, _act(Action.DRINK_POTION_RED))
    assert healed.player_health.tolist() == [13.0, 13.0]
    assert healed.inventory.potions[:, 0].tolist() == [0, 0]
    assert healed.achievements[:, int(Achievement.DRINK_POTION)].tolist() == [
        True,
        True,
    ]


def test_ability_tensor_factories_use_the_state_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potion_state = _state(num_envs=3)
    potion_state.inventory.potions[:, 0] = 1
    arrow_state = _state(num_envs=3)
    arrow_state.inventory.bow[:] = 1
    arrow_state.inventory.arrows[:] = 1
    spell_state = _state(num_envs=3)
    spell_state.learned_spells[:, 0] = True
    book_state = _state(num_envs=3)
    book_state.inventory.books[:] = 1
    enchant_state = _state(num_envs=3)
    enchant_state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    enchant_state.inventory.sword[:] = 1
    enchant_state.inventory.ruby[:] = 1
    plant_state = _state(num_envs=3)
    plant_state.growing_plants_mask[:, 0] = True
    plant_state.growing_plants_positions[:, 0] = torch.tensor([10, 12])
    level_state = _state(num_envs=3)
    level_state.player_xp[:] = 1
    actions = [_act(action, num_envs=3) for action in Action]
    book_generator = torch.Generator().manual_seed(0)
    enchant_generator = torch.Generator().manual_seed(1)
    factory_calls: list[tuple[str, dict[str, object]]] = []
    on_device_devices: list[torch.device] = []
    multinomial_generators: list[torch.Generator | None] = []
    zeros = torch.zeros
    full = torch.full
    arange = torch.arange
    multinomial = torch.multinomial
    on_device = constants.on_device
    original_batch_rows = batch_rows
    batch_row_devices: list[object] = []

    def tracked_batch_rows(num_envs: int, device: torch.device) -> Tensor:
        batch_row_devices.append(device)
        return original_batch_rows(num_envs, device)

    def tracked_zeros(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_calls.append(("zeros", {"dtype": dtype, "device": device}))
        return zeros(size, dtype=dtype, device=device)

    def tracked_full(
        size: tuple[int, ...],
        fill_value: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_calls.append(("full", {"dtype": dtype, "device": device}))
        return full(size, fill_value, dtype=dtype, device=device)

    def tracked_arange(
        end: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_calls.append(("arange", {"device": device}))
        return arange(end, device=device)

    def tracked_on_device(table: Tensor, device: torch.device) -> Tensor:
        on_device_devices.append(device)
        return on_device(table, device)

    def tracked_multinomial(
        input: Tensor,
        num_samples: int,
        *,
        replacement: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        multinomial_generators.append(generator)
        return multinomial(
            input,
            num_samples,
            replacement=replacement,
            generator=generator,
        )

    monkeypatch.setattr(torch, "zeros", tracked_zeros)
    monkeypatch.setattr(torch, "full", tracked_full)
    monkeypatch.setattr(torch, "arange", tracked_arange)
    monkeypatch.setattr(torch, "multinomial", tracked_multinomial)
    monkeypatch.setattr(constants, "on_device", tracked_on_device)
    monkeypatch.setattr(
        "priml.baselines.craftax.game.abilities.batch_rows",
        tracked_batch_rows,
    )

    abilities.drink_potion(potion_state, actions[int(Action.DRINK_POTION_RED)])
    abilities.shoot_arrow(arrow_state, actions[int(Action.SHOOT_ARROW)])
    abilities.cast_spell(spell_state, actions[int(Action.CAST_FIREBALL)])
    abilities.read_book(
        book_state,
        actions[int(Action.READ_BOOK)],
        generator=book_generator,
    )
    abilities.enchant(
        enchant_state,
        actions[int(Action.ENCHANT_SWORD)],
        generator=enchant_generator,
    )
    abilities.grow_plants(plant_state)
    abilities.level_up(level_state, actions[int(Action.LEVEL_UP_STRENGTH)])

    assert factory_calls
    assert all(
        kwargs.get("device") == potion_state.device for _, kwargs in factory_calls
    )
    assert [
        kwargs.get("dtype") for name, kwargs in factory_calls if name == "zeros"
    ] == [
        torch.bool,
        torch.long,
    ]
    assert on_device_devices
    assert all(device == potion_state.device for device in on_device_devices)
    assert batch_row_devices
    assert all(device == potion_state.device for device in batch_row_devices)
    assert multinomial_generators == [book_generator, enchant_generator]


def test_each_potion_colour_uses_its_own_mapping_entry() -> None:
    state = _state(num_envs=7)
    state.inventory.potions[:6] = 1
    actions = torch.tensor(
        [
            int(Action.DRINK_POTION_RED),
            int(Action.DRINK_POTION_GREEN),
            int(Action.DRINK_POTION_BLUE),
            int(Action.DRINK_POTION_PINK),
            int(Action.DRINK_POTION_CYAN),
            int(Action.DRINK_POTION_YELLOW),
            int(Action.NOOP),
        ],
        dtype=torch.int32,
    )

    drunk = abilities.drink_potion(state, actions)

    assert drunk.player_health.tolist() == [13.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    assert drunk.player_mana.tolist() == [9, 9, 17, 6, 9, 9, 9]
    assert drunk.player_energy.tolist() == [5, 5, 5, 5, 13, 2, 5]
    assert drunk.inventory.potions.tolist() == [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    assert drunk.achievements[:, int(Achievement.DRINK_POTION)].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]


def test_the_same_colour_can_poison_under_a_different_mapping() -> None:
    # This shuffling is the game's one genuinely hidden variable.
    state = _state()
    state.inventory.potions[:, 0] = 1
    state.potion_mapping[:, 0] = 1
    hurt = abilities.drink_potion(state, _act(Action.DRINK_POTION_RED))
    assert hurt.player_health.tolist() == [2.0, 2.0]


def test_potion_actions_select_distinct_colours_per_environment() -> None:
    state = _state()
    state.inventory.potions[:, 0] = 1
    state.inventory.potions[:, 2] = 1
    state.potion_mapping[:, 0] = 0
    state.potion_mapping[:, 2] = 4
    action = torch.tensor(
        [int(Action.DRINK_POTION_RED), int(Action.DRINK_POTION_BLUE)],
        dtype=torch.int32,
    )

    drunk = abilities.drink_potion(state, action)

    assert drunk.player_health.tolist() == [13.0, 5.0]
    assert drunk.player_energy.tolist() == [5, 13]
    assert drunk.inventory.potions[:, [0, 2]].tolist() == [[0, 1], [1, 0]]
    assert drunk.achievements[:, int(Achievement.DRINK_POTION)].tolist() == [
        True,
        True,
    ]


def test_drinking_a_potion_you_do_not_have_does_nothing() -> None:
    state = abilities.drink_potion(_state(), _act(Action.DRINK_POTION_RED))
    assert state.player_health.tolist() == [5.0, 5.0]


def test_potion_delta_preserves_the_state_tensor_dtype() -> None:
    state = _state()
    state.player_health = torch.tensor([5, 5], dtype=torch.int8)
    state.inventory.potions[:, 0] = 1

    drunk = abilities.drink_potion(state, _act(Action.DRINK_POTION_RED))

    assert drunk.player_health.tolist() == [13, 13]
    assert drunk.player_health.dtype == torch.int8


@pytest.mark.parametrize(
    ("effect", "field", "expected"),
    [(2, "player_mana", 17), (3, "player_mana", 6), (4, "player_energy", 13)],
)
def test_potions_reach_mana_and_energy_too(
    effect: int,
    field: str,
    expected: int,
) -> None:
    state = _state()
    state.inventory.potions[:, 0] = 1
    state.potion_mapping[:, 0] = effect
    drunk = abilities.drink_potion(state, _act(Action.DRINK_POTION_RED))
    values: object = getattr(drunk, field)  # pyright: ignore[reportAny] -- field names select Tensor state attributes.
    assert isinstance(values, Tensor)
    values_list = [int(value) for value in values]
    assert values_list == [expected, expected]


def test_shooting_an_arrow_needs_a_bow_and_spends_one() -> None:
    unarmed = _state()
    unarmed.inventory.arrows[:] = 3
    assert not abilities.shoot_arrow(
        unarmed,
        _act(Action.SHOOT_ARROW),
    ).player_projectiles.mask.any()

    armed = _state()
    armed.inventory.bow[:] = 1
    armed.inventory.arrows[:] = 1
    fired = abilities.shoot_arrow(armed, _act(Action.SHOOT_ARROW))
    assert fired.inventory.arrows.tolist() == [0, 0]
    assert fired.player_projectiles.mask[0, 0].tolist() == [True, False, False]
    assert fired.player_projectiles.type_id[0, 0, 0].item() == int(
        ProjectileType.ARROW2,
    )
    assert fired.player_projectiles.position[0, 0, 0].tolist() == [10, 10]
    assert fired.player_projectile_directions[0, 0, 0].tolist() == [0, 1]
    assert fired.achievements[:, int(Achievement.FIRE_BOW)].tolist() == [True, True]


def test_arrows_use_each_environments_first_free_slot() -> None:
    state = _state()
    state.inventory.bow[:] = 1
    state.inventory.arrows[:] = 2
    state.player_projectiles.mask[0, 0, 0] = True
    state.player_projectiles.type_id[0, 0, 0] = int(ProjectileType.FIREBALL)
    state.player_projectiles.position[0, 0, 0] = torch.tensor([2, 3])

    fired = abilities.shoot_arrow(state, _act(Action.SHOOT_ARROW))

    assert fired.player_projectiles.mask[:, 0].tolist() == [
        [True, True, False],
        [True, False, False],
    ]
    assert fired.player_projectiles.type_id[:, 0, :2].tolist() == [
        [int(ProjectileType.FIREBALL), int(ProjectileType.ARROW2)],
        [int(ProjectileType.ARROW2), 0],
    ]
    assert fired.player_projectiles.position[0, 0, 0].tolist() == [2, 3]
    assert fired.inventory.arrows.tolist() == [1, 1]


def test_arrow_slot_selection_uses_each_environment_level() -> None:
    state = _state()
    state.player_level[:] = torch.tensor([0, 1])
    state.inventory.bow[:] = 1
    state.inventory.arrows[:] = 2
    state.player_projectiles.mask[0, 0] = True
    state.player_projectiles.mask[1, 1, 0] = True

    fired = abilities.shoot_arrow(state, _act(Action.SHOOT_ARROW))

    assert fired.inventory.arrows.tolist() == [2, 1]
    assert fired.player_projectiles.mask[0, 0].tolist() == [True, True, True]
    assert fired.player_projectiles.mask[1, 1].tolist() == [True, True, False]
    assert fired.player_projectiles.type_id[1, 1, 1].item() == int(
        ProjectileType.ARROW2,
    )


def test_a_full_projectile_pool_rejects_arrows() -> None:
    # Upstream gates costs and achievements on a free slot (game_logic.py:2506-2525).
    state = _state()
    state.inventory.bow[:] = 1
    state.inventory.arrows[:] = 3
    state.player_projectiles.mask[:, 0] = True
    rejected = abilities.shoot_arrow(state, _act(Action.SHOOT_ARROW))
    assert rejected.inventory.arrows.tolist() == [3, 3]
    assert rejected.achievements[:, int(Achievement.FIRE_BOW)].tolist() == [
        False,
        False,
    ]


def test_a_full_projectile_pool_only_blocks_its_own_environment() -> None:
    state = _state()
    state.inventory.bow[:] = 1
    state.inventory.arrows[:] = 1
    state.player_projectiles.mask[0, 0] = True

    fired = abilities.shoot_arrow(state, _act(Action.SHOOT_ARROW))

    assert fired.inventory.arrows.tolist() == [1, 0]
    assert fired.player_projectiles.mask[0, 0].tolist() == [True, True, True]
    assert fired.player_projectiles.mask[1, 0].tolist() == [True, False, False]
    assert fired.achievements[:, int(Achievement.FIRE_BOW)].tolist() == [False, True]


def test_a_full_projectile_pool_rejects_spells() -> None:
    # Upstream gates spell costs and achievements on free slots (game_logic.py:2541-2590).
    state = _state()
    state.learned_spells[:, 0] = True
    state.player_projectiles.mask[:, 0] = True
    rejected = abilities.cast_spell(state, _act(Action.CAST_FIREBALL))
    assert rejected.player_mana.tolist() == [9, 9]
    assert rejected.achievements[:, int(Achievement.CAST_FIREBALL)].tolist() == [
        False,
        False,
    ]


def test_spell_slot_availability_is_per_environment() -> None:
    state = _state(num_envs=3)
    state.player_level[:] = torch.tensor([0, 1, 0])
    state.learned_spells[:, 0] = True
    state.player_projectiles.mask[0, 0] = True
    state.player_projectiles.mask[1, 1, 0] = True
    actions = torch.full((3,), int(Action.CAST_FIREBALL), dtype=torch.int32)

    cast = abilities.cast_spell(state, actions)

    assert cast.player_mana.tolist() == [9, 7, 7]
    assert cast.player_projectiles.mask[0, 0].tolist() == [True, True, True]
    assert cast.player_projectiles.mask[1, 1].tolist() == [True, True, False]
    assert cast.player_projectiles.mask[2, 0, 0].item()
    assert cast.achievements[:, int(Achievement.CAST_FIREBALL)].tolist() == [
        False,
        True,
        True,
    ]


def test_a_spell_must_be_learned_before_it_can_be_cast() -> None:
    unlearned = abilities.cast_spell(_state(), _act(Action.CAST_FIREBALL))
    assert not unlearned.player_projectiles.mask.any()
    assert unlearned.player_mana.tolist() == [9, 9]

    learned = _state()
    learned.learned_spells[:, 0] = True
    cast = abilities.cast_spell(learned, _act(Action.CAST_FIREBALL))
    assert bool(cast.player_projectiles.mask[0, 0].any())
    assert cast.player_mana.tolist() == [7, 7]
    assert cast.achievements[:, int(Achievement.CAST_FIREBALL)].tolist() == [True, True]


def test_spells_accept_exact_mana_and_select_each_action_per_environment() -> None:
    state = _state(num_envs=3)
    state.player_mana[:2] = 2
    state.learned_spells[:] = True
    actions = torch.tensor(
        [int(Action.CAST_FIREBALL), int(Action.CAST_ICEBALL), int(Action.NOOP)],
        dtype=torch.int32,
    )

    cast = abilities.cast_spell(state, actions)

    assert cast.player_mana.tolist() == [0, 0, 9]
    assert cast.player_projectiles.mask[:, 0, 0].tolist() == [True, True, False]
    assert cast.player_projectiles.type_id[:, 0, 0].tolist() == [
        int(ProjectileType.FIREBALL),
        int(ProjectileType.ICEBALL),
        0,
    ]
    assert cast.achievements[:, int(Achievement.CAST_FIREBALL)].tolist() == [
        True,
        False,
        False,
    ]
    assert cast.achievements[:, int(Achievement.CAST_ICEBALL)].tolist() == [
        False,
        True,
        False,
    ]


def test_casting_without_mana_fails() -> None:
    state = _state()
    state.learned_spells[:, 0] = True
    state.player_mana[:] = 1
    cast = abilities.cast_spell(state, _act(Action.CAST_FIREBALL))
    assert not cast.player_projectiles.mask.any()


def test_the_spell_kind_follows_the_action() -> None:
    state = _state()
    state.learned_spells[:] = True
    ice = abilities.cast_spell(state, _act(Action.CAST_ICEBALL))
    assert int(ice.player_projectiles.type_id[0, 0, 0]) == int(ProjectileType.ICEBALL)


def test_a_book_teaches_an_unknown_spell() -> None:
    state = _state()
    state.inventory.books[:] = 1
    read = abilities.read_book(
        state,
        _act(Action.READ_BOOK),
        generator=torch.Generator().manual_seed(0),
    )
    assert bool(read.learned_spells.any())
    assert read.inventory.books.tolist() == [0, 0]


def test_reading_without_a_book_preserves_spells_and_inventory() -> None:
    state = _state()
    state.learned_spells[:, 0] = True
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state()

    abilities.read_book(state, _act(Action.READ_BOOK), generator=generator)

    assert state.learned_spells.tolist() == [[True, False], [True, False]]
    assert state.inventory.books.tolist() == [0, 0]
    assert torch.equal(generator.get_state(), before) is False
    assert not state.achievements[:, int(Achievement.LEARN_FIREBALL)].any()
    assert not state.achievements[:, int(Achievement.LEARN_ICEBALL)].any()


def test_a_book_teaches_the_spell_still_unknown() -> None:
    state = _state()
    state.inventory.books[:] = 1
    state.learned_spells[:, 0] = True
    read = abilities.read_book(
        state,
        _act(Action.READ_BOOK),
        generator=torch.Generator().manual_seed(0),
    )
    assert read.learned_spells[:, 1].tolist() == [True, True]


def test_book_action_is_applied_per_environment() -> None:
    state = _state(num_envs=3)
    state.inventory.books[:2] = 1
    state.learned_spells[0, 0] = True
    state.learned_spells[1, :] = True
    actions = torch.tensor(
        [int(Action.READ_BOOK), int(Action.READ_BOOK), int(Action.READ_BOOK)],
        dtype=torch.int32,
    )

    read = abilities.read_book(
        state,
        actions,
        generator=torch.Generator().manual_seed(0),
    )

    assert read.learned_spells.tolist() == [
        [True, True],
        [True, True],
        [False, False],
    ]
    assert read.inventory.books.tolist() == [0, 0, 0]
    assert read.achievements[:, int(Achievement.LEARN_FIREBALL)].tolist() == [
        False,
        True,
        False,
    ]
    assert read.achievements[:, int(Achievement.LEARN_ICEBALL)].tolist() == [
        True,
        False,
        False,
    ]


def test_book_action_is_gated_per_environment() -> None:
    state = _state(num_envs=3)
    state.inventory.books[:] = 1
    actions = torch.tensor(
        [int(Action.READ_BOOK), int(Action.NOOP), int(Action.NOOP)],
        dtype=torch.int32,
    )

    read = abilities.read_book(
        state,
        actions,
        generator=torch.Generator().manual_seed(0),
    )

    assert read.inventory.books.tolist() == [0, 1, 1]
    assert read.learned_spells[0].tolist() != [False, False]
    assert read.learned_spells[1].tolist() == [False, False]
    assert read.learned_spells[2].tolist() == [False, False]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_reading_with_both_spells_known_spends_the_book_on_first_spell(
    seed: int,
) -> None:
    """Upstream's zero-probability choice falls to index zero (game_logic.py:2697-2713)."""
    state = _state(num_envs=1)
    state.inventory.books[:] = 1
    state.learned_spells[:] = True

    read = abilities.read_book(
        state,
        _act(Action.READ_BOOK, num_envs=1),
        generator=torch.Generator().manual_seed(seed),
    )

    assert read.inventory.books.tolist() == [0]
    assert read.achievements[0, int(Achievement.LEARN_FIREBALL)]
    assert not read.achievements[0, int(Achievement.LEARN_ICEBALL)]


def test_reading_draws_the_spell_even_when_nothing_is_left_to_learn() -> None:
    # As the reference does. Skipping the draw would need the host to read the
    # batch first, a stall that also keeps a step out of a CUDA graph.
    state = _state()
    state.inventory.books[:] = 1
    state.learned_spells[:] = True
    generator = torch.Generator().manual_seed(0)
    before = generator.get_state()
    abilities.read_book(state, _act(Action.READ_BOOK), generator=generator)
    assert not torch.equal(generator.get_state(), before)


def test_enchanting_binds_the_table_element_and_spends_its_gem() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.sword[:] = 1
    state.inventory.ruby[:] = 1
    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_SWORD),
        generator=torch.Generator().manual_seed(0),
    )
    assert enchanted.sword_enchantment.tolist() == [1, 1]
    assert enchanted.inventory.ruby.tolist() == [0, 0]
    assert enchanted.player_mana.tolist() == [0, 0]
    assert enchanted.achievements[:, int(Achievement.ENCHANT_SWORD)].tolist() == [
        True,
        True,
    ]


def test_enchanting_a_bow_on_an_ice_table_spends_only_sapphires() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_ICE)
    state.inventory.bow[:] = 1
    state.inventory.ruby[:] = 2
    state.inventory.sapphire[:] = 1

    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_BOW),
        generator=torch.Generator().manual_seed(0),
    )

    assert enchanted.bow_enchantment.tolist() == [2, 2]
    assert enchanted.sword_enchantment.tolist() == [0, 0]
    assert enchanted.inventory.ruby.tolist() == [2, 2]
    assert enchanted.inventory.sapphire.tolist() == [0, 0]
    assert enchanted.player_mana.tolist() == [0, 0]


def test_enchanting_a_non_enchant_action_preserves_resources() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.sword[:] = 1
    state.inventory.ruby[:] = 1

    abilities.enchant(
        state,
        _act(Action.NOOP),
        generator=torch.Generator().manual_seed(0),
    )

    assert state.sword_enchantment.tolist() == [0, 0]
    assert state.inventory.ruby.tolist() == [1, 1]
    assert state.player_mana.tolist() == [9, 9]
    assert not state.achievements[:, int(Achievement.ENCHANT_SWORD)].any()


def test_an_ice_table_spends_sapphires_instead() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_ICE)
    state.inventory.sword[:] = 2
    state.inventory.sapphire[:] = 2
    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_SWORD),
        generator=torch.Generator().manual_seed(0),
    )
    assert enchanted.sword_enchantment.tolist() == [2, 2]
    assert enchanted.inventory.sapphire.tolist() == [1, 1]


def test_enchanting_requires_the_selected_equipment() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.ruby[:] = 1

    abilities.enchant(
        state,
        _act(Action.ENCHANT_SWORD),
        generator=torch.Generator().manual_seed(0),
    )

    assert state.sword_enchantment.tolist() == [0, 0]
    assert state.inventory.ruby.tolist() == [1, 1]
    assert state.player_mana.tolist() == [9, 9]
    assert not state.achievements[:, int(Achievement.ENCHANT_SWORD)].any()


def test_non_enchant_actions_do_not_enchant_available_equipment() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.bow[:] = 1
    state.inventory.armour[:, 0] = 1
    state.inventory.ruby[:] = 1

    abilities.enchant(
        state,
        _act(Action.NOOP),
        generator=torch.Generator().manual_seed(0),
    )

    assert state.bow_enchantment.tolist() == [0, 0]
    assert state.armour_enchantments.tolist() == [[0, 0, 0, 0]] * 2
    assert state.inventory.ruby.tolist() == [1, 1]
    assert state.player_mana.tolist() == [9, 9]


def test_armour_enchantment_requires_armour_per_environment() -> None:
    state = _state(num_envs=2)
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.armour[0, 0] = 1
    state.inventory.ruby[:] = 1
    actions = _act(Action.ENCHANT_ARMOUR)

    enchanted = abilities.enchant(
        state,
        actions,
        generator=torch.Generator().manual_seed(0),
    )

    assert int((enchanted.armour_enchantments[0] == 1).sum()) == 1
    assert enchanted.armour_enchantments[1].tolist() == [0, 0, 0, 0]
    assert enchanted.inventory.ruby.tolist() == [0, 1]
    assert enchanted.player_mana.tolist() == [0, 9]


def test_enchanting_a_bow_requires_a_bow() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.ruby[:] = 1

    abilities.enchant(
        state,
        _act(Action.ENCHANT_BOW),
        generator=torch.Generator().manual_seed(0),
    )

    assert state.bow_enchantment.tolist() == [0, 0]
    assert state.inventory.ruby.tolist() == [1, 1]
    assert state.player_mana.tolist() == [9, 9]


def test_non_enchant_action_on_ice_table_does_not_spend_sapphires() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_ICE)
    state.inventory.sapphire[:] = 1

    abilities.enchant(
        state,
        _act(Action.NOOP),
        generator=torch.Generator().manual_seed(0),
    )

    assert state.inventory.sapphire.tolist() == [1, 1]
    assert state.player_mana.tolist() == [9, 9]
    assert state.sword_enchantment.tolist() == [0, 0]


def test_enchanting_needs_a_table_a_gem_and_mana() -> None:
    for missing in ("table", "gem", "mana"):
        state = _state()
        state.inventory.sword[:] = 2
        state.inventory.ruby[:] = 2
        if missing != "table":
            state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
        if missing == "gem":
            state.inventory.ruby[:] = 0
        if missing == "mana":
            state.player_mana[:] = 3
        enchanted = abilities.enchant(
            state,
            _act(Action.ENCHANT_SWORD),
            generator=torch.Generator().manual_seed(0),
        )
        assert enchanted.sword_enchantment.tolist() == [0, 0], missing


def test_enchanting_armour_fills_a_bare_piece() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.armour[:] = 1
    state.inventory.ruby[:] = 2
    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_ARMOUR),
        generator=torch.Generator().manual_seed(0),
    )
    assert int((enchanted.armour_enchantments == 1).sum(-1)[0]) == 1
    assert enchanted.achievements[:, int(Achievement.ENCHANT_ARMOUR)].tolist() == [
        True,
        True,
    ]


def test_armour_enchant_prefers_bare_piece_then_replaces_opposite_element() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.armour[:, 0] = 1
    state.inventory.ruby[:] = 1
    state.armour_enchantments[0] = torch.tensor([1, 0, 1, 1])
    state.armour_enchantments[1] = torch.tensor([1, 2, 1, 1])

    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_ARMOUR),
        generator=torch.Generator().manual_seed(0),
    )

    assert enchanted.armour_enchantments.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    assert enchanted.inventory.ruby.tolist() == [0, 0]
    assert enchanted.player_mana.tolist() == [0, 0]


def test_armour_enchant_falls_back_when_every_piece_has_same_element() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.armour[:, 0] = 1
    state.inventory.ruby[:] = 1
    state.armour_enchantments[:] = torch.tensor(
        [[1, 0, 1, 1], [1, 1, 1, 1]],
    )

    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_ARMOUR),
        generator=torch.Generator().manual_seed(0),
    )

    assert enchanted.armour_enchantments.tolist() == [[1, 1, 1, 1]] * 2
    assert enchanted.inventory.ruby.tolist() == [0, 0]
    assert enchanted.player_mana.tolist() == [0, 0]


def test_levelling_spends_experience_and_stops_at_the_cap() -> None:
    state = _state()
    state.player_xp[:] = 2
    raised = abilities.level_up(state, _act(Action.LEVEL_UP_STRENGTH))
    assert raised.player_strength.tolist() == [2, 2]
    assert raised.player_xp.tolist() == [1, 1]

    capped = _state()
    capped.player_xp[:] = 5
    capped.player_strength[:] = constants.MAX_ATTRIBUTE
    held = abilities.level_up(capped, _act(Action.LEVEL_UP_STRENGTH))
    assert held.player_strength.tolist() == [constants.MAX_ATTRIBUTE] * 2
    assert held.player_xp.tolist() == [5, 5]


def test_levelling_intelligence_spends_one_point_per_eligible_environment() -> None:
    state = _state()
    state.player_xp[:] = torch.tensor([1, 0])
    state.player_intelligence[:] = torch.tensor([1, 2])
    action = torch.tensor(
        [
            int(Action.LEVEL_UP_INTELLIGENCE),
            int(Action.LEVEL_UP_INTELLIGENCE),
        ],
        dtype=torch.int32,
    )

    raised = abilities.level_up(state, action)

    assert raised.player_intelligence.tolist() == [2, 2]
    assert raised.player_xp.tolist() == [0, 0]


def test_levelling_without_experience_does_nothing() -> None:
    raised = abilities.level_up(_state(), _act(Action.LEVEL_UP_DEXTERITY))
    assert raised.player_dexterity.tolist() == [1, 1]


def test_a_sown_plant_ripens_once_it_is_old_enough() -> None:
    # Upstream update_plants uses age >= 600 (game_logic.py:1979-1987).
    state = _state()
    state.growing_plants_mask[:, 0] = True
    state.growing_plants_positions[:, 0] = torch.tensor([10, 12], dtype=torch.int32)
    state.growing_plants_age[:, 0] = 599
    grown = abilities.grow_plants(state)
    assert grown.growing_plants_age[:, 0].tolist() == [600, 600]
    assert grown.map[0, 0, 10, 12].item() == int(BlockType.RIPE_PLANT)

    inactive = _state()
    inactive.growing_plants_age[:, 0] = 10
    assert abilities.grow_plants(inactive).growing_plants_age[:, 0].tolist() == [0, 0]


def test_plants_age_and_ripen_independently_by_slot() -> None:
    state = _state()
    state.growing_plants_mask[:, :3] = torch.tensor(
        [[True, True, False], [False, True, True]],
    )
    state.growing_plants_age[:, :3] = torch.tensor(
        [[599, 600, 17], [5, 598, 599]],
    )
    state.growing_plants_positions[:, :2] = torch.tensor(
        [[[10, 12], [10, 13]], [[10, 12], [10, 13]]],
        dtype=torch.int32,
    )
    state.growing_plants_positions[:, 2] = torch.tensor(
        [[10, 14], [10, 14]],
        dtype=torch.int32,
    )

    grown = abilities.grow_plants(state)

    assert grown.growing_plants_age[:, :3].tolist() == [
        [600, 601, 0],
        [0, 599, 600],
    ]
    assert grown.map[:, 0, 10, 12].tolist() == [
        int(BlockType.RIPE_PLANT),
        int(BlockType.GRASS),
    ]
    assert grown.map[:, 0, 10, 13].tolist() == [
        int(BlockType.RIPE_PLANT),
        int(BlockType.GRASS),
    ]
    assert grown.map[:, 0, 10, 14].tolist() == [
        int(BlockType.GRASS),
        int(BlockType.RIPE_PLANT),
    ]


def test_a_young_plant_is_not_yet_ripe() -> None:
    state = _state()
    state.growing_plants_mask[:, 0] = True
    state.growing_plants_positions[:, 0] = torch.tensor([10, 12], dtype=torch.int32)
    grown = abilities.grow_plants(state)
    assert grown.map[0, 0, 10, 12].item() == int(BlockType.GRASS)


def test_plants_ripen_on_the_surface_while_the_player_is_below() -> None:
    """Upstream ripens floor 0 whatever floor the player is on (game_logic.py:2007-2011)."""
    state = _state()
    state.player_level[:] = torch.tensor([0, 3], dtype=torch.int32)
    state.growing_plants_mask[:, 0] = True
    state.growing_plants_positions[:, 0] = torch.tensor([5, 5], dtype=torch.int32)
    state.growing_plants_age[:, 0] = 599
    state.map[:, 0, 5, 5] = int(BlockType.PLANT)
    grown = abilities.grow_plants(state)
    assert grown.map[:, 0, 5, 5].tolist() == [int(BlockType.RIPE_PLANT)] * 2
    assert grown.map[:, 3, 5, 5].tolist() == [int(BlockType.GRASS)] * 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
