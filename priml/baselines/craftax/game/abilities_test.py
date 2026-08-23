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


def test_the_same_colour_can_poison_under_a_different_mapping() -> None:
    # This shuffling is the game's one genuinely hidden variable.
    state = _state()
    state.inventory.potions[:, 0] = 1
    state.potion_mapping[:, 0] = 1
    hurt = abilities.drink_potion(state, _act(Action.DRINK_POTION_RED))
    assert hurt.player_health.tolist() == [2.0, 2.0]


def test_drinking_a_potion_you_do_not_have_does_nothing() -> None:
    state = abilities.drink_potion(_state(), _act(Action.DRINK_POTION_RED))
    assert state.player_health.tolist() == [5.0, 5.0]


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
    assert getattr(drunk, field).tolist() == [expected, expected]


def test_shooting_an_arrow_needs_a_bow_and_spends_one() -> None:
    unarmed = _state()
    unarmed.inventory.arrows[:] = 3
    assert not abilities.shoot_arrow(
        unarmed,
        _act(Action.SHOOT_ARROW),
    ).player_projectiles.mask.any()

    armed = _state()
    armed.inventory.bow[:] = 1
    armed.inventory.arrows[:] = 3
    fired = abilities.shoot_arrow(armed, _act(Action.SHOOT_ARROW))
    assert fired.inventory.arrows.tolist() == [2, 2]
    assert bool(fired.player_projectiles.mask[0, 0].any())
    assert fired.achievements[:, int(Achievement.FIRE_BOW)].tolist() == [True, True]


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


def test_enchanting_binds_the_table_element_and_spends_its_gem() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.ENCHANTMENT_TABLE_FIRE)
    state.inventory.sword[:] = 2
    state.inventory.ruby[:] = 2
    enchanted = abilities.enchant(
        state,
        _act(Action.ENCHANT_SWORD),
        generator=torch.Generator().manual_seed(0),
    )
    assert enchanted.sword_enchantment.tolist() == [1, 1]
    assert enchanted.inventory.ruby.tolist() == [1, 1]
    assert enchanted.player_mana.tolist() == [0, 0]
    assert enchanted.achievements[:, int(Achievement.ENCHANT_SWORD)].tolist() == [
        True,
        True,
    ]


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


def test_levelling_without_experience_does_nothing() -> None:
    raised = abilities.level_up(_state(), _act(Action.LEVEL_UP_DEXTERITY))
    assert raised.player_dexterity.tolist() == [1, 1]


def test_a_sown_plant_ripens_once_it_is_old_enough() -> None:
    state = _state()
    state.growing_plants_mask[:, 0] = True
    state.growing_plants_positions[:, 0] = torch.tensor([10, 12], dtype=torch.int32)
    state.growing_plants_age[:, 0] = 600
    grown = abilities.grow_plants(state)
    assert grown.growing_plants_age[:, 0].tolist() == [601, 601]
    assert grown.map[0, 0, 10, 12].item() == int(BlockType.RIPE_PLANT)


def test_a_young_plant_is_not_yet_ripe() -> None:
    state = _state()
    state.growing_plants_mask[:, 0] = True
    state.growing_plants_positions[:, 0] = torch.tensor([10, 12], dtype=torch.int32)
    grown = abilities.grow_plants(state)
    assert grown.map[0, 0, 10, 12].item() == int(BlockType.GRASS)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
