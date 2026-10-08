"""Check the legal-action rules on decoded frames, by hand and against the game."""

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.world_model.rules import legal_actions
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.testing import played_frames
from priml.lib.codec import from_plain


def frame(*, item: int = 1, **values: int) -> tuple[Tensor, Tensor]:
    """Return a visible grass board and aux values: attributes 1, the rest 0.

    Args:
      item: Item field (``item + 1``) of the player's cell.
      **values: Auxiliary values by schema name.

    Returns:
      cells: Cell values, uint8 ``[99, 8]``.
      aux: Auxiliary values, int16 ``[51]``.

    """
    names = craftax_schema().scalar_names
    defaults = {"dexterity": 1, "strength": 1, "intelligence": 1, "health": 180}
    defaults |= {"food": 9, "drink": 9, "energy": 9}
    merged = defaults | values
    aux = torch.tensor([merged.get(name, 0) for name in names], dtype=torch.int16)
    cells = torch.zeros(99, 8, dtype=torch.uint8)
    cells[:, :3] = torch.tensor([2, 1, 1], dtype=torch.uint8)
    cells[49, 1] = item
    return cells, aux


def legal(**values: int) -> set[int]:
    """Return the legal actions of ``frame(**values)`` as a set."""
    return set(
        from_plain(legal_actions(*frame(**values)).nonzero()[:, 0].tolist(), list[int]),
    )


@pytest.mark.parametrize("world_seed", [3, 11, 29])
def test_frames_of_played_states_allow_what_the_games_mask_allows(
    world_seed: int,
) -> None:
    played = played_frames(world_seed=world_seed, decisions=200)
    got = legal_actions(played.cells, played.aux)
    # A dark player cell leaves both ladder actions undetermined, so allowed.
    dark = played.cells[:, 49, 2] == 0
    expected = played.masks.clone()
    expected[dark, 18:20] = True
    assert len(played.masks) > 20
    assert torch.equal(got, expected)


def test_empty_inventory_allows_movement_do_and_noop_only() -> None:
    # Full energy and health rule out sleep and rest: max need is 7 + 2 * 1.
    assert legal(energy=9, health=180) == {0, 1, 2, 3, 4, 5}


def test_sleeping_or_resting_allows_only_noop() -> None:
    assert legal(sleeping=1, wood=5) == {0}
    assert legal(resting=1, wood=5) == {0}


def test_needs_and_health_below_maximum_allow_sleep_and_rest() -> None:
    assert {6, 17} <= legal(energy=8, health=179)
    # Maximum energy is 7 + 2 * dexterity: 9 at dexterity 1, 11 at dexterity 2.
    assert 6 in legal(energy=10, dexterity=2)
    assert 6 not in legal(energy=11, dexterity=2)
    # Health is on the 0.05-HP grid: 180 is 9 HP, the maximum at strength 1.
    assert 17 not in legal(health=180)
    assert 17 in legal(health=199, strength=2)


def test_crafting_follows_inventory_and_tool_level() -> None:
    assert {8, 11, 14} <= legal(wood=2)
    assert 8 not in legal(wood=1)
    assert {7, 9, 12, 15, 25} <= legal(wood=1, stone=1)
    assert 11 not in legal(wood=1, pickaxe=1)
    iron = legal(wood=1, stone=1, iron=1, coal=1)
    assert {13, 16, 38} <= iron
    assert 38 not in legal(wood=1, coal=1, torches=99)
    assert 22 not in iron
    assert 22 in legal(iron=3, coal=3, armour_0=1, armour_1=2)
    assert 22 not in legal(
        iron=3,
        coal=3,
        armour_0=1,
        armour_1=1,
        armour_2=1,
        armour_3=1,
    )
    assert {20, 21, 23} <= legal(wood=1, diamond=3)
    assert 20 not in legal(wood=1, diamond=2)
    assert 25 not in legal(wood=1, stone=1, arrows=99)


def test_ladders_need_the_item_under_the_player() -> None:
    # The item field stores ``item + 1``: 3 is a down ladder, 4 an up ladder.
    assert 18 in legal(item=3, floor_clear=1)
    assert 18 not in legal(item=3)
    assert 18 not in legal(item=3, floor_clear=1, floor=8)
    assert 19 in legal(item=4, floor=1)
    assert 19 not in legal(item=4)
    assert 19 not in legal(item=3, floor=1)


def test_dark_player_cell_leaves_ladders_undetermined_and_allowed() -> None:
    cells, aux = frame(floor=1, floor_clear=1)
    cells[49] = 0
    mask = legal_actions(cells, aux)
    assert bool(mask[18])
    assert bool(mask[19])


def test_spells_potions_books_and_attributes() -> None:
    assert {26, 27} <= legal(learned_fireball=1, learned_iceball=1, mana=2)
    assert 26 not in legal(learned_fireball=1, mana=1)
    assert 24 in legal(bow=1, arrows=1)
    assert 24 not in legal(bow=1)
    assert 24 not in legal(arrows=1)
    assert legal(potion_pink=1) - legal() == {32}
    assert 35 in legal(books=1)
    enchant = legal(mana=9, ruby=1, sword=1, bow=1, armour_2=1)
    assert {36, 37, 42} <= enchant
    assert 42 not in legal(mana=9, ruby=1, sword=1, armour_2=1)
    assert 36 not in legal(mana=8, ruby=1, sword=1)
    assert 36 in legal(mana=9, sapphire=1, sword=1)
    assert {39, 40, 41} <= legal(xp=1)
    assert 39 not in legal(xp=1, dexterity=5)
    assert 28 in legal(torches=1)
    assert 10 in legal(sapling=1)


def test_mask_is_rank_agnostic() -> None:
    cells, aux = frame(wood=1)
    batch = legal_actions(cells.expand(2, 3, 99, 8), aux.expand(2, 3, 51))
    assert batch.shape == (2, 3, 43)
    assert torch.equal(batch[1, 2], legal_actions(cells, aux))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
