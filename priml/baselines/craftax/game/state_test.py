"""Tests for the batched world state."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.game import constants
from priml.baselines.craftax.game.state import EnvState, empty_state


def _state(num_envs: int = 4) -> EnvState:
    return empty_state(num_envs=num_envs, device=torch.device("cpu"))


def test_every_field_carries_the_environment_axis() -> None:
    state = _state(num_envs=3)
    for name, value in state.state_dict().items():
        assert value.shape[0] == 3, name


def test_shapes_follow_the_declared_world_size() -> None:
    state = _state()
    levels, (rows, columns) = constants.NUM_LEVELS, constants.MAP_SIZE
    assert state.map.shape == (4, levels, rows, columns)
    assert state.item_map.shape == (4, levels, rows, columns)
    assert state.light_map.shape == (4, levels, rows, columns)
    assert state.mob_map.shape == (4, levels, rows, columns)
    assert state.down_ladders.shape == (4, levels, 2)
    assert state.up_ladders.shape == (4, levels, 2)
    assert state.chests_opened.shape == (4, levels)
    assert state.monsters_killed.shape == (4, levels)
    assert state.player_position.shape == (4, 2)
    assert state.inventory.armour.shape == (4, 4)
    assert state.inventory.potions.shape == (4, 6)
    assert state.melee_mobs.position.shape == (
        4,
        levels,
        constants.MAX_MELEE_MOBS,
        2,
    )
    assert state.passive_mobs.position.shape == (
        4,
        levels,
        constants.MAX_PASSIVE_MOBS,
        2,
    )
    assert state.ranged_mobs.position.shape == (
        4,
        levels,
        constants.MAX_RANGED_MOBS,
        2,
    )
    assert state.mob_projectiles.position.shape == (
        4,
        levels,
        constants.MAX_MOB_PROJECTILES,
        2,
    )
    assert state.player_projectiles.position.shape == (
        4,
        levels,
        constants.MAX_PLAYER_PROJECTILES,
        2,
    )
    assert state.mob_projectile_directions.shape == (
        4,
        levels,
        constants.MAX_MOB_PROJECTILES,
        2,
    )
    assert state.player_projectile_directions.shape == (
        4,
        levels,
        constants.MAX_PLAYER_PROJECTILES,
        2,
    )
    assert state.growing_plants_positions.shape == (
        4,
        constants.MAX_GROWING_PLANTS,
        2,
    )
    assert state.growing_plants_age.shape == (4, constants.MAX_GROWING_PLANTS)
    assert state.growing_plants_mask.shape == (4, constants.MAX_GROWING_PLANTS)
    assert state.potion_mapping.shape == (4, 6)
    assert state.learned_spells.shape == (4, 2)
    assert state.armour_enchantments.shape == (4, 4)
    assert state.achievements.shape == (4, len(constants.Achievement))
    assert state.num_envs == 4


def test_empty_state_allocates_every_field_with_its_dtype_and_device() -> None:
    state = empty_state(num_envs=2, device=torch.device("meta"))
    names = state.state_dict()
    bool_fields = {
        "mob_map",
        "chests_opened",
        "is_sleeping",
        "is_resting",
        "growing_plants_mask",
        "learned_spells",
        "achievements",
    }
    float_fields = {
        "light_map",
        "player_health",
        "player_recover",
        "player_hunger",
        "player_thirst",
        "player_fatigue",
        "player_recover_mana",
        "light_level",
        "melee_mobs.health",
        "passive_mobs.health",
        "ranged_mobs.health",
        "mob_projectiles.health",
        "player_projectiles.health",
    }
    for name, value in names.items():
        assert value.device.type == "meta", name
        expected_dtype = (
            torch.bool
            if name in bool_fields or name.endswith(".mask")
            else torch.float32
            if name in float_fields
            else torch.int32
        )
        assert value.dtype == expected_dtype, name


def test_float_fields_ignore_the_default_dtype() -> None:
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        state = empty_state(num_envs=2, device=torch.device("cpu"))
    finally:
        torch.set_default_dtype(original_dtype)

    assert state.light_map.dtype == torch.float32
    assert state.player_health.dtype == torch.float32
    assert state.melee_mobs.health.dtype == torch.float32


def test_select_takes_rows_from_the_replacement_only_where_asked() -> None:
    current = _state()
    current.player_health += 9.0
    current.map += 1
    fresh = _state()

    merged = current.select(torch.tensor([True, False, True, False]), fresh)

    assert merged.player_health.tolist() == [0.0, 9.0, 0.0, 9.0]
    # The choice is per environment, and it reaches every rank of every field.
    assert merged.map[0].max() == 0
    assert merged.map[1].min() == 1
    assert merged.inventory.wood.shape == (4,)


def test_select_leaves_the_operands_untouched() -> None:
    current = _state()
    current.player_health += 5.0
    fresh = _state()

    _ = current.select(torch.ones(4, dtype=torch.bool), fresh)

    assert current.player_health.tolist() == [5.0] * 4
    assert fresh.player_health.tolist() == [0.0] * 4


def test_state_dict_round_trips_through_a_checkpoint() -> None:
    state = _state()
    state.player_health += 3.0
    state.inventory.wood += 7
    saved = {name: value.clone() for name, value in state.state_dict().items()}

    restored = _state()
    restored.load_state_dict(saved)

    assert restored.player_health.tolist() == [3.0] * 4
    assert restored.inventory.wood.tolist() == [7] * 4


def test_state_dict_names_nested_fields_by_path() -> None:
    names = _state().state_dict()
    assert "inventory.wood" in names
    assert "melee_mobs.position" in names
    assert "map" in names


def test_potion_mapping_is_per_environment() -> None:
    # Potion effects are randomized per episode, so this must never be shared
    # across the batch -- that is what makes the game partially observable.
    state = _state()
    state.potion_mapping[0] = torch.arange(6, dtype=torch.int32)
    assert state.potion_mapping[1].tolist() == [0] * 6


@pytest.mark.parametrize(
    ("field", "dtype"),
    [
        ("map", torch.int32),
        ("light_map", torch.float32),
        ("mob_map", torch.bool),
        ("achievements", torch.bool),
        ("player_health", torch.float32),
        ("player_food", torch.int32),
    ],
)
def test_field_dtypes_match_their_meaning(field: str, dtype: torch.dtype) -> None:
    value: object = getattr(_state(), field)  # pyright: ignore[reportAny] -- Field names are selected dynamically by pytest parameters.
    assert isinstance(value, torch.Tensor)
    assert value.dtype == dtype


def test_take_deals_one_batch_across_another() -> None:
    """Re-index the environment axis, repeating rows freely.

    This is what lets the optimistic reset generate two worlds and hand them
    to four finished workers.
    """
    state = empty_state(num_envs=2, device=torch.device("cpu"))
    state.timestep[:] = torch.tensor([7, 9], dtype=state.timestep.dtype)
    state.player_position[:] = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    dealt = state.take(torch.tensor([0, 1, 0, 1]))

    assert dealt.num_envs == 4
    assert dealt.timestep.tolist() == [7, 9, 7, 9]
    assert dealt.player_position.tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]


def test_take_returns_an_independent_state() -> None:
    # Rows are repeated, so a shared storage would make one worker's move
    # move its twin too.
    state = empty_state(num_envs=1, device=torch.device("cpu"))
    dealt = state.take(torch.tensor([0, 0]))
    dealt.timestep[0] = 5
    assert int(state.timestep[0]) == 0
    assert int(dealt.timestep[1]) == 0


def test_head_is_a_view_that_writes_through() -> None:
    state = _state(num_envs=3)
    state.head(2).timestep[:] = 4
    assert state.timestep.tolist() == [4, 4, 0]


def test_a_shallow_copy_can_be_rebound_without_moving_the_original() -> None:
    # A step rebinds the fields of the state it is handed; the environment's
    # own state must keep addressing the memory a CUDA graph replays against.
    state = _state()
    timestep = state.timestep
    alias = state.shallow_copy()
    alias.timestep = alias.timestep + 1
    alias.inventory.wood = alias.inventory.wood + 1
    alias.map[:] = 3
    assert state.timestep is timestep
    assert int(state.inventory.wood.sum()) == 0
    assert bool((state.map == 3).all())


def test_copy_overwrites_every_tensor_in_its_own_memory() -> None:
    state, source = _state(), _state()
    source.player_health += 2.0
    source.melee_mobs.health += 5.0
    health = state.player_health
    state.copy_(source)
    assert state.player_health is health
    assert state.player_health.tolist() == [2.0] * 4
    assert state.melee_mobs.health.eq(6.0).all()


def test_loading_a_checkpoint_keeps_the_state_in_its_own_memory() -> None:
    state, saved = _state(), _state()
    saved.timestep += 7
    timestep = state.timestep
    state.load_state_dict({k: v.clone() for k, v in saved.state_dict().items()})
    assert state.timestep is timestep
    assert state.timestep.tolist() == [7] * 4


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
