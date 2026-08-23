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
    for name, tensor in state.state_dict().items():
        assert tensor.shape[0] == 3, name


def test_shapes_follow_the_declared_world_size() -> None:
    state = _state()
    levels, (rows, columns) = constants.NUM_LEVELS, constants.MAP_SIZE
    assert state.map.shape == (4, levels, rows, columns)
    assert state.melee_mobs.position.shape == (
        4,
        levels,
        constants.MAX_MELEE_MOBS,
        2,
    )
    assert state.achievements.shape == (4, len(constants.Achievement))
    assert state.inventory.potions.shape == (4, 6)
    assert state.num_envs == 4


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
    saved = {name: tensor.clone() for name, tensor in state.state_dict().items()}

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
    assert getattr(_state(), field).dtype == dtype


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
