"""Tests for movement and the survival meters."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax import constants, mechanics, survival
from priml.baselines.craftax.constants import Achievement, Action, BlockType
from priml.baselines.craftax.state import EnvState, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.player_direction[:] = int(Action.UP)
    state.map[:] = int(BlockType.GRASS)
    state.player_health[:] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        getattr(state, meter)[:] = 9
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    return state


def _act(action: Action, num_envs: int = 2) -> torch.Tensor:
    return torch.full((num_envs,), int(action), dtype=torch.int32)


def test_walking_moves_one_tile_in_the_chosen_direction() -> None:
    state = survival.move_player(_state(), _act(Action.RIGHT))
    assert state.player_position[0].tolist() == [10, 11]


def test_walking_into_stone_turns_the_player_without_moving_them() -> None:
    # Facing a wall you cannot enter is what lets you mine it.
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.STONE)
    state = survival.move_player(state, _act(Action.RIGHT))
    assert state.player_position[0].tolist() == [10, 10]
    assert state.player_direction[0].item() == int(Action.RIGHT)


def test_a_non_movement_action_leaves_the_facing_alone() -> None:
    state = _state()
    state.player_direction[:] = int(Action.LEFT)
    state = survival.move_player(state, _act(Action.DO))
    assert state.player_direction[0].item() == int(Action.LEFT)


def test_each_environment_moves_independently() -> None:
    state = _state()
    action = torch.tensor([int(Action.UP), int(Action.DOWN)], dtype=torch.int32)
    state = survival.move_player(state, action)
    assert state.player_position.tolist() == [[9, 10], [11, 10]]


def test_hunger_costs_a_food_point_only_when_it_crosses_over() -> None:
    state = _state()
    state.player_hunger[:] = 24.5
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_food.tolist() == [8, 8]
    assert state.player_hunger.tolist() == [0.0, 0.0]


def test_hunger_below_the_threshold_costs_nothing_yet() -> None:
    state = _state()
    state.player_hunger[:] = 10.0
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_food.tolist() == [9, 9]
    assert state.player_hunger.tolist() == [11.0, 11.0]


def test_dexterity_slows_the_whole_body_clock() -> None:
    quick = _state()
    quick.player_dexterity[:] = 5
    quick = survival.update_intrinsics(quick, _act(Action.NOOP))
    ordinary = survival.update_intrinsics(_state(), _act(Action.NOOP))
    assert float(quick.player_hunger[0]) < float(ordinary.player_hunger[0])


def test_sleeping_starts_only_when_tired() -> None:
    rested = _state()
    rested = survival.update_intrinsics(rested, _act(Action.SLEEP))
    assert rested.is_sleeping.tolist() == [False, False]

    tired = _state()
    tired.player_energy[:] = 3
    tired = survival.update_intrinsics(tired, _act(Action.SLEEP))
    assert tired.is_sleeping.tolist() == [True, True]


def test_waking_at_full_energy_unlocks_its_achievement() -> None:
    state = _state()
    state.is_sleeping[:] = True
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.is_sleeping.tolist() == [False, False]
    assert state.achievements[:, int(Achievement.WAKE_UP)].tolist() == [True, True]


def test_sleep_slows_hunger_and_repays_fatigue() -> None:
    awake = survival.update_intrinsics(_state(), _act(Action.NOOP))
    asleep = _state()
    asleep.player_energy[:] = 3
    asleep.is_sleeping[:] = True
    asleep = survival.update_intrinsics(asleep, _act(Action.NOOP))
    assert float(asleep.player_hunger[0]) < float(awake.player_hunger[0])
    assert float(asleep.player_fatigue[0]) < float(awake.player_fatigue[0])


def test_resting_stops_when_the_stomach_empties() -> None:
    # Otherwise resting would be a way to sit out starvation.
    state = _state()
    state.player_health[:] = 3.0
    state.is_resting[:] = True
    state.player_food[:] = 0
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.is_resting.tolist() == [False, False]


def test_a_sustained_player_heals_over_time() -> None:
    state = _state()
    state.player_health[:] = 4.0
    state.player_recover[:] = 25.0
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_health.tolist() == [5.0, 5.0]


def test_a_starving_player_loses_health() -> None:
    state = _state()
    state.player_food[:] = 0
    state.player_recover[:] = -15.0
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_health.tolist() == [8.0, 8.0]


def test_starvation_is_suspended_on_the_boss_floor() -> None:
    # The final fight is decided by combat, not by the clock.
    state = _state()
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.player_hunger[:] = 26.0
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_food.tolist() == [9, 9]


def test_mana_refills_and_intelligence_speeds_it() -> None:
    state = _state()
    state.player_mana[:] = 2
    state.player_recover_mana[:] = 30.0
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_mana.tolist() == [3, 3]

    clever = _state()
    clever.player_intelligence[:] = 5
    clever = survival.update_intrinsics(clever, _act(Action.NOOP))
    ordinary = survival.update_intrinsics(_state(), _act(Action.NOOP))
    assert float(clever.player_recover_mana[0]) > float(
        ordinary.player_recover_mana[0],
    )


def test_meters_never_leave_their_range_over_a_long_life() -> None:
    state = _state()
    for _ in range(200):
        state = mechanics.clip_meters(
            survival.update_intrinsics(state, _act(Action.NOOP))
        )
    assert float(state.player_food.min()) >= 0
    assert float(state.player_health.min()) >= 0
    assert float(state.player_food.max()) <= float(mechanics.max_food(state)[0])
    assert float(state.player_health.max()) <= float(mechanics.max_health(state)[0])


def test_an_unfed_player_eventually_dies() -> None:
    # The whole survival loop must terminate an idle episode.
    state = _state()
    state.player_food[:] = 0
    state.player_drink[:] = 0
    for _ in range(400):
        state = mechanics.clip_meters(
            survival.update_intrinsics(state, _act(Action.NOOP))
        )
    assert float(state.player_health.max()) == pytest.approx(0.0)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
