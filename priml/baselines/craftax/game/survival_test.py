"""Tests for movement and the survival meters."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import constants, mechanics, survival
from priml.baselines.craftax.game.constants import Achievement, Action, BlockType
from priml.baselines.craftax.game.state import EnvState, empty_state


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


def _act(action: Action, num_envs: int = 2) -> Tensor:
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
            survival.update_intrinsics(state, _act(Action.NOOP)),
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
    state.player_health[:] = 1
    for _ in range(100):
        state = mechanics.clip_meters(
            survival.update_intrinsics(state, _act(Action.NOOP)),
        )
    assert float(state.player_health.max()) == pytest.approx(0.0)


def test_fatigue_boundaries_pay_back_energy_once() -> None:
    state = _state()
    state.is_sleeping[:] = True
    state.player_fatigue[:] = torch.tensor([-10.0, -9.0])
    state.player_energy[:] = 3

    survival._tick_fatigue(
        state,
        decay=torch.ones(2),
        starves=torch.ones(2, dtype=torch.bool),
    )

    assert state.player_fatigue.tolist() == [0.0, -10.0]
    assert state.player_energy.tolist() == [4, 3]


def test_fatigue_exhaustion_and_recovery_have_strict_thresholds() -> None:
    state = _state()
    state.player_fatigue[:] = torch.tensor([29.0, 30.0])
    state.player_energy[:] = 3

    survival._tick_fatigue(
        state,
        decay=torch.ones(2),
        starves=torch.ones(2, dtype=torch.bool),
    )

    assert state.player_fatigue.tolist() == [30.0, 0.0]
    assert state.player_energy.tolist() == [3, 2]


def test_health_recovery_requires_crossing_its_thresholds() -> None:
    state = _state()
    state.player_recover[:] = torch.tensor([25.0, -14.0])
    state.player_health[:] = 4.0
    state.player_food[:] = torch.tensor([9, 0])
    state.player_drink[:] = torch.tensor([9, 0])
    state.player_energy[:] = torch.tensor([1, 0])

    survival._tick_health(state, starves=torch.ones(2, dtype=torch.bool))

    assert state.player_health.tolist() == [5.0, 4.0]
    assert state.player_recover.tolist() == [0.0, -15.0]


def test_sleep_rest_and_mana_use_exact_meter_boundaries() -> None:
    state = _state()
    state.player_recover_mana[:] = 29.0
    survival._tick_mana(state)
    assert state.player_mana.tolist() == [9, 9]
    assert state.player_recover_mana.tolist() == [30.0, 30.0]

    state.player_recover_mana[:] = 30.0
    survival._tick_mana(state)
    assert state.player_mana.tolist() == [10, 10]
    assert state.player_recover_mana.tolist() == [0.0, 0.0]

    resting = _state()
    resting.player_health[:] = mechanics.max_health(resting)
    survival._update_sleep_and_rest(resting, _act(Action.REST))
    assert resting.is_resting.tolist() == [False, False]


def test_sleeping_and_drinking_meter_crossings_are_exact() -> None:
    state = _state()
    state.player_thirst[:] = 19.0
    state.player_drink[:] = 3
    state.player_energy[:] = 8
    state.is_sleeping[:] = True
    state = survival.update_intrinsics(state, _act(Action.NOOP))
    assert state.player_thirst.tolist() == [19.5, 19.5]
    assert state.player_drink.tolist() == [3, 3]


def test_movement_updates_facing_only_for_each_moving_environment() -> None:
    state = _state()
    state.player_direction[:] = torch.tensor([int(Action.UP), int(Action.LEFT)])
    action = torch.tensor([int(Action.RIGHT), int(Action.DO)], dtype=torch.int32)

    survival.move_player(state, action)

    assert state.player_position.tolist() == [[10, 11], [10, 10]]
    assert state.player_direction.tolist() == [int(Action.RIGHT), int(Action.LEFT)]


def test_sleeping_fatigue_clamps_at_zero_and_exhaustion_keeps_zero_energy() -> None:
    resting = _state()
    resting.is_sleeping[:] = True
    resting.player_fatigue[:] = torch.tensor([1.5, -9.0])
    survival._tick_fatigue(
        resting,
        decay=torch.ones(2),
        starves=torch.ones(2, dtype=torch.bool),
    )
    assert resting.player_fatigue.tolist() == [0.0, -10.0]
    assert resting.player_energy.tolist() == [9, 9]

    exhausted = _state()
    exhausted.player_fatigue[:] = 30.0
    exhausted.player_energy[:] = 0
    survival._tick_fatigue(
        exhausted,
        decay=torch.ones(2),
        starves=torch.zeros(2, dtype=torch.bool),
    )
    assert exhausted.player_fatigue.tolist() == [0.0, 0.0]
    assert exhausted.player_energy.tolist() == [0, 0]


def test_health_recovery_uses_exact_sustainability_and_sleep_rates() -> None:
    state = _state(num_envs=6)
    state.player_food[:] = torch.tensor([1, 0, 1, 1, 1, 0])
    state.player_drink[:] = torch.tensor([1, 1, 0, 1, 1, 1])
    state.player_energy[:] = torch.tensor([1, 1, 1, 0, 0, 0])
    state.is_sleeping[:] = torch.tensor([False, False, False, False, True, True])
    state.player_recover[:] = 0.0

    survival._tick_health(state, starves=torch.ones(6, dtype=torch.bool))

    assert state.player_health.tolist() == [9.0] * 6
    assert state.player_recover.tolist() == [1.0, -1.0, -1.0, -1.0, 2.0, -0.5]


def test_health_thresholds_are_strict_and_boss_floor_does_not_bleed() -> None:
    state = _state()
    state.player_health[:] = 4.0
    state.player_recover[:] = 24.0
    survival._tick_health(state, starves=torch.ones(2, dtype=torch.bool))
    assert state.player_health.tolist() == [4.0, 4.0]
    assert state.player_recover.tolist() == [25.0, 25.0]

    boss = _state()
    boss.player_food[:] = 0
    boss.player_recover[:] = 0.0
    survival._tick_health(boss, starves=torch.zeros(2, dtype=torch.bool))
    assert boss.player_recover.tolist() == [0.0, 0.0]
    assert boss.player_health.tolist() == [9.0, 9.0]


def test_mana_recovery_rate_is_exact_for_sleep_and_intelligence() -> None:
    state = _state(num_envs=3)
    state.player_intelligence[:] = torch.tensor([1, 5, 1])
    state.is_sleeping[:] = torch.tensor([False, False, True])
    state.player_recover_mana[:] = 0.0

    survival._tick_mana(state)

    assert state.player_mana.tolist() == [9, 9, 9]
    assert state.player_recover_mana.tolist() == [1.0, 2.0, 2.0]


def test_survival_accumulators_use_exact_sleep_and_dexterity_rates() -> None:
    state = _state(num_envs=4)
    state.player_dexterity[:] = torch.tensor([1, 5, 1, 5])
    state.player_energy[2:] = 3
    state.is_sleeping[:] = torch.tensor([False, False, True, True])

    state = survival.update_intrinsics(state, _act(Action.NOOP, num_envs=4))

    assert state.player_hunger.tolist() == [1.0, 0.5, 0.5, 0.25]
    assert state.player_thirst.tolist() == [1.0, 0.5, 0.5, 0.25]


def test_meter_does_not_spend_at_the_exact_threshold() -> None:
    accumulator, meter = survival._tick_meter(
        accumulator=torch.tensor([25.0, 20.0]),
        meter=torch.tensor([3, 4]),
        threshold=25.0,
        starves=torch.tensor([True, True]),
    )

    assert accumulator.tolist() == [25.0, 20.0]
    assert meter.tolist() == [3, 4]


def test_thirst_spends_a_drink_when_accumulator_crosses_twenty() -> None:
    state = _state()
    state.player_thirst[:] = 20.0
    state.player_drink[:] = 3

    state = survival.update_intrinsics(state, _act(Action.NOOP))

    assert state.player_thirst.tolist() == [0.0, 0.0]
    assert state.player_drink.tolist() == [2, 2]


def test_sleep_and_rest_start_only_for_the_requested_action() -> None:
    resting = _state()
    resting.player_health[:] = 8.0
    resting.player_energy[:] = 3
    survival._update_sleep_and_rest(resting, _act(Action.NOOP))
    assert resting.is_resting.tolist() == [False, False]

    sleep_at_cap = _state()
    survival._update_sleep_and_rest(sleep_at_cap, _act(Action.SLEEP))
    assert sleep_at_cap.is_sleeping.tolist() == [False, False]

    start_rest = _state()
    start_rest.player_health[:] = 8.0
    survival._update_sleep_and_rest(start_rest, _act(Action.REST))
    assert start_rest.is_resting.tolist() == [True, True]


def test_rest_stops_at_zero_food_or_drink_but_not_one() -> None:
    food_one = _state()
    food_one.player_health[:] = 8.0
    food_one.is_resting[:] = True
    food_one.player_food[:] = 1
    survival._update_sleep_and_rest(food_one, _act(Action.NOOP))
    assert food_one.is_resting.tolist() == [True, True]

    full_health = _state()
    full_health.is_resting[:] = True
    survival._update_sleep_and_rest(full_health, _act(Action.NOOP))
    assert full_health.is_resting.tolist() == [False, False]

    drink_zero = _state()
    drink_zero.player_health[:] = 8.0
    drink_zero.is_resting[:] = True
    drink_zero.player_drink[:] = 0
    survival._update_sleep_and_rest(drink_zero, _act(Action.NOOP))
    assert drink_zero.is_resting.tolist() == [False, False]

    drink_one = _state()
    drink_one.player_health[:] = 8.0
    drink_one.is_resting[:] = True
    drink_one.player_drink[:] = 1
    survival._update_sleep_and_rest(drink_one, _act(Action.NOOP))
    assert drink_one.is_resting.tolist() == [True, True]


def test_move_player_passes_its_device_to_lookup_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    devices: list[torch.device | None] = []
    original = constants.on_device

    def on_device(table: Tensor, device: torch.device | None) -> Tensor:
        if table is constants.DIRECTIONS or table is constants.PLAYER_COLLIDES_WITH:
            devices.append(device)
        assert device is not None
        return original(table, device)

    monkeypatch.setattr(constants, "on_device", on_device)
    survival.move_player(state, _act(Action.UP))

    assert devices == [state.device, state.device]


def test_waking_creates_achievement_indices_on_the_state_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = empty_state(num_envs=1, device=torch.device("meta"))
    state.player_energy[:] = mechanics.max_energy(state)
    state.is_sleeping[:] = True
    achievement_devices: list[torch.device] = []
    unlock = mechanics.unlock_achievement

    def unlock_spy(
        state: EnvState,
        achievement: Tensor,
        earned: Tensor,
    ) -> Tensor:
        achievement_devices.append(achievement.device)
        return unlock(state, achievement, earned)

    monkeypatch.setattr(mechanics, "unlock_achievement", unlock_spy)
    result = survival._update_sleep_and_rest(
        state,
        torch.tensor(
            [int(Action.NOOP)],
            dtype=torch.int32,
            device=torch.device("meta"),
        ),
    )

    assert achievement_devices == [torch.device("meta")]
    assert result.achievements.device == torch.device("meta")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
