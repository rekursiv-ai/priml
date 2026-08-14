"""Tests for creature behaviour."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax import constants, mobs
from priml.baselines.craftax.constants import Achievement, BlockType
from priml.baselines.craftax.state import EnvState, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.map[:] = int(BlockType.GRASS)
    state.player_health[:] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        getattr(state, meter)[:] = 9
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    state.light_level[:] = 1.0
    return state


def _with_melee(state: EnvState, *, at: tuple[int, int]) -> EnvState:
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 5.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor(at, dtype=torch.int32)
    state.mob_map[:, 0, at[0], at[1]] = True
    return state


def _seed(value: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(value)


def test_a_nearby_hunter_closes_on_the_player() -> None:
    state = _with_melee(_state(), at=(10, 13))
    state = mobs.update_mobs(state, generator=_seed())
    gap = int(
        (state.melee_mobs.position[0, 0, 0] - state.player_position[0]).abs().sum()
    )
    assert gap < 3


def test_an_adjacent_hunter_strikes_instead_of_stepping() -> None:
    state = _with_melee(_state(), at=(10, 11))
    state = mobs.update_mobs(state, generator=_seed())
    assert float(state.player_health[0]) < 9.0
    assert state.melee_mobs.position[0, 0, 0].tolist() == [10, 11]


def test_striking_starts_a_cooldown_so_blows_are_not_every_step() -> None:
    state = _with_melee(_state(), at=(10, 11))
    state = mobs.update_mobs(state, generator=_seed())
    assert int(state.melee_mobs.attack_cooldown[0, 0, 0]) == 5

    after = float(state.player_health[0])
    state = mobs.update_mobs(state, generator=_seed())
    assert float(state.player_health[0]) == pytest.approx(after)


def test_a_blow_wakes_a_sleeping_player() -> None:
    state = _with_melee(_state(), at=(10, 11))
    state.is_sleeping[:] = True
    state = mobs.update_mobs(state, generator=_seed())
    assert state.is_sleeping.tolist() == [False, False]
    assert state.achievements[:, int(Achievement.WAKE_UP)].tolist() == [True, True]


def test_sleeping_through_an_attack_costs_far_more_health() -> None:
    # Sleeping is a gamble, not a rest stop.
    awake = _with_melee(_state(), at=(10, 11))
    awake = mobs.update_mobs(awake, generator=_seed())
    asleep = _with_melee(_state(), at=(10, 11))
    asleep.is_sleeping[:] = True
    asleep = mobs.update_mobs(asleep, generator=_seed())
    assert float(asleep.player_health[0]) < float(awake.player_health[0])


def test_armour_blunts_a_creature_blow() -> None:
    bare = _with_melee(_state(), at=(10, 11))
    bare = mobs.update_mobs(bare, generator=_seed())
    armoured = _with_melee(_state(), at=(10, 11))
    armoured.inventory.armour[:] = 2
    armoured = mobs.update_mobs(armoured, generator=_seed())
    assert float(armoured.player_health[0]) > float(bare.player_health[0])


def test_a_creature_will_not_walk_into_stone() -> None:
    state = _with_melee(_state(), at=(10, 13))
    state.map[:, 0, 10, 12] = int(BlockType.STONE)
    state.map[:, 0, 9, 13] = int(BlockType.STONE)
    state.map[:, 0, 11, 13] = int(BlockType.STONE)
    state = mobs.update_mobs(state, generator=_seed())
    landed = state.melee_mobs.position[0, 0, 0].tolist()
    assert landed != [10, 12]
    assert state.map[0, 0, landed[0], landed[1]].item() != int(BlockType.STONE)


def test_a_distant_creature_despawns_to_free_its_slot() -> None:
    # The fixed slots are what keep the state rectangular, so a creature that
    # has wandered out of reach must give one up.
    state = _with_melee(_state(), at=(40, 40))
    state = mobs.update_mobs(state, generator=_seed())
    assert state.melee_mobs.mask[:, 0, 0].tolist() == [False, False]


def test_the_occupancy_grid_follows_the_creature() -> None:
    state = _with_melee(_state(), at=(10, 13))
    state = mobs.update_mobs(state, generator=_seed())
    landed = state.melee_mobs.position[0, 0, 0]
    assert bool(state.mob_map[0, 0, landed[0], landed[1]])
    assert not bool(state.mob_map[0, 0, 10, 13]) or landed.tolist() == [10, 13]


def test_a_grazing_creature_wanders_rather_than_hunting() -> None:
    state = _state(num_envs=32)
    state.passive_mobs.mask[:, 0, 0] = True
    state.passive_mobs.health[:, 0, 0] = 3.0
    state.passive_mobs.position[:, 0, 0] = torch.tensor([10, 14], dtype=torch.int32)
    state = mobs.update_mobs(state, generator=_seed(3))
    gaps = (state.passive_mobs.position[:, 0, 0] - state.player_position).abs().sum(-1)
    # A hunter would close on every draw; a wanderer sometimes retreats.
    assert int((gaps > 4).sum()) > 0


def test_an_archer_keeps_its_distance_and_fires_down_a_line() -> None:
    state = _state()
    state.ranged_mobs.mask[:, 0, 0] = True
    state.ranged_mobs.health[:, 0, 0] = 3.0
    state.ranged_mobs.position[:, 0, 0] = torch.tensor([10, 16], dtype=torch.int32)
    state = mobs.update_mobs(state, generator=_seed())
    assert bool(state.mob_projectiles.mask[0, 0].any())


def test_a_projectile_flies_and_wounds_the_player() -> None:
    state = _state()
    state.mob_projectiles.mask[:, 0, 0] = True
    state.mob_projectiles.position[:, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.mob_projectile_directions[:, 0, 0] = torch.tensor([0, -1], dtype=torch.int32)
    state = mobs.update_mobs(state, generator=_seed())
    assert float(state.player_health[0]) < 9.0
    assert state.mob_projectiles.mask[:, 0, 0].tolist() == [False, False]


def test_a_projectile_stops_at_a_wall() -> None:
    state = _state()
    state.mob_projectiles.mask[:, 0, 0] = True
    state.mob_projectiles.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)
    state.mob_projectile_directions[:, 0, 0] = torch.tensor([0, 1], dtype=torch.int32)
    state.map[:, 0, 5, 6] = int(BlockType.STONE)
    state = mobs.update_mobs(state, generator=_seed())
    assert state.mob_projectiles.mask[:, 0, 0].tolist() == [False, False]
    assert float(state.player_health[0]) == pytest.approx(9.0)


def test_spawning_fills_empty_slots_near_the_player() -> None:
    state = mobs.spawn_mobs(_state(num_envs=16), generator=_seed(5))
    spawned = state.melee_mobs.mask[:, 0].any(-1) | state.passive_mobs.mask[:, 0].any(
        -1
    )
    assert bool(spawned.any())
    positions = state.passive_mobs.position[:, 0, 0]
    gaps = (positions - state.player_position).abs().sum(-1)
    alive = state.passive_mobs.mask[:, 0, 0]
    # Nothing appears on top of the player.
    assert int(gaps[alive].min()) > 0


def test_an_uncleared_floor_spawns_faster() -> None:
    cleared = _state(num_envs=64)
    cleared.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    cleared = mobs.spawn_mobs(cleared, generator=_seed(7))

    uncleared = _state(num_envs=64)
    uncleared = mobs.spawn_mobs(uncleared, generator=_seed(7))

    assert int(uncleared.melee_mobs.mask.sum()) >= int(cleared.melee_mobs.mask.sum())


def test_night_brings_more_monsters_to_the_surface() -> None:
    day = _state(num_envs=128)
    day.light_level[:] = 1.0
    day = mobs.spawn_mobs(day, generator=_seed(11))

    night = _state(num_envs=128)
    night.light_level[:] = 0.0
    night = mobs.spawn_mobs(night, generator=_seed(11))

    assert int(night.melee_mobs.mask.sum()) > int(day.melee_mobs.mask.sum())


def test_no_cattle_graze_on_the_boss_floor() -> None:
    state = _state(num_envs=32)
    state.player_level[:] = constants.NUM_LEVELS - 1
    state = mobs.spawn_mobs(state, generator=_seed(13))
    assert int(state.passive_mobs.mask.sum()) == 0


def test_creatures_never_leave_the_map() -> None:
    state = _with_melee(_state(num_envs=8), at=(0, 0))
    for _ in range(20):
        state = mobs.update_mobs(state, generator=_seed(17))
    positions = state.melee_mobs.position
    assert int(positions.min()) >= 0
    assert int(positions[..., 0].max()) < constants.MAP_SIZE[0]
    assert int(positions[..., 1].max()) < constants.MAP_SIZE[1]


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
