"""Tests for creature behaviour."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.game import constants, mobs
from priml.baselines.craftax.game.constants import Achievement, BlockType
from priml.baselines.craftax.game.state import EnvState, empty_state
from priml.lib.custom_json import ListCodec


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
        (state.melee_mobs.position[0, 0, 0] - state.player_position[0]).abs().sum(),
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
    landed = ListCodec.coerce(state.melee_mobs.position[0, 0, 0].tolist(), int)
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


def test_archer_fires_only_at_valid_distance_and_cooldown() -> None:
    """Upstream fires at gaps 4-5, unless cooldown blocks it (game_logic.py:1465-1489)."""
    state = _state(num_envs=3)
    state.ranged_mobs.mask[:, 0, 0] = True
    state.ranged_mobs.health[:, 0, 0] = 3.0
    state.ranged_mobs.position[:, 0, 0] = torch.tensor(
        [[10, 14], [10, 13], [10, 14]],
        dtype=torch.int32,
    )
    state.ranged_mobs.attack_cooldown[2, 0, 0] = 2
    state = mobs.update_mobs(state, generator=_seed())
    assert bool(state.mob_projectiles.mask[0, 0].any())
    assert not bool(state.mob_projectiles.mask[1, 0].any())
    assert not bool(state.mob_projectiles.mask[2, 0].any())


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


def test_a_projectile_stops_at_a_creature() -> None:
    """Upstream blocks projectile movement on mob occupancy (game_logic.py:1632-1637)."""
    state = _state(num_envs=1)
    state.player_position[:] = torch.tensor([10, 20], dtype=torch.int32)
    state.passive_mobs.mask[0, 0, 0] = True
    state.passive_mobs.health[0, 0, 0] = 3.0
    state.passive_mobs.position[0, 0, 0] = torch.tensor([10, 12], dtype=torch.int32)
    state.mob_map[0, 0, 10, 12] = True
    for row in range(9, 12):
        for column in range(11, 14):
            if (row, column) != (10, 12):
                state.map[0, 0, row, column] = int(BlockType.STONE)
    state.mob_projectiles.mask[0, 0, 0] = True
    state.mob_projectiles.position[0, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.mob_projectile_directions[0, 0, 0] = torch.tensor([0, 1], dtype=torch.int32)
    state = mobs.update_mobs(state, generator=_seed())
    assert not bool(state.mob_projectiles.mask[0, 0, 0])


def test_a_mob_projectile_hit_clears_resting() -> None:
    """Upstream projectile hits clear rest (game_logic.py:1696-1698)."""
    state = _state(num_envs=1)
    state.is_resting[:] = True
    state.mob_projectiles.mask[0, 0, 0] = True
    state.mob_projectiles.position[0, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.mob_projectile_directions[0, 0, 0] = torch.tensor([0, -1], dtype=torch.int32)
    state = mobs.update_mobs(state, generator=_seed())
    assert not bool(state.is_resting[0])


def test_a_mob_despawns_from_its_pre_move_distance() -> None:
    """Upstream tests initial distance against despawn range (game_logic.py:1319-1325)."""
    state = _state(num_envs=1)
    state.passive_mobs.mask[0, 0, 0] = True
    state.passive_mobs.health[0, 0, 0] = 3.0
    state.passive_mobs.position[0, 0, 0] = torch.tensor([10, 23], dtype=torch.int32)
    state.mob_map[0, 0, 10, 23] = True
    state = mobs.update_mobs(state, generator=_seed(91))
    assert bool(state.passive_mobs.mask[0, 0, 0])


def _arrow_at_melee(*, level: int, bow_enchantment: int) -> EnvState:
    """Place a player arrow one tile from a sturdy melee creature on ``level``."""
    state = _state()
    state.player_level[:] = level
    state.map[:] = int(BlockType.GRASS)
    state.melee_mobs.mask[:, level, 0] = True
    state.melee_mobs.health[:, level, 0] = 99.0
    state.melee_mobs.type_id[:, level, 0] = int(
        constants.FLOOR_MOB_TYPE[level, 1],
    )
    state.melee_mobs.position[:, level, 0] = torch.tensor([10, 14], dtype=torch.int32)
    state.mob_map[:, level, 10, 14] = True
    state.player_projectiles.mask[:, level, 0] = True
    state.player_projectiles.type_id[:, level, 0] = int(
        constants.ProjectileType.ARROW2,
    )
    state.player_projectiles.position[:, level, 0] = torch.tensor(
        [10, 13],
        dtype=torch.int32,
    )
    state.player_projectile_directions[:, level, 0] = torch.tensor(
        [0, 1],
        dtype=torch.int32,
    )
    state.bow_enchantment[:] = bow_enchantment
    return state


def test_a_player_arrow_wounds_the_creature_it_reaches() -> None:
    state = mobs.update_mobs(
        _arrow_at_melee(level=0, bow_enchantment=0),
        generator=_seed(),
    )
    assert float(state.melee_mobs.health[0, 0, 0]) < 99.0
    assert state.player_projectiles.mask[:, 0, 0].tolist() == [False, False]


def test_a_player_arrow_kill_counts_toward_clearing_the_floor() -> None:
    state = _arrow_at_melee(level=0, bow_enchantment=0)
    state.melee_mobs.health[:, 0, 0] = 0.5
    state = mobs.update_mobs(state, generator=_seed())
    assert state.melee_mobs.mask[:, 0, 0].tolist() == [False, False]
    assert state.monsters_killed[:, 0].tolist() == [1, 1]
    assert not bool(state.mob_map[0, 0, 10, 14])


def test_an_ice_bow_beats_a_fire_bow_in_the_fire_realm() -> None:
    # Fire Realm creatures shrug off fire and take ice in full, so the bow's
    # element decides the damage there -- the wall the carried-state
    # experiments are about.
    fire = mobs.update_mobs(
        _arrow_at_melee(level=6, bow_enchantment=1),
        generator=_seed(),
    )
    ice = mobs.update_mobs(
        _arrow_at_melee(level=6, bow_enchantment=2),
        generator=_seed(),
    )
    assert float(ice.melee_mobs.health[0, 6, 0]) < float(
        fire.melee_mobs.health[0, 6, 0],
    )


def test_spawning_fills_empty_slots_near_the_player() -> None:
    state = mobs.spawn_mobs(_state(num_envs=16), generator=_seed(5))
    spawned = state.melee_mobs.mask[:, 0].any(-1) | state.passive_mobs.mask[:, 0].any(
        -1,
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


def test_spawn_health_uses_species_not_floor() -> None:
    """Upstream indexes health by mob species (game_logic.py:2136-2140)."""
    state = _state(num_envs=64)
    state.player_level[:] = 1
    state = mobs.spawn_mobs(state, generator=_seed(51))
    groups = (
        (state.passive_mobs, 0),
        (state.melee_mobs, 1),
        (state.ranged_mobs, 2),
    )
    rows = torch.arange(state.num_envs)
    for group, mob_class in groups:
        spawned = group.mask[rows, 1]
        species = group.type_id[rows, 1]
        health = group.health[rows, 1]
        assert bool(spawned.any())
        assert torch.equal(
            health[spawned],
            constants.MOB_HEALTH.to(state.device)[species[spawned].long(), mob_class],
        )


def test_monsters_spawn_beyond_nine_tiles_only() -> None:
    """Upstream uses >9 tiles, or <=6 during a boss fight (game_logic.py:2181-2188)."""
    state = _state(num_envs=64)
    state.player_level[:] = 1
    state.map[:] = int(BlockType.STONE)
    state.map[:, 1, 10, 15] = int(BlockType.GRASS)
    state.passive_mobs.mask[:, 1] = True
    state = mobs.spawn_mobs(state, generator=_seed(3))
    assert not bool(state.melee_mobs.mask[:, 1].any())
    assert not bool(state.ranged_mobs.mask[:, 1].any())


def test_uncleared_multiplier_does_not_change_passive_spawning() -> None:
    """Upstream applies the uncleared multiplier only to monsters (game_logic.py:2056-2078)."""
    cleared = _state(num_envs=64)
    cleared.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    cleared = mobs.spawn_mobs(cleared, generator=_seed(71))
    uncleared = mobs.spawn_mobs(_state(num_envs=64), generator=_seed(71))
    assert torch.equal(cleared.passive_mobs.mask, uncleared.passive_mobs.mask)
    assert int(uncleared.melee_mobs.mask.sum()) >= int(cleared.melee_mobs.mask.sum())

    no_wave = _state(num_envs=8)
    no_wave.player_level[:] = constants.NUM_LEVELS - 1
    no_wave.map[:] = int(BlockType.STONE)
    no_wave.map[:, 8, 10, 15] = int(BlockType.GRAVE)
    no_wave = mobs.spawn_mobs(no_wave, generator=_seed(43))
    wave = _state(num_envs=8)
    wave.player_level[:] = constants.NUM_LEVELS - 1
    wave.boss_timesteps_to_spawn_this_round[:] = 1
    wave.map[:] = int(BlockType.STONE)
    wave.map[:, 8, 10, 15] = int(BlockType.GRAVE)
    wave = mobs.spawn_mobs(wave, generator=_seed(43))
    assert not bool(no_wave.melee_mobs.mask[:, 8].any())
    assert bool(wave.melee_mobs.mask[:, 8].any())


def test_deep_thing_needs_water_to_spawn() -> None:
    """Upstream restricts deep things to water tiles (game_logic.py:2326-2334)."""
    state = _state(num_envs=64)
    state.player_level[:] = 5
    state.map[:] = int(BlockType.STONE)
    state.map[:, 5, 10, 22] = int(BlockType.GRASS)
    state.passive_mobs.mask[:, 5] = True
    state.melee_mobs.mask[:, 5] = True
    state = mobs.spawn_mobs(state, generator=_seed(17))
    assert not bool(state.ranged_mobs.mask[:, 5].any())


def test_boss_floor_uses_progress_species_and_graves() -> None:
    """Upstream gates boss-floor spawns to graves and boss progress (game_logic.py:2195-2233)."""
    state = _state(num_envs=8)
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.boss_progress[:] = 2
    state.boss_timesteps_to_spawn_this_round[:] = 1
    state.map[:] = int(BlockType.STONE)
    state.map[:, 8, 10, 15] = int(BlockType.GRAVE)
    state = mobs.spawn_mobs(state, generator=_seed(21))
    assert not bool(state.passive_mobs.mask[:, 8].any())
    for group, mob_class in ((state.melee_mobs, 1), (state.ranged_mobs, 2)):
        live = group.mask[:, 8]
        assert bool(live.any())
        assert bool(
            (group.type_id[:, 8][live] == constants.FLOOR_MOB_TYPE[2, mob_class]).all(),
        )


def test_spawn_preserves_reused_slot_cooldown() -> None:
    """Upstream leaves a free slot's cooldown untouched (game_logic.py:2259-2281)."""
    state = _state(num_envs=64)
    state.melee_mobs.attack_cooldown[:] = 7
    state = mobs.spawn_mobs(state, generator=_seed(31))
    live = state.melee_mobs.mask[:, 0]
    assert bool(live.any())
    assert bool((state.melee_mobs.attack_cooldown[:, 0][live] == 7).all())


def test_creatures_never_leave_the_map() -> None:
    state = _with_melee(_state(num_envs=1), at=(0, 0))
    for _ in range(4):
        state = mobs.update_mobs(state, generator=_seed(17))
    positions = state.melee_mobs.position
    assert int(positions.min()) >= 0
    assert int(positions[..., 0].max()) < constants.MAP_SIZE[0]
    assert int(positions[..., 1].max()) < constants.MAP_SIZE[1]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
