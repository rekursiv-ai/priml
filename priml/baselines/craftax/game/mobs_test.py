"""Tests for creature behaviour."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.game import constants, indexing, mechanics, mobs
from priml.baselines.craftax.game.constants import Achievement, BlockType
from priml.baselines.craftax.game.state import EnvState, Mobs, empty_state
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


def test_hunter_switches_to_wandering_at_ten_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=4)
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.position[:, 0, 0] = torch.tensor(
        [[10, 19], [10, 20], [10, 21], [10, 19]],
        dtype=torch.int32,
    )
    state.mob_map[torch.arange(4), 0, 10, torch.tensor([19, 20, 21, 19])] = True
    calls = 0

    def pinned_rand(
        size: tuple[int, ...] | int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        nonlocal calls
        del generator
        calls += 1
        shape = (size,) if isinstance(size, int) else size
        if calls % 2:
            return torch.full(
                shape,
                0.6,
                dtype=dtype or torch.float32,
                device=device,
            )
        return torch.tensor(
            [0.5, 0.5, 0.5, 0.75],
            dtype=dtype or torch.float32,
            device=device,
        )

    def choose_last(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del low, generator
        return torch.full(size, high - 1, device=device, dtype=dtype or torch.int64)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    monkeypatch.setattr(torch, "randint", choose_last)
    state = mobs._update_melee(state, generator=_seed(29))

    assert state.melee_mobs.position[:, 0, 0].tolist() == [
        [10, 18],
        (torch.tensor([10, 20]) + constants.CLOSE_BLOCKS[3]).tolist(),
        (torch.tensor([10, 21]) + constants.CLOSE_BLOCKS[3]).tolist(),
        (torch.tensor([10, 19]) + constants.CLOSE_BLOCKS[3]).tolist(),
    ]


def test_striking_starts_a_cooldown_so_blows_are_not_every_step() -> None:
    state = _with_melee(_state(), at=(10, 11))
    state.melee_mobs.attack_cooldown[1, 0, 0] = 1
    state = mobs.update_mobs(state, generator=_seed())
    assert state.melee_mobs.attack_cooldown[:, 0, 0].tolist() == [5, 0]
    assert float(state.player_health[0]) < 9.0
    assert state.player_health.tolist()[1] == 9.0

    after = float(state.player_health[0])
    state = mobs.update_mobs(state, generator=_seed())
    assert float(state.player_health[0]) == pytest.approx(after)


def test_a_blow_wakes_a_sleeping_player() -> None:
    state = _with_melee(_state(), at=(10, 11))
    state.is_sleeping[:] = True
    state = mobs.update_mobs(state, generator=_seed())
    assert state.is_sleeping.tolist() == [False, False]
    assert state.achievements[:, int(Achievement.WAKE_UP)].tolist() == [True, True]


def test_strikes_only_wake_sleepers_and_interrupt_resting() -> None:
    state = _state(num_envs=4)
    state.is_sleeping[:] = torch.tensor([True, False, True, False])
    state.is_resting[:] = True

    state = mobs._strike_player(
        state,
        species=torch.zeros(4, dtype=torch.int32),
        mob_class=1,
        striking=torch.tensor([True, True, False, False]),
    )

    assert state.is_sleeping.tolist() == [False, False, True, False]
    assert state.is_resting.tolist() == [False, False, True, True]
    assert state.achievements[:, int(Achievement.WAKE_UP)].tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_strike_player_scales_sleep_damage_and_wakes_only_hit_sleepers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=4)
    state.is_sleeping[:] = torch.tensor([True, True, False, False])
    state.is_resting[:] = True
    full_devices: list[torch.device | str | None] = []
    real_full = torch.full

    def full(
        size: tuple[int, ...],
        fill_value: float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        full_devices.append(device)
        return real_full(size, fill_value, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "full", full)
    state = mobs._strike_player(
        state,
        species=torch.zeros(4, dtype=torch.int32),
        mob_class=1,
        striking=torch.tensor([True, False, True, False]),
    )

    assert state.player_health.tolist() == [2.0, 9.0, 7.0, 9.0]
    assert state.is_sleeping.tolist() == [False, True, False, False]
    assert state.is_resting.tolist() == [False, True, False, True]
    assert state.achievements[:, int(Achievement.WAKE_UP)].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert full_devices == [state.device]


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


def test_melee_collision_uses_its_class_on_the_fire_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.player_level[:] = 6
    state.melee_mobs.mask[0, 6, 0] = True
    state.melee_mobs.position[0, 6, 0] = torch.tensor([10, 13], dtype=torch.int32)
    state.mob_map[0, 6, 10, 13] = True
    state.map[0, 6, 10, 12] = int(BlockType.WATER)

    def no_randomness(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.zeros(size, device=device)

    monkeypatch.setattr(torch, "rand", no_randomness)
    state = mobs._update_melee(state, generator=_seed(53))

    assert state.melee_mobs.position[0, 6, 0].tolist() == [10, 13]
    assert state.mob_map[0, 6, 10, 13].item() is True
    assert state.mob_map[0, 6, 10, 12].item() is False


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


def test_passive_wandering_has_four_stationary_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream samples DIRECTIONS[1:9], including four no-ops (game_logic.py:1297-1305)."""
    state = _state(num_envs=1)
    state.passive_mobs.position[0, 0, 0] = torch.tensor([10, 14], dtype=torch.int32)
    state.passive_mobs.mask[0, 0, 0] = True
    state.mob_map[0, 0, 10, 14] = True

    def choose_last(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del low, generator
        return torch.full(size, high - 1, device=device, dtype=dtype or torch.int64)

    monkeypatch.setattr(torch, "randint", choose_last)
    state = mobs.update_mobs(state, generator=_seed())

    assert state.passive_mobs.position[0, 0, 0].tolist() == [10, 14]


def test_empty_passive_slots_keep_the_upstream_proposed_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream writes proposed positions regardless of the mob mask (game_logic.py:1359-1369)."""
    state = _state(num_envs=1)
    state.passive_mobs.position[0, 0, 1] = torch.tensor([10, 14], dtype=torch.int32)

    def choose_first(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del high, generator
        return torch.full(size, low, device=device, dtype=dtype or torch.int64)

    monkeypatch.setattr(torch, "randint", choose_first)
    updated = mobs.update_mobs(state, generator=_seed())

    assert updated.passive_mobs.position[0, 0, 1].tolist() == [10, 13]
    assert not updated.passive_mobs.mask[0, 0, 1]


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


def test_projectile_hits_update_only_the_matching_players_state() -> None:
    state = _state(num_envs=2)
    state.is_sleeping[:] = True
    state.is_resting[:] = True
    state.mob_projectiles.mask[:, 0, 0] = True
    state.mob_projectiles.position[:, 0, 0] = torch.tensor(
        [[10, 11], [8, 11]],
        dtype=torch.int32,
    )
    state.mob_projectile_directions[:, 0, 0] = torch.tensor(
        [[0, -1], [0, -1]],
        dtype=torch.int32,
    )

    state = mobs._update_projectiles(state)

    assert state.player_health.tolist() == [7.0, 9.0]
    assert state.is_sleeping.tolist() == [False, True]
    assert state.is_resting.tolist() == [False, True]
    assert state.mob_projectiles.mask[:, 0, 0].tolist() == [False, True]
    assert state.mob_projectiles.position[:, 0, 0].tolist() == [[10, 10], [8, 10]]


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


def test_hit_player_projectile_keeps_the_upstream_impact_position() -> None:
    """Upstream records impact position before clearing the mask (game_logic.py:1795-1814)."""
    state = mobs.update_mobs(
        _arrow_at_melee(level=0, bow_enchantment=0),
        generator=_seed(),
    )
    assert state.player_projectiles.position[:, 0, 0].tolist() == [[10, 14], [10, 14]]


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


def test_passive_spawns_accept_each_walkable_ground_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=5)
    state.map[:] = int(BlockType.STONE)
    state.map[:, 0, 10, 14] = torch.tensor(
        [
            int(BlockType.GRASS),
            int(BlockType.PATH),
            int(BlockType.FIRE_GRASS),
            int(BlockType.ICE_GRASS),
            int(BlockType.STONE),
        ],
    )

    def always_try(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.zeros(size, dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", always_try)
    state = mobs.spawn_mobs(state, generator=_seed(79))

    assert state.passive_mobs.mask[:, 0, 0].tolist() == [True, True, True, True, False]
    assert state.passive_mobs.position[:4, 0, 0].tolist() == [[10, 14]] * 4


def test_passive_spawns_respect_both_distance_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=3)
    state.map[:] = int(BlockType.STONE)
    state.map[0, 0, 10, 13] = int(BlockType.GRASS)
    state.map[1, 0, 10, 14] = int(BlockType.GRASS)
    state.map[2, 0, 10, 10 + constants.MOB_DESPAWN_DISTANCE] = int(
        BlockType.GRASS,
    )

    def always_try(
        size: tuple[int, ...] | int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        shape = (size,) if isinstance(size, int) else size
        return torch.zeros(shape, dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", always_try)
    state = mobs.spawn_mobs(state, generator=_seed(31))

    assert state.passive_mobs.mask[:, 0, 0].tolist() == [False, True, False]


def test_later_spawn_classes_see_earlier_mob_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream rebuilds spawn masks per class (game_logic.py:2104-2106, 2227-2229)."""
    state = _state(num_envs=1)
    state.map[:] = int(BlockType.STONE)
    state.map[0, 0, 10, 20] = int(BlockType.GRASS)

    def pinned_rand(
        size: tuple[int, ...] | int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        shape = (size,) if isinstance(size, int) else size
        return torch.zeros(shape, dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed())

    assert state.passive_mobs.mask[0, 0, 0]
    assert not state.melee_mobs.mask[0, 0].any()
    assert not state.ranged_mobs.mask[0, 0].any()


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


def test_melee_spawn_chance_uses_the_squared_light_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.light_level[:] = 0.5
    state.monsters_killed[:] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state.map[:] = int(BlockType.STONE)
    state.map[0, 0, 10, 20] = int(BlockType.GRASS)
    draws = iter((0.99, 0.055, 0.99))

    def pinned_rand(
        size: tuple[int, ...] | int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        shape = (size,) if isinstance(size, int) else size
        return torch.full(
            shape,
            next(draws),
            dtype=dtype or torch.float32,
            device=device,
        )

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed(37))

    assert not state.melee_mobs.mask[0, 0].any()


def test_melee_spawn_uses_exact_night_probability_and_strict_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=2)
    state.light_level[:] = 0.5
    state.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state.map[:] = int(BlockType.STONE)
    state.map[:, 0, 10, 20] = int(BlockType.GRASS)
    chance = (
        constants.FLOOR_MOB_SPAWN_CHANCE[0, 1]
        + constants.FLOOR_MOB_SPAWN_CHANCE[0, 3] * (1.0 - state.light_level) ** 2
    )
    draws = iter(
        (
            torch.ones(2),
            torch.stack((chance[0] - 0.005, chance[0])),
            torch.ones(2),
        ),
    )

    def pinned_rand(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator, size
        return next(draws).to(device=device, dtype=dtype or torch.float32)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed(97))

    assert state.melee_mobs.mask[:, 0, 0].tolist() == [True, False]
    assert state.melee_mobs.position[0, 0, 0].tolist() == [10, 20]


def test_uncleared_melee_spawn_uses_a_threefold_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=2)
    state.map[:] = int(BlockType.STONE)
    state.map[:, 0, 10, 20] = int(BlockType.GRASS)
    draws = iter((torch.ones(2), torch.tensor([0.05, 0.07]), torch.ones(2)))

    def pinned_rand(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator, size
        return next(draws).to(device=device, dtype=dtype or torch.float32)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed(101))

    assert state.melee_mobs.mask[:, 0, 0].tolist() == [True, False]
    assert state.melee_mobs.position[0, 0, 0].tolist() == [10, 20]


def test_no_cattle_graze_on_the_boss_floor() -> None:
    state = _state(num_envs=32)
    state.player_level[:] = constants.NUM_LEVELS - 1
    state = mobs.spawn_mobs(state, generator=_seed(13))
    assert int(state.passive_mobs.mask.sum()) == 0


def test_empty_spawn_slots_record_the_current_floor_species(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream updates slot species even when spawn is false (game_logic.py:2159-2161, 2283-2285)."""
    state = _state(num_envs=1)
    state.player_level[:] = 1

    def prevent_spawn(
        size: tuple[int, ...] | int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        shape = (size,) if isinstance(size, int) else size
        return torch.full(shape, 0.99, dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", prevent_spawn)
    state = mobs.spawn_mobs(state, generator=_seed())

    assert not state.passive_mobs.mask[0, 1].any()
    assert not state.melee_mobs.mask[0, 1].any()
    assert not state.ranged_mobs.mask[0, 1].any()
    assert state.passive_mobs.type_id[0, 1, 0] == constants.FLOOR_MOB_TYPE[1, 0]
    assert state.melee_mobs.type_id[0, 1, 0] == constants.FLOOR_MOB_TYPE[1, 1]
    assert state.ranged_mobs.type_id[0, 1, 0] == constants.FLOOR_MOB_TYPE[1, 2]


def test_spawning_requires_a_free_slot_and_uses_its_first_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=2)
    state.map[:] = int(BlockType.STONE)
    state.map[:, 0, 10, 20] = int(BlockType.GRASS)
    state.melee_mobs.mask[0, 0] = True
    state.melee_mobs.mask[1, 0, 0] = True
    state.melee_mobs.position[1, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)
    draws = iter((torch.ones(2), torch.zeros(2), torch.ones(2)))

    def pinned_rand(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator, size
        return next(draws).to(device=device, dtype=dtype or torch.float32)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed(103))

    assert state.melee_mobs.mask[0, 0].tolist() == [True, True, True]
    assert state.melee_mobs.mask[1, 0].tolist() == [True, True, False]
    assert state.melee_mobs.position[1, 0, 0].tolist() == [5, 5]
    assert state.melee_mobs.position[1, 0, 1].tolist() == [10, 20]


def test_spawn_health_uses_species_not_floor() -> None:
    """Upstream indexes health by species (game_logic.py:2136-2140)."""
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


def test_monster_spawn_distance_is_strict_at_nine_and_despawn_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=3)
    state.map[:] = int(BlockType.STONE)
    state.map[0, 0, 10, 19] = int(BlockType.GRASS)
    state.map[1, 0, 10, 20] = int(BlockType.GRASS)
    state.map[2, 0, 10, 24] = int(BlockType.GRASS)
    state.passive_mobs.mask[:, 0] = True

    def always_try(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.zeros(size, dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", always_try)
    state = mobs.spawn_mobs(state, generator=_seed(83))

    assert state.melee_mobs.mask[:, 0, 0].tolist() == [False, True, False]
    assert state.melee_mobs.position[1, 0, 0].tolist() == [10, 20]
    assert not bool(state.ranged_mobs.mask[:, 0].any())


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
    # Melee spawns first and occupies its grave, so ranged needs a second one.
    state.map[:, 8, 10, 15] = int(BlockType.GRAVE)
    state.map[:, 8, 10, 5] = int(BlockType.GRAVE)
    state = mobs.spawn_mobs(state, generator=_seed(21))
    assert not bool(state.passive_mobs.mask[:, 8].any())
    for group, mob_class in ((state.melee_mobs, 1), (state.ranged_mobs, 2)):
        live = group.mask[:, 8]
        assert bool(live.any())
        assert bool(
            (group.type_id[:, 8][live] == constants.FLOOR_MOB_TYPE[2, mob_class]).all(),
        )


def test_boss_spawns_accept_all_graves_through_distance_six(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=4)
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.boss_timesteps_to_spawn_this_round[:] = 1
    state.map[:] = int(BlockType.STONE)
    state.map[0, 8, 10, 16] = int(BlockType.GRAVE)
    state.map[1, 8, 10, 16] = int(BlockType.GRAVE2)
    state.map[2, 8, 10, 16] = int(BlockType.GRAVE3)
    state.map[3, 8, 10, 17] = int(BlockType.GRAVE)

    def always_try(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.zeros(size, dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", always_try)
    state = mobs.spawn_mobs(state, generator=_seed(89))

    assert state.melee_mobs.mask[:, 8, 0].tolist() == [True, True, True, False]
    assert state.melee_mobs.position[:3, 8, 0].tolist() == [[10, 16]] * 3
    assert state.passive_mobs.mask[:, 8].any().item() is False


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


def test_distance_to_player_is_euclidean_for_each_map_axis() -> None:
    state = _state(num_envs=1)
    state.player_position[0] = torch.tensor([11, 13], dtype=torch.int32)

    distance = mobs._distance_to_player(state)

    assert distance[0, 14, 17].item() == pytest.approx(5.0)
    assert distance[0, 8, 9].item() == pytest.approx(5.0)


def test_distance_to_player_builds_indices_on_the_state_device() -> None:
    device = torch.device("meta")
    state = empty_state(num_envs=2, device=device)

    assert mobs._distance_to_player(state).device == state.device


def test_mob_updates_preserve_generator_and_device_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    state = _state()
    generator = _seed(73)
    rand_generators: list[torch.Generator | None] = []
    rand_devices: list[torch.device | str | None] = []
    randint_generators: list[torch.Generator | None] = []
    randint_devices: list[torch.device | str | None] = []
    lookup_devices: list[torch.device] = []
    row_devices: list[torch.device] = []
    factory_calls: list[tuple[str, torch.dtype | None, torch.device | str | None]] = []
    real_rand = torch.rand
    real_randint = torch.randint
    real_ones = torch.ones
    real_zeros = torch.zeros
    real_full = torch.full
    real_on_device = constants.on_device
    real_batch_rows = indexing.batch_rows
    real_sample_position = mobs._sample_position
    sample_generators: list[torch.Generator | None] = []
    random_step_counts: list[int] = []
    real_random_step = mobs._random_step

    def random_step(
        num_envs: int,
        target: torch.device,
        source: torch.Generator | None,
        *,
        moves: int,
    ) -> torch.Tensor:
        random_step_counts.append(moves)
        return real_random_step(num_envs, target, source, moves=moves)

    def sample_position(
        room: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        sample_generators.append(generator)
        return real_sample_position(room, generator=generator)

    def rand(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        rand_generators.append(generator)
        rand_devices.append(device)
        return real_rand(*size, generator=generator, device=device, dtype=dtype)

    def randint(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        randint_generators.append(generator)
        randint_devices.append(device)
        return real_randint(low, high, size, generator=generator, device=device)

    def ones(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        factory_calls.append(("ones", dtype, device))
        return real_ones(size, dtype=dtype, device=device)

    def zeros(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        factory_calls.append(("zeros", dtype, device))
        return real_zeros(size, dtype=dtype, device=device)

    def full(
        size: tuple[int, ...],
        fill_value: float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        factory_calls.append(("full", dtype, device))
        return real_full(size, fill_value, dtype=dtype, device=device)

    def on_device(table: torch.Tensor, target: torch.device) -> torch.Tensor:
        lookup_devices.append(target)
        return real_on_device(table, target)

    def batch_rows(envs: int, target: torch.device) -> torch.Tensor:
        row_devices.append(target)
        return real_batch_rows(envs, target)

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "full", full)
    monkeypatch.setattr(constants, "on_device", on_device)
    monkeypatch.setattr("priml.baselines.craftax.game.mobs.batch_rows", batch_rows)
    monkeypatch.setattr(mobs, "_sample_position", sample_position)
    monkeypatch.setattr(mobs, "_random_step", random_step)

    mobs.update_mobs(state, generator=generator)
    mobs.spawn_mobs(state, generator=generator)

    assert rand_generators == [generator] * 13
    assert rand_devices == [device] * 13
    assert randint_generators == [generator] * 8
    assert randint_devices == [device] * 8
    assert sample_generators == [generator] * 3
    assert random_step_counts == [4, 4, 4, 8, 8, 8, 4, 4]
    assert lookup_devices
    assert all(target == device for target in lookup_devices)
    assert row_devices
    assert all(target == device for target in row_devices)
    assert factory_calls
    assert all(target == device for _, _, target in factory_calls)
    assert all(
        dtype == torch.bool for name, dtype, _ in factory_calls if name != "full"
    )
    assert ("full", None, device) in factory_calls


def test_firing_uses_free_slots_and_axis_aligned_headings() -> None:
    state = _state(num_envs=4)
    state.mob_projectiles.mask[0, 0] = True
    state.mob_projectiles.mask[2, 0, 0] = True
    state.mob_projectiles.position[0, 0] = torch.tensor(
        [[4, 4], [5, 5], [6, 6]],
        dtype=torch.int32,
    )
    source = torch.tensor([[10, 10], [11, 9], [12, 11], [13, 12]], dtype=torch.int32)
    toward = torch.tensor([[4, 0], [3, 0], [0, -3], [0, 3]], dtype=torch.int32)

    state = mobs._fire_projectile(
        state,
        source=source,
        toward=toward,
        species=torch.tensor([0, 0, 2, 1]),
        firing=torch.tensor([True, True, True, False]),
    )

    assert state.mob_projectiles.position[0, 0].tolist() == [
        [4, 4],
        [5, 5],
        [6, 6],
    ]
    assert state.mob_projectiles.position[1, 0, 0].tolist() == [12, 9]
    assert state.mob_projectile_directions[1, 0, 0].tolist() == [1, 0]
    assert state.mob_projectiles.type_id[1, 0, 0].item() == int(
        constants.ProjectileType.ARROW,
    )
    assert state.mob_projectiles.position[2, 0, 1].tolist() == [12, 10]
    assert state.mob_projectile_directions[2, 0, 1].tolist() == [0, -1]
    assert state.mob_projectiles.type_id[2, 0, 1].item() == int(
        constants.ProjectileType.FIREBALL,
    )
    assert not state.mob_projectiles.mask[3, 0].any()


def test_firing_keeps_projectile_tensors_on_a_non_cpu_device() -> None:
    device = torch.device("meta")
    state = empty_state(num_envs=2, device=device)
    state.player_level[:] = torch.tensor([0, 1], device=device)

    mobs._fire_projectile(
        state,
        # Projectile positions and offsets are [batch, (row, column)].
        source=torch.empty((2, 2), dtype=torch.int32, device=device),
        toward=torch.empty((2, 2), dtype=torch.int32, device=device),
        species=torch.empty(2, dtype=torch.int64, device=device),
        firing=torch.empty(2, dtype=torch.bool, device=device),
    )

    assert state.mob_projectiles.position.device == device
    assert state.mob_projectiles.type_id.device == device
    assert state.mob_projectile_directions.device == device


def test_level_selection_preserves_a_non_cpu_device() -> None:
    device = torch.device("meta")
    field = torch.empty((2, 3, 4), device=device)
    level = torch.tensor([0, 2], device=device)

    selected = mobs._on_level(field, level)

    assert selected.shape == (2, 4)
    assert selected.device == device


def test_place_mob_updates_only_the_spawning_environment() -> None:
    state = _state(num_envs=2)
    state.melee_mobs.position[1, 0, 1] = torch.tensor([4, 5], dtype=torch.int32)
    state.melee_mobs.health[1, 0, 1] = 8.0
    state.mob_map[1, 0, 4, 5] = True

    mobs._place_mob(
        state,
        mobs=state.melee_mobs,
        slot=torch.ones(2, dtype=torch.int64),
        position=torch.tensor([[12, 13], [14, 15]], dtype=torch.int32),
        species=torch.tensor([2, 3], dtype=torch.int32),
        health=torch.tensor([6.0, 7.0]),
        spawning=torch.tensor([True, False]),
    )

    assert state.melee_mobs.position[:, 0, 1].tolist() == [[12, 13], [4, 5]]
    assert state.melee_mobs.health[:, 0, 1].tolist() == [6.0, 8.0]
    assert state.melee_mobs.type_id[:, 0, 1].tolist() == [2, 3]
    assert state.melee_mobs.mask[:, 0, 1].tolist() == [True, False]
    assert bool(state.mob_map[0, 0, 12, 13])
    assert bool(state.mob_map[1, 0, 4, 5])
    assert not bool(state.mob_map[1, 0, 14, 15])


def test_spawn_updates_world_on_its_device() -> None:
    device = torch.device("meta")
    state = empty_state(num_envs=2, device=device)
    state.player_level[:] = torch.tensor([0, 1], device=device)
    mobs._place_mob(
        state,
        mobs=state.melee_mobs,
        slot=torch.zeros(2, dtype=torch.int64, device=device),
        position=torch.tensor([[10, 11], [12, 13]], dtype=torch.int32, device=device),
        species=torch.tensor([1, 2], dtype=torch.int32, device=device),
        health=torch.tensor([5.0, 6.0], device=device),
        spawning=torch.tensor([True, False], device=device),
    )

    assert state.mob_map.device == device


def test_sample_position_handles_empty_rows_and_uses_its_generator() -> None:
    room = torch.tensor(
        [[[False, False, False], [False, False, True]], [[False] * 3] * 2],
    )
    actual = _seed(41)
    expected = _seed(41)
    weights = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0] * 6])
    flat = torch.multinomial(weights, 1, generator=expected).squeeze(-1)

    position = mobs._sample_position(room, generator=actual)

    assert position.tolist() == [
        [flat[0].item() // 3, flat[0].item() % 3],
        [flat[1].item() // 3, flat[1].item() % 3],
    ]
    assert actual.get_state().equal(expected.get_state())


def test_step_toward_player_tie_uses_rows_at_exactly_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = empty_state(num_envs=2, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([[10, 10], [10, 10]], dtype=torch.int32)

    def half_coin(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full((size,), 0.5, device=device)

    monkeypatch.setattr(torch, "rand", half_coin)
    step = mobs._step_toward_player(
        state,
        torch.tensor([[9, 9], [11, 11]], dtype=torch.int32),
        generator=_seed(47),
    )

    assert step.tolist() == [[1, 0], [-1, 0]]


def test_step_toward_player_follows_the_larger_axis() -> None:
    state = empty_state(num_envs=5, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([[10, 10]] * 5, dtype=torch.int32)
    position = torch.tensor(
        [[8, 11], [11, 8], [7, 8], [8, 7], [8, 8]],
        dtype=torch.int32,
    )

    step = mobs._step_toward_player(state, position, generator=_seed(23))

    assert step[:4].tolist() == [[1, 0], [0, 1], [1, 0], [0, 1]]
    assert step[4].abs().sum().item() == 1


def test_random_step_uses_its_generator_for_each_move_set() -> None:
    device = torch.device("cpu")
    actual = torch.Generator().manual_seed(37)
    expected = torch.Generator().manual_seed(37)

    for moves, offsets in (
        (4, constants.CLOSE_BLOCKS[:4]),
        (8, constants.DIRECTIONS[1:9]),
    ):
        choices = torch.randint(0, moves, (4,), generator=expected, device=device)
        observed = mobs._random_step(4, device, actual, moves=moves)
        assert torch.equal(observed, offsets[choices])
        assert torch.equal(actual.get_state(), expected.get_state())


def test_archer_firing_boundaries_and_blocked_retreat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=6)
    state.ranged_mobs.mask[:, 0, 0] = True
    state.ranged_mobs.position[:, 0, 0] = torch.tensor(
        [[10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 14]],
        dtype=torch.int32,
    )
    state.ranged_mobs.attack_cooldown[5, 0, 0] = 1
    state.mob_map[
        torch.arange(6),
        0,
        10,
        torch.tensor([12, 13, 14, 15, 16, 14]),
    ] = True
    state.map[0, 0, 10, 13] = int(BlockType.STONE)
    state.map[1, 0, 10, 14] = int(BlockType.STONE)

    def no_wander(
        size: tuple[int, ...] | int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del generator
        shape = (size,) if isinstance(size, int) else size
        return torch.full(shape, 0.99, dtype=dtype or torch.float32, device=device)

    def choose_up(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del low, generator
        return torch.full(
            size,
            min(2, high - 1),
            device=device,
            dtype=dtype or torch.int64,
        )

    monkeypatch.setattr(torch, "rand", no_wander)
    monkeypatch.setattr(torch, "randint", choose_up)
    state = mobs._update_ranged(state, generator=_seed(19))

    assert state.mob_projectiles.mask[:, 0, 0].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert state.mob_projectiles.position[:, 0, 0].tolist()[:4] == [
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
    ]
    assert state.ranged_mobs.attack_cooldown[:, 0, 0].tolist() == [4, 4, 4, 4, -1, 0]
    assert state.ranged_mobs.position[4, 0, 0].tolist() == [10, 15]
    assert state.ranged_mobs.position[5, 0, 0].tolist() == [9, 14]


def test_ranged_mob_uses_class_collision_to_decide_a_close_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.ranged_mobs.mask[0, 0, 0] = True
    state.ranged_mobs.position[0, 0, 0] = torch.tensor([10, 13], dtype=torch.int32)
    state.mob_map[0, 0, 10, 13] = True
    state.map[0, 0, 10, 14] = int(BlockType.WATER)

    def no_wander(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full((size,), 0.99, device=device)

    monkeypatch.setattr(torch, "rand", no_wander)
    state = mobs._update_ranged(state, generator=_seed(59))

    assert state.mob_projectiles.mask[0, 0, 0].item() is True
    assert state.ranged_mobs.position[0, 0, 0].tolist() == [10, 13]


def test_ranged_wander_threshold_includes_exactly_point_eight_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.ranged_mobs.mask[0, 0, 0] = True
    state.ranged_mobs.position[0, 0, 0] = torch.tensor([10, 17], dtype=torch.int32)

    def rand(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full((size,), 0.85, device=device)

    def choose_up(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del low, high, generator
        return torch.full(size, 2, device=device)

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", choose_up)
    state = mobs._update_ranged(state, generator=_seed(67))

    assert state.ranged_mobs.position[0, 0, 0].tolist() == [9, 17]


def test_ranged_mob_despawns_after_leaving_its_range() -> None:
    state = _state(num_envs=1)
    state.ranged_mobs.mask[0, 0, 0] = True
    state.ranged_mobs.position[0, 0, 0] = torch.tensor([10, 24], dtype=torch.int32)
    state.mob_map[0, 0, 10, 24] = True

    state = mobs._update_ranged(state, generator=_seed(61))

    assert state.ranged_mobs.mask[0, 0, 0].item() is False


def test_projectile_kills_clear_occupancy_for_monsters_and_passives() -> None:
    state = _state(num_envs=2)
    positions = torch.tensor([[10, 11], [10, 11]], dtype=torch.int32)
    state.melee_mobs.mask[0, 0, 0] = True
    state.melee_mobs.health[0, 0, 0] = 1.0
    state.melee_mobs.position[0, 0, 0] = positions[0]
    state.mob_map[0, 0, 10, 11] = True
    state.passive_mobs.mask[1, 0, 0] = True
    state.passive_mobs.health[1, 0, 0] = 1.0
    state.passive_mobs.position[1, 0, 0] = positions[1]
    state.mob_map[1, 0, 10, 11] = True

    state, struck = mobs._projectile_hits_tile(
        state,
        target=positions,
        damage=torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    assert struck.tolist() == [True, True]
    assert not state.melee_mobs.mask[0, 0, 0]
    assert not state.passive_mobs.mask[1, 0, 0]
    assert state.monsters_killed[:, 0].tolist() == [1, 0]
    assert not bool(state.mob_map[:, 0, 10, 11].any())


def test_projectile_hit_uses_each_mob_class_and_unlock_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=3)
    position = torch.tensor([[10, 11], [10, 11], [10, 11]], dtype=torch.int32)
    groups = (state.melee_mobs, state.passive_mobs, state.ranged_mobs)
    for row, group in enumerate(groups):
        group.mask[row, 0, 0] = True
        group.health[row, 0, 0] = 1.0
        group.position[row, 0, 0] = position[row]
        group.type_id[row, 0, 0] = 0
        state.mob_map[row, 0, 10, 11] = True

    calls: list[tuple[int, tuple[bool, ...]]] = []
    real_attack = mechanics.attack_mob_class

    def attack_mob_class(
        attack_state: EnvState,
        group: Mobs,
        *,
        position: torch.Tensor,
        damage: torch.Tensor,
        mob_class: int,
        can_unlock: torch.Tensor,
    ) -> tuple[Mobs, torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append((mob_class, tuple(bool(flag) for flag in can_unlock)))
        return real_attack(
            attack_state,
            group,
            position=position,
            damage=damage,
            mob_class=mob_class,
            can_unlock=can_unlock,
        )

    monkeypatch.setattr(mechanics, "attack_mob_class", attack_mob_class)
    state, struck = mobs._projectile_hits_tile(
        state,
        target=position,
        damage=torch.tensor([[5.0, 0.0, 0.0]] * 3),
    )

    assert calls == [
        (1, (True, True, True)),
        (0, (False, False, False)),
        (2, (True, True, True)),
    ]
    assert struck.tolist() == [True, True, True]
    assert state.melee_mobs.mask[0, 0, 0].item() is False
    assert state.passive_mobs.mask[1, 0, 0].item() is False
    assert state.ranged_mobs.mask[2, 0, 0].item() is False
    assert state.monsters_killed[:, 0].tolist() == [1, 0, 1]
    assert state.achievements[:, int(Achievement.DEFEAT_ZOMBIE)].tolist() == [
        True,
        False,
        False,
    ]
    assert state.achievements[:, int(Achievement.EAT_COW)].tolist() == [
        False,
        False,
        False,
    ]
    assert state.achievements[:, int(Achievement.DEFEAT_SKELETON)].tolist() == [
        False,
        False,
        True,
    ]
    assert state.mob_map[:, 0, 10, 11].tolist() == [False, False, False]


def test_projectile_damage_applies_bow_and_spell_scaling() -> None:
    state = _state(num_envs=6)
    state.player_dexterity[:] = 3
    state.player_intelligence[:] = 3
    state.bow_enchantment[:] = torch.tensor([1, 2, 0, 0, 0, 0])
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 100.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor(
        [[10, 11]] * 6,
        dtype=torch.int32,
    )
    species = torch.tensor([0, 4, 2, 3, 1, 1], dtype=torch.int32)

    state, hits = mobs._strike_with_projectile(
        state,
        species=species,
        at=(torch.tensor([[10, 11]] * 6), torch.tensor([[10, 12]] * 6)),
        alive=torch.tensor([True, True, True, True, False, True]),
    )

    assert hits.tolist() == [True, True, True, True, False, True]
    assert torch.equal(
        state.melee_mobs.health[:, 0, 0],
        torch.tensor([95.8, 89.5, 94.0, 94.0, 100.0, 96.0]),
    )


def test_relocating_despawns_only_at_the_distance_boundary() -> None:
    device = torch.device("cpu")
    group = Mobs.empty(num_envs=2, num_levels=2, num_slots=2, device=device)
    old = torch.tensor([[10, 23], [10, 24]], dtype=torch.int32)
    new = torch.tensor([[10, 24], [10, 25]], dtype=torch.int32)
    group.mask[:, 0, 0] = True
    group.position[:, 0, 0] = old
    state = empty_state(num_envs=2, device=device)
    state.player_level.zero_()
    state.player_position[:] = torch.tensor([[10, 10], [10, 10]], dtype=torch.int32)
    state.melee_mobs = group
    state.mob_map.zero_()
    state.mob_map[0, 0, 10, 23] = True
    state.mob_map[1, 0, 10, 24] = True

    mobs._relocate(
        state,
        mobs=state.melee_mobs,
        slot=0,
        old=old,
        new=new,
        cooldown=torch.tensor([4, 5], dtype=torch.int32),
        despawns=torch.ones(2, dtype=torch.bool),
    )

    assert state.melee_mobs.mask[:, 0, 0].tolist() == [True, False]
    assert state.mob_map[:, 0, 10, 23].tolist() == [False, False]
    assert state.mob_map[0, 0, 10, 24]
    assert not state.mob_map[1, 0, 10, 25]
    assert state.melee_mobs.attack_cooldown[:, 0, 0].tolist() == [4, 5]


def test_passive_mob_uses_its_class_collision_on_water(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.player_level[:] = 3
    state.passive_mobs.mask[0, 3, 0] = True
    state.passive_mobs.position[0, 3, 0] = torch.tensor([10, 14], dtype=torch.int32)
    state.mob_map[0, 3, 10, 14] = True
    state.map[0, 3, 10, 15] = int(BlockType.WATER)

    def step(
        num_envs: int,
        device: torch.device,
        generator: torch.Generator | None,
        *,
        moves: int,
    ) -> torch.Tensor:
        del generator
        assert moves == 8
        return torch.tensor([[0, 1]] * num_envs, dtype=torch.int32, device=device)

    monkeypatch.setattr(mobs, "_random_step", step)
    state = mobs._update_passive(state, generator=_seed())

    assert state.passive_mobs.position[0, 3, 0].tolist() == [10, 14]
    assert state.mob_map[0, 3, 10, 14]
    assert not state.mob_map[0, 3, 10, 15]


def test_ranged_mob_uses_its_collision_class_for_the_final_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.ranged_mobs.mask[0, 0, 0] = True
    state.ranged_mobs.position[0, 0, 0] = torch.tensor([10, 16], dtype=torch.int32)
    state.mob_map[0, 0, 10, 16] = True
    state.map[0, 0, 10, 15] = int(BlockType.WATER)

    def toward(
        state: EnvState,
        position: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del generator
        return torch.tensor([[0, -1]], dtype=position.dtype, device=state.device)

    def wander(
        num_envs: int,
        device: torch.device,
        generator: torch.Generator | None,
        *,
        moves: int,
    ) -> torch.Tensor:
        del generator
        assert moves == 4
        return torch.tensor([[0, 1]] * num_envs, dtype=torch.int32, device=device)

    def no_wander(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full((size,), 0.99, device=device)

    monkeypatch.setattr(mobs, "_step_toward_player", toward)
    monkeypatch.setattr(mobs, "_random_step", wander)
    monkeypatch.setattr(torch, "rand", no_wander)
    state = mobs._update_ranged(state, generator=_seed())

    assert state.ranged_mobs.position[0, 0, 0].tolist() == [10, 16]
    assert not state.mob_projectiles.mask[0, 0].any()
    assert not state.mob_map[0, 0, 10, 15]


def test_ranged_firing_routes_gap_three_and_four_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=2)
    state.ranged_mobs.mask[:, 0, 0] = True
    state.ranged_mobs.position[:, 0, 0] = torch.tensor(
        [[10, 13], [10, 14]],
        dtype=torch.int32,
    )
    state.mob_map[0, 0, 10, 13] = True
    state.mob_map[1, 0, 10, 14] = True
    state.map[0, 0, 10, 14] = int(BlockType.STONE)

    def toward(
        state: EnvState,
        position: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del generator
        return torch.tensor(
            [[0, -1]] * state.num_envs,
            dtype=position.dtype,
            device=state.device,
        )

    def wander(
        num_envs: int,
        device: torch.device,
        generator: torch.Generator | None,
        *,
        moves: int,
    ) -> torch.Tensor:
        del generator
        assert moves == 4
        return torch.tensor([[0, 1]] * num_envs, dtype=torch.int32, device=device)

    def no_wander(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full((size,), 0.99, device=device)

    monkeypatch.setattr(mobs, "_step_toward_player", toward)
    monkeypatch.setattr(mobs, "_random_step", wander)
    monkeypatch.setattr(torch, "rand", no_wander)
    state = mobs._update_ranged(state, generator=_seed())

    assert state.mob_projectiles.mask[:, 0, 0].tolist() == [True, True]
    assert state.ranged_mobs.attack_cooldown[:, 0, 0].tolist() == [4, 4]


def test_water_only_ranged_spawns_respect_occupancy_and_environment_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=3)
    state.player_level[:] = 5
    state.map[:] = int(BlockType.STONE)
    state.map[:, 5, 10, 20] = int(BlockType.GRASS)
    state.map[0, 5, 10, 21] = int(BlockType.WATER)
    state.map[1, 5, 10, 21] = int(BlockType.WATER)
    state.ranged_mobs.mask[1, 5, 0] = True
    state.ranged_mobs.type_id[1, 5, 0] = 5
    state.ranged_mobs.position[1, 5, 0] = torch.tensor([10, 21], dtype=torch.int32)
    state.mob_map[1, 5, 10, 21] = True
    draws = iter((torch.full((3,), 0.99), torch.full((3,), 0.99), torch.zeros(3)))

    def pinned_rand(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del generator
        values = next(draws)
        assert values.shape == (size,)
        return values.to(device=device)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed())

    assert state.ranged_mobs.mask[:, 5].tolist() == [
        [True, False],
        [True, False],
        [False, False],
    ]
    assert state.ranged_mobs.position[0, 5, 0].tolist() == [10, 21]


def test_water_only_ranged_spawns_stop_at_the_despawn_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.player_level[:] = 5
    state.map[:] = int(BlockType.STONE)
    state.map[0, 5, 10, 20] = int(BlockType.GRASS)
    state.map[0, 5, 10, 24] = int(BlockType.WATER)
    draws = iter((torch.full((1,), 0.99), torch.full((1,), 0.99), torch.zeros(1)))

    def pinned_rand(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del generator
        values = next(draws)
        assert values.shape == (size,)
        return values.to(device=device)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed())

    assert not state.ranged_mobs.mask[0, 5].any()


def test_boss_wave_spawn_multiplier_scales_the_melee_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=1)
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.monsters_killed[0, -1] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state.boss_timesteps_to_spawn_this_round[:] = 1
    state.map[:] = int(BlockType.STONE)
    state.map[0, -1, 10, 12] = int(BlockType.GRAVE)
    monkeypatch.setattr(
        constants,
        "FLOOR_MOB_SPAWN_CHANCE",
        torch.full_like(constants.FLOOR_MOB_SPAWN_CHANCE, 0.0005),
    )
    draws = iter(
        (
            torch.full((1,), 0.99),
            torch.full((1,), 0.50025),
            torch.full((1,), 0.99),
        ),
    )

    def pinned_rand(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del generator
        values = next(draws)
        assert values.shape == (size,)
        return values.to(device=device)

    monkeypatch.setattr(torch, "rand", pinned_rand)
    state = mobs.spawn_mobs(state, generator=_seed())

    assert not state.melee_mobs.mask[0, -1].any()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
