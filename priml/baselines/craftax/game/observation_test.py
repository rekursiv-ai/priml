"""Tests for the symbolic observation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.conftest import (
    generated_world,
    reference,
    requires_craftax,
)
from priml.baselines.craftax.game import constants, observation
from priml.baselines.craftax.game.constants import Action, BlockType, ItemType
from priml.baselines.craftax.game.indexing import batch_rows, local_view
from priml.baselines.craftax.game.state import EnvState, Mobs, empty_state


if TYPE_CHECKING:
    from collections.abc import Callable


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([20, 20], dtype=torch.int32)
    state.player_direction[:] = int(Action.UP)
    state.map[:] = int(BlockType.GRASS)
    state.light_map[:] = 1.0
    state.player_health[:] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        getattr(state, meter)[:] = 9
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    return state


def test_the_observation_has_the_published_width() -> None:
    # This width is part of the benchmark's contract, not an implementation
    # detail: a model trained against a different one is not comparable.
    assert observation.observation_size() == 8_268
    assert observation.render(_state()).shape == (2, 8_268)


def test_every_value_is_finite_and_bounded() -> None:
    rendered = observation.render(_state())
    assert bool(torch.isfinite(rendered).all())
    assert float(rendered.min()) >= 0.0
    assert float(rendered.max()) <= 1.0


def test_the_view_follows_the_player() -> None:
    near = _state()
    near.map[:, 0, 20, 21] = int(BlockType.STONE)
    far = _state()
    far.map[:, 0, 40, 40] = int(BlockType.STONE)
    assert not torch.equal(observation.render(near), observation.render(far))


def test_light_threshold_is_strictly_greater_than_point_zero_five() -> None:
    state = _state(num_envs=2)
    state.light_map[:, 0, 20, 20] = 0.05
    state.light_map[:, 0, 20, 21] = 0.0501

    rendered = observation.render(state)
    center_tile = (4 * constants.OBS_DIM[1] + 5) * observation.CHANNELS_PER_TILE
    adjacent_tile = center_tile + observation.CHANNELS_PER_TILE
    light_channel = observation.CHANNELS_PER_TILE - 1
    assert torch.equal(rendered[:, center_tile + light_channel], torch.zeros(2))
    assert torch.equal(rendered[:, adjacent_tile + light_channel], torch.ones(2))


def test_darkness_hides_the_world() -> None:
    # An unlit tile shows nothing at all, which is what makes a torch matter.
    lit = _state()
    lit.map[:, 0, 20, 21] = int(BlockType.STONE)
    dark = _state()
    dark.map[:, 0, 20, 21] = int(BlockType.STONE)
    dark.light_map[:] = 0.0

    assert not torch.equal(observation.render(lit), observation.render(dark))
    # With the whole floor dark, only the light channel and the player's own
    # scalars carry information.
    view_width = observation.observation_size() - constants.INVENTORY_OBS_SIZE
    assert float(observation.render(dark)[:, :view_width].sum()) == 0.0


def test_view_padding_uses_out_of_bounds_block_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(num_envs=2)
    state.player_position[:] = 0
    original = local_view
    calls = 0

    def force_light(
        grid: Tensor,
        centers: Tensor,
        size: tuple[int, int],
        *,
        outside: float = 0.0,
    ) -> Tensor:
        nonlocal calls
        result = original(grid, centers, size, outside=outside)
        calls += 1
        return torch.ones_like(result) if calls == 3 else result

    monkeypatch.setattr(
        "priml.baselines.craftax.game.observation.local_view",
        force_light,
    )
    rendered = observation.render(state)
    tile_channels = observation.CHANNELS_PER_TILE

    assert torch.equal(rendered[:, tile_channels - 1], torch.ones(2))
    assert torch.equal(
        rendered[:, int(BlockType.OUT_OF_BOUNDS)],
        torch.ones(2),
    )


def test_the_world_edge_is_visible_as_out_of_bounds() -> None:
    # Padding with zero would read as a legitimate block; the agent must be
    # able to see where the map stops.
    middle = _state()
    corner = _state()
    corner.player_position[:] = torch.tensor([0, 0], dtype=torch.int32)
    assert not torch.equal(observation.render(middle), observation.render(corner))


def test_inventory_shows_up_in_the_observation() -> None:
    empty = observation.render(_state())
    stocked = _state()
    stocked.inventory.wood[:] = 4
    assert not torch.equal(empty, observation.render(stocked))


def test_counts_are_compressed_so_early_gains_matter_most() -> None:
    # The first log should move the observation more than the ninetieth.
    def wood(amount: int) -> Tensor:
        state = _state()
        state.inventory.wood[:] = amount
        return observation.render(state)

    early = (wood(1) - wood(0)).abs().sum()
    late = (wood(90) - wood(89)).abs().sum()
    assert float(early) > float(late)


def test_a_visible_creature_appears_in_the_view() -> None:
    plain = observation.render(_state())
    haunted = _state()
    haunted.melee_mobs.mask[:, 0, 0] = True
    haunted.melee_mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
    assert not torch.equal(plain, observation.render(haunted))


def test_a_distant_creature_is_not_visible() -> None:
    """A creature outside the window leaves no mark on the view.

    Only the view is compared: a live creature anywhere on the floor also
    shields the boss, and that is reported among the player's scalars, so
    comparing whole observations would conflate the two.
    """
    view_width = observation.observation_size() - constants.INVENTORY_OBS_SIZE
    plain = observation.render(_state())
    distant = _state()
    distant.melee_mobs.mask[:, 0, 0] = True
    distant.melee_mobs.position[:, 0, 0] = torch.tensor([40, 40], dtype=torch.int32)
    assert torch.equal(
        plain[:, :view_width],
        observation.render(distant)[:, :view_width],
    )


def test_every_creature_class_and_species_uses_its_exact_plane() -> None:
    classes = (
        "melee_mobs",
        "passive_mobs",
        "ranged_mobs",
        "mob_projectiles",
        "player_projectiles",
    )
    tile = (4 * constants.OBS_DIM[1] + 5 + 2) * observation.CHANNELS_PER_TILE
    first_mob_channel = len(BlockType) + len(ItemType)
    for class_index, name in enumerate(classes):
        state = _state(num_envs=2)
        mobs = cast(Mobs, getattr(state, name))
        mobs.mask[:, 0, 0] = True
        mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
        mobs.type_id[:, 0, 0] = 7

        rendered = observation.render(state)
        expected = tile + first_mob_channel + class_index * 8 + 7
        assert torch.equal(rendered[:, expected], torch.ones(2))
        assert int(rendered[0, expected - 1]) == 0


def test_creatures_on_each_view_edge_are_visible() -> None:
    for position in ([16, 15], [24, 25]):
        state = _state(num_envs=2)
        state.player_projectiles.mask[:, 0, 0] = True
        state.player_projectiles.position[:, 0, 0] = torch.tensor(position)
        rendered = observation.render(state)
        row, column = position[0] - 16, position[1] - 15
        tile = (row * constants.OBS_DIM[1] + column) * observation.CHANNELS_PER_TILE
        channel = tile + len(BlockType) + len(ItemType) + 4 * 8
        assert torch.equal(rendered[:, channel], torch.ones(2))


def test_render_mob_tensors_use_the_state_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_device = torch.device("cpu")
    state = _state()
    factory_devices: list[object] = []
    factories: dict[str, Callable[..., Tensor]] = {
        "zeros": torch.zeros,
        "tensor": torch.tensor,
        "arange": torch.arange,
    }
    for name in ("zeros", "tensor", "arange"):
        factory = factories[name]

        def record_device(
            *args: object,
            factory: Callable[..., Tensor] = factory,
            **kwargs: object,
        ) -> Tensor:
            device = kwargs.get("device")
            factory_devices.append(device)
            captured_factory = factory
            del factory
            return captured_factory(*args, **kwargs)

        monkeypatch.setattr(torch, name, record_device)

    observation._render_mobs(state, view=(3, 5))

    assert factory_devices
    assert all(device == expected_device for device in factory_devices)


def test_player_rows_use_the_state_device(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    devices: list[object] = []

    def record_device(envs: int, device: torch.device) -> Tensor:
        devices.append(device)
        return batch_rows(envs, device)

    monkeypatch.setattr(
        "priml.baselines.craftax.game.observation.batch_rows",
        record_device,
    )

    observation._render_player(state)

    assert devices == [state.device]


def test_creatures_outside_each_view_edge_are_hidden() -> None:
    view_width = observation.observation_size() - constants.INVENTORY_OBS_SIZE
    for position in ([15, 20], [25, 20], [20, 14], [20, 26]):
        state = _state(num_envs=2)
        state.player_projectiles.mask[:, 0, 0] = True
        state.player_projectiles.position[:, 0, 0] = torch.tensor(position)
        rendered = observation.render(state)
        assert torch.equal(
            rendered[:, :view_width],
            observation.render(_state(num_envs=2))[:, :view_width],
        )


def test_creature_classes_are_distinguishable() -> None:
    def creature(field: str) -> Tensor:
        state = _state()
        mobs = cast(Mobs, getattr(state, field))
        mobs.mask[:, 0, 0] = True
        mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
        return observation.render(state)

    assert not torch.equal(creature("melee_mobs"), creature("passive_mobs"))
    assert not torch.equal(creature("melee_mobs"), creature("ranged_mobs"))


def test_melee_and_passive_use_their_reference_channels() -> None:
    melee = _state(num_envs=1)
    melee.melee_mobs.mask[:, 0, 0] = True
    melee.melee_mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
    passive = _state(num_envs=1)
    passive.passive_mobs.mask[:, 0, 0] = True
    passive.passive_mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
    tile_index = (
        constants.OBS_DIM[0] // 2 * constants.OBS_DIM[1] + constants.OBS_DIM[1] // 2 + 2
    ) * observation.CHANNELS_PER_TILE
    melee_channel = tile_index + len(BlockType) + len(ItemType)
    passive_channel = melee_channel + 8
    melee_render = observation.render(melee)[0]
    passive_render = observation.render(passive)[0]

    assert melee_render[melee_channel] == 1
    assert melee_render[passive_channel] == 0
    assert passive_render[melee_channel] == 0
    assert passive_render[passive_channel] == 1


def test_player_scalars_are_encoded_at_their_exact_values() -> None:
    state = _state(num_envs=2)
    inventory_fields = (
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapphire",
        "ruby",
        "sapling",
        "torches",
        "arrows",
    )
    for field_index, field in enumerate(inventory_fields, start=1):
        getattr(state.inventory, field)[:] = torch.tensor(
            [field_index, field_index + 1],
        )
    state.inventory.books[:] = torch.tensor([2, 4])
    state.inventory.pickaxe[:] = torch.tensor([4, 8])
    state.inventory.sword[:] = torch.tensor([8, 12])
    state.sword_enchantment[:] = torch.tensor([1, 2])
    state.bow_enchantment[:] = torch.tensor([2, 1])
    state.inventory.bow[:] = torch.tensor([1, 2])
    state.inventory.potions[:] = torch.tensor([[4] * 6, [9] * 6])
    state.player_health[:] = torch.tensor([3.0, 6.0])
    state.player_food[:] = torch.tensor([4, 7])
    state.player_drink[:] = torch.tensor([5, 8])
    state.player_energy[:] = torch.tensor([6, 9])
    state.player_mana[:] = torch.tensor([7, 10])
    state.player_xp[:] = torch.tensor([8, 11])
    state.player_dexterity[:] = torch.tensor([2, 3])
    state.player_strength[:] = torch.tensor([3, 4])
    state.player_intelligence[:] = torch.tensor([4, 5])
    state.player_direction[:] = torch.tensor([int(Action.UP), int(Action.LEFT)])
    state.inventory.armour[:] = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    state.armour_enchantments[:] = torch.tensor([[1, 2, 3, 0], [2, 1, 0, 3]])
    state.light_level[:] = torch.tensor([0.25, 0.75])
    state.is_sleeping[:] = torch.tensor([False, True])
    state.is_resting[:] = torch.tensor([True, False])
    state.learned_spells[:] = torch.tensor([[1, 0], [0, 1]])
    state.player_level[:] = torch.tensor([1, 2])
    state.monsters_killed[0, 1] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state.monsters_killed[1, 2] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL - 1
    state.boss_timesteps_to_spawn_this_round[:] = 1

    rendered = observation.render(state)
    view_width = observation.observation_size() - constants.INVENTORY_OBS_SIZE
    expected = torch.tensor(
        [
            [
                *([value**0.5 / 10 for value in range(1, 11)]),
                1.0,
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                *([0.2] * 6),
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.2,
                0.3,
                0.4,
                0.0,
                0.0,
                1.0,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                1.0,
                2.0,
                3.0,
                0.0,
                0.25,
                0.0,
                1.0,
                1.0,
                0.0,
                0.1,
                1.0,
                0.0,
            ],
            [
                *([value**0.5 / 10 for value in range(2, 12)]),
                2.0,
                2.0,
                3.0,
                2.0,
                1.0,
                2.0,
                *([0.3] * 6),
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                0.3,
                0.4,
                0.5,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.5,
                2.0,
                2.5,
                2.0,
                1.0,
                0.0,
                3.0,
                0.75,
                1.0,
                0.0,
                0.0,
                1.0,
                0.2,
                0.0,
                0.0,
            ],
        ],
    )
    expected[:, :10] = (
        torch.tensor(
            [list(range(1, 11)), list(range(2, 12))],
            dtype=torch.float32,
        ).sqrt()
        / 10
    )
    torch.testing.assert_close(
        rendered[:, view_width:],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_the_facing_direction_is_reported() -> None:
    facing_up = _state()
    facing_down = _state()
    facing_down.player_direction[:] = int(Action.DOWN)
    assert not torch.equal(
        observation.render(facing_up),
        observation.render(facing_down),
    )


def test_items_are_reported_alongside_blocks() -> None:
    bare = observation.render(_state())
    laddered = _state()
    laddered.item_map[:, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    assert not torch.equal(bare, observation.render(laddered))


def test_each_environment_renders_its_own_world() -> None:
    state = generated_world(num_envs=3, seed=0)
    rendered = observation.render(state)
    assert len({tuple(row.tolist()) for row in rendered}) == 3


@requires_craftax
@pytest.mark.compute_jax_jit
def test_the_width_matches_the_reference_environment() -> None:
    """The published width, read off a reference env built at minimum size.

    The window is 9x11 whatever the floor is, so the observation width is a
    property of the VOCABULARIES, not of the world: a 9x9 single-level map with
    one creature of each class reports the same 8_268 as the benchmark's 48x48
    nine-level one. Building the small one asserts that independence, where
    instantiating the full env would tie this test to a world size it is not
    about.

    The env is constructed rather than calling ``get_flat_map_obs_shape``
    directly because the shape a caller actually receives comes from
    ``observation_space``, and a reference change that moved the inventory
    block out of that space would leave the two module functions agreeing.

    The small env does NOT make this fast, and shrinking it further will not:
    construction is 3e-05s and the test still takes seconds, all of it
    importing the reference env module, whose module scope runs ~51 XLA
    compiles beyond the ``constants`` import the conftest warms. Hence the
    ``compute_jax_jit`` marker: the JIT is the cost, so it runs in the slow
    tier.
    """
    upstream = cast(
        _ReferenceModule,
        reference("craftax.envs.craftax_symbolic_env"),
    )
    smallest = upstream.StaticEnvParams(
        map_size=(9, 9),
        num_levels=1,
        max_melee_mobs=1,
        max_passive_mobs=1,
        max_ranged_mobs=1,
        max_mob_projectiles=1,
        max_player_projectiles=1,
        max_growing_plants=1,
    )
    environment = upstream.CraftaxSymbolicEnvNoAutoReset(smallest)
    space = environment.observation_space(environment.default_params)
    assert space.shape == (observation.observation_size(),)


class _Space(Protocol):
    shape: tuple[int, ...]


class _Environment(Protocol):
    default_params: object

    def observation_space(self, params: object) -> _Space: ...


class _ReferenceModule(Protocol):
    StaticEnvParams: Callable[..., object]
    CraftaxSymbolicEnvNoAutoReset: Callable[[object], _Environment]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
