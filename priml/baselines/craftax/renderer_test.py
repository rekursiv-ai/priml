"""Tests for the symbolic observation."""

from __future__ import annotations

import torch

from priml.baselines.craftax import constants, renderer, world_gen
from priml.baselines.craftax.conftest import reference, requires_craftax
from priml.baselines.craftax.constants import Action, BlockType, ItemType
from priml.baselines.craftax.state import EnvState, empty_state


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
    assert renderer.OBSERVATION_SIZE == 8_268
    assert renderer.render(_state()).shape == (2, 8_268)


def test_every_value_is_finite_and_bounded() -> None:
    observation = renderer.render(_state())
    assert bool(torch.isfinite(observation).all())
    assert float(observation.min()) >= 0.0
    assert float(observation.max()) <= 1.0


def test_the_view_follows_the_player() -> None:
    near = _state()
    near.map[:, 0, 20, 21] = int(BlockType.STONE)
    far = _state()
    far.map[:, 0, 40, 40] = int(BlockType.STONE)
    assert not torch.equal(renderer.render(near), renderer.render(far))


def test_darkness_hides_the_world() -> None:
    # An unlit tile shows nothing at all, which is what makes a torch matter.
    lit = _state()
    lit.map[:, 0, 20, 21] = int(BlockType.STONE)
    dark = _state()
    dark.map[:, 0, 20, 21] = int(BlockType.STONE)
    dark.light_map[:] = 0.0

    assert not torch.equal(renderer.render(lit), renderer.render(dark))
    # With the whole floor dark, only the light channel and the player's own
    # scalars carry information.
    view_width = renderer.OBSERVATION_SIZE - constants.INVENTORY_OBS_SIZE
    assert float(renderer.render(dark)[:, :view_width].sum()) == 0.0


def test_the_world_edge_is_visible_as_out_of_bounds() -> None:
    # Padding with zero would read as a legitimate block; the agent must be
    # able to see where the map stops.
    middle = _state()
    corner = _state()
    corner.player_position[:] = torch.tensor([0, 0], dtype=torch.int32)
    assert not torch.equal(renderer.render(middle), renderer.render(corner))


def test_inventory_shows_up_in_the_observation() -> None:
    empty = renderer.render(_state())
    stocked = _state()
    stocked.inventory.wood[:] = 4
    assert not torch.equal(empty, renderer.render(stocked))


def test_counts_are_compressed_so_early_gains_matter_most() -> None:
    # The first log should move the observation more than the ninetieth.
    def wood(amount: int) -> torch.Tensor:
        state = _state()
        state.inventory.wood[:] = amount
        return renderer.render(state)

    early = (wood(1) - wood(0)).abs().sum()
    late = (wood(90) - wood(89)).abs().sum()
    assert float(early) > float(late)


def test_a_visible_creature_appears_in_the_view() -> None:
    plain = renderer.render(_state())
    haunted = _state()
    haunted.melee_mobs.mask[:, 0, 0] = True
    haunted.melee_mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
    assert not torch.equal(plain, renderer.render(haunted))


def test_a_distant_creature_is_not_visible() -> None:
    """A creature outside the window leaves no mark on the view.

    Only the view is compared: a live creature anywhere on the floor also
    shields the boss, and that is reported among the player's scalars, so
    comparing whole observations would conflate the two.
    """
    view_width = renderer.OBSERVATION_SIZE - constants.INVENTORY_OBS_SIZE
    plain = renderer.render(_state())
    distant = _state()
    distant.melee_mobs.mask[:, 0, 0] = True
    distant.melee_mobs.position[:, 0, 0] = torch.tensor([40, 40], dtype=torch.int32)
    assert torch.equal(
        plain[:, :view_width],
        renderer.render(distant)[:, :view_width],
    )


def test_creature_classes_are_distinguishable() -> None:
    def creature(field: str) -> torch.Tensor:
        state = _state()
        mobs = getattr(state, field)
        mobs.mask[:, 0, 0] = True
        mobs.position[:, 0, 0] = torch.tensor([20, 22], dtype=torch.int32)
        return renderer.render(state)

    assert not torch.equal(creature("melee_mobs"), creature("passive_mobs"))
    assert not torch.equal(creature("melee_mobs"), creature("ranged_mobs"))


def test_the_facing_direction_is_reported() -> None:
    facing_up = _state()
    facing_down = _state()
    facing_down.player_direction[:] = int(Action.DOWN)
    assert not torch.equal(renderer.render(facing_up), renderer.render(facing_down))


def test_items_are_reported_alongside_blocks() -> None:
    bare = renderer.render(_state())
    laddered = _state()
    laddered.item_map[:, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    assert not torch.equal(bare, renderer.render(laddered))


def test_each_environment_renders_its_own_world() -> None:
    state = world_gen.generate_world(
        num_envs=3,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    observation = renderer.render(state)
    assert len({tuple(row.tolist()) for row in observation}) == 3


@requires_craftax
def test_the_width_matches_the_reference_environment() -> None:
    upstream = reference("craftax.envs.craftax_symbolic_env")
    assert (
        upstream.get_flat_map_obs_shape() + upstream.get_inventory_obs_shape()
        == renderer.OBSERVATION_SIZE
    )


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
