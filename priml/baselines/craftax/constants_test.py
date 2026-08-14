"""Tests that the ported constants match the reference tables exactly.

The tables are the game's rules. A transcription slip in one of them changes
what the environment IS, and would show up only as a slightly wrong score much
later, so each is compared elementwise against the reference package.
"""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax import constants
from priml.baselines.craftax.conftest import (
    as_tensor,
    reference,
    requires_craftax,
)


def test_reward_table_sums_to_the_scoring_denominator() -> None:
    # The normalized score divides by this total, so a table that does not sum
    # to it would silently rescale every reported result.
    assert float(constants.ACHIEVEMENT_REWARD.sum()) == constants.REWARD_CEILING
    assert len(constants.ACHIEVEMENT_REWARD) == len(constants.Achievement)


def test_observation_size_matches_the_declared_layout() -> None:
    rows, columns = constants.OBS_DIM
    channels = (
        len(constants.BlockType)
        + len(constants.ItemType)
        + 5 * 8  # five mob classes, eight species each
        + 1  # light
    )
    assert rows * columns * channels + constants.INVENTORY_OBS_SIZE == 8_268


def test_movement_directions_are_indexed_by_action() -> None:
    assert constants.DIRECTIONS[constants.Action.LEFT].tolist() == [0, -1]
    assert constants.DIRECTIONS[constants.Action.RIGHT].tolist() == [0, 1]
    assert constants.DIRECTIONS[constants.Action.UP].tolist() == [-1, 0]
    assert constants.DIRECTIONS[constants.Action.DOWN].tolist() == [1, 0]
    # A non-movement action must index a zero row rather than fall off the end.
    assert constants.DIRECTIONS[constants.Action.DO].tolist() == [0, 0]


def test_torch_light_map_peaks_at_its_own_tile_and_falls_to_zero() -> None:
    light = constants.TORCH_LIGHT_MAP
    assert light.shape == (9, 9)
    assert float(light[4, 4]) == pytest.approx(1.0)
    assert float(light[0, 0]) == pytest.approx(0.0)
    assert torch.equal(light, light.flip(0))
    assert torch.equal(light, light.flip(1))


@requires_craftax
@pytest.mark.parametrize(
    ("ported", "upstream_name"),
    [
        ("SOLID_BLOCK", "SOLID_BLOCK_MAPPING"),
        ("CAN_PLACE_ITEM_ON", "CAN_PLACE_ITEM_MAPPING"),
        ("FLOOR_MOB_TYPE", "FLOOR_MOB_MAPPING"),
        ("FLOOR_MOB_SPAWN_CHANCE", "FLOOR_MOB_SPAWN_CHANCE"),
        ("MOB_COLLIDES_WITH", "MOB_TYPE_COLLISION_MAPPING"),
        ("MOB_DAMAGE", "MOB_TYPE_DAMAGE_MAPPING"),
        ("MOB_HEALTH", "MOB_TYPE_HEALTH_MAPPING"),
        ("MOB_DEFENSE", "MOB_TYPE_DEFENSE_MAPPING"),
        ("RANGED_MOB_PROJECTILE", "RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING"),
        ("ACHIEVEMENT_REWARD", "ACHIEVEMENT_REWARD_MAP"),
        ("LEVEL_ACHIEVEMENT", "LEVEL_ACHIEVEMENT_MAP"),
        ("MOB_ACHIEVEMENT", "MOB_ACHIEVEMENT_MAP"),
        ("CLOSE_BLOCKS", "CLOSE_BLOCKS"),
    ],
)
def test_table_matches_reference(ported: str, upstream_name: str) -> None:
    upstream = reference("craftax.constants")
    expected = as_tensor(getattr(upstream, upstream_name))
    actual = getattr(constants, ported)
    assert actual.shape == expected.shape, ported
    assert torch.equal(actual.to(expected.dtype), expected), ported


@requires_craftax
def test_enumerations_match_reference_values() -> None:
    upstream = reference("craftax.constants")
    for ported, expected in (
        (constants.BlockType, upstream.BlockType),
        (constants.ItemType, upstream.ItemType),
        (constants.Action, upstream.Action),
        (constants.Achievement, upstream.Achievement),
        (constants.ProjectileType, upstream.ProjectileType),
    ):
        assert {member.name: int(member) for member in ported} == {
            member.name: member.value for member in expected
        }


@requires_craftax
def test_torch_light_map_matches_reference_to_one_ulp() -> None:
    """The light map agrees with the reference except in its last bit.

    The reference's square root truncates where IEEE-754 rounds to nearest, so
    40 off-axis entries differ by one ulp. Reproducing that would require a
    deliberately less accurate square root; the tolerance here is the size of
    that rounding step, not a slackened comparison.
    """
    upstream = reference("craftax.constants")
    expected = as_tensor(upstream.TORCH_LIGHT_MAP)
    difference = (constants.TORCH_LIGHT_MAP - expected).abs()
    assert float(difference.max()) <= 2.0**-23
    # The light threshold the renderer compares against is 0.05, so a one-ulp
    # difference can never flip a tile between lit and dark.
    assert torch.equal(
        constants.TORCH_LIGHT_MAP > 0.05,
        expected > 0.05,
    )


@requires_craftax
def test_movement_directions_match_the_reference_where_it_is_defined() -> None:
    """The steps agree; ours simply spans the whole action space.

    The reference table stops short because its indices are traced, so an
    out-of-range lookup is silently clamped. Torch raises instead, so the
    table here covers every action and the extra rows are zero.
    """
    upstream = reference("craftax.constants")
    expected = as_tensor(upstream.DIRECTIONS)
    actual = constants.DIRECTIONS.to(expected.dtype)
    assert len(actual) == len(constants.Action)
    assert torch.equal(actual[: len(expected)], expected)
    assert int(actual[len(expected) :].abs().sum()) == 0


@requires_craftax
def test_scalar_rules_match_reference() -> None:
    upstream = reference("craftax.constants")
    state = reference("craftax.craftax_state")
    EnvParams, StaticEnvParams = state.EnvParams, state.StaticEnvParams
    assert constants.OBS_DIM == upstream.OBS_DIM
    assert constants.MAX_OBS_DIM == upstream.MAX_OBS_DIM
    assert (
        constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
        == upstream.MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    assert constants.BOSS_FIGHT_EXTRA_DAMAGE == upstream.BOSS_FIGHT_EXTRA_DAMAGE
    assert constants.BOSS_FIGHT_SPAWN_TURNS == upstream.BOSS_FIGHT_SPAWN_TURNS

    static, params = StaticEnvParams(), EnvParams()
    assert static.map_size == constants.MAP_SIZE
    assert static.num_levels == constants.NUM_LEVELS
    assert static.max_melee_mobs == constants.MAX_MELEE_MOBS
    assert static.max_passive_mobs == constants.MAX_PASSIVE_MOBS
    assert static.max_ranged_mobs == constants.MAX_RANGED_MOBS
    assert static.max_mob_projectiles == constants.MAX_MOB_PROJECTILES
    assert static.max_player_projectiles == constants.MAX_PLAYER_PROJECTILES
    assert static.max_growing_plants == constants.MAX_GROWING_PLANTS
    assert params.max_timesteps == constants.MAX_TIMESTEPS
    assert params.day_length == constants.DAY_LENGTH
    assert params.mob_despawn_distance == constants.MOB_DESPAWN_DISTANCE
    assert params.max_attribute == constants.MAX_ATTRIBUTE


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
