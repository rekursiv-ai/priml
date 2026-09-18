"""Tests that the ported constants match the reference tables exactly.

The tables are the game's rules. A transcription slip in one of them changes
what the environment IS, and would show up only as a slightly wrong score much
later, so each is compared elementwise against the reference package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.conftest import (
    as_tensor,
    reference,
    requires_craftax,
)
from priml.baselines.craftax.game import constants


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


pytestmark = pytest.mark.usefixtures("warm_reference")


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
        + 5 * 8  # Five mob classes, eight species each.
        + 1  # `light`.
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
    upstream = cast(_ReferenceConstants, reference("craftax.constants"))
    expected = as_tensor(_reference_value(upstream, upstream_name))
    actual = _ported_tensor(ported)
    assert actual.shape == expected.shape, ported
    assert torch.equal(actual.to(expected.dtype), expected), ported


@requires_craftax
def test_enumerations_match_reference_values() -> None:
    upstream = cast(_ReferenceConstants, reference("craftax.constants"))
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
    upstream = cast(_ReferenceConstants, reference("craftax.constants"))
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
    upstream = cast(_ReferenceConstants, reference("craftax.constants"))
    expected = as_tensor(upstream.DIRECTIONS)
    actual = constants.DIRECTIONS.to(expected.dtype)
    assert len(actual) == len(constants.Action)
    assert torch.equal(actual[: len(expected)], expected)
    assert int(actual[len(expected) :].abs().sum()) == 0


@requires_craftax
def test_scalar_rules_match_reference() -> None:
    upstream = cast(_ReferenceConstants, reference("craftax.constants"))
    state = cast(_ReferenceState, reference("craftax.craftax_state"))
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


class _EnumType(Protocol):
    def __iter__(self) -> Iterator[_EnumMember]: ...


class _EnumMember(Protocol):
    name: str
    value: int


class _ReferenceConstants(Protocol):
    SOLID_BLOCK_MAPPING: object
    CAN_PLACE_ITEM_MAPPING: object
    FLOOR_MOB_MAPPING: object
    FLOOR_MOB_SPAWN_CHANCE: object
    MOB_TYPE_COLLISION_MAPPING: object
    MOB_TYPE_DAMAGE_MAPPING: object
    MOB_TYPE_HEALTH_MAPPING: object
    MOB_TYPE_DEFENSE_MAPPING: object
    RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING: object
    ACHIEVEMENT_REWARD_MAP: object
    LEVEL_ACHIEVEMENT_MAP: object
    MOB_ACHIEVEMENT_MAP: object
    CLOSE_BLOCKS: object
    BlockType: _EnumType
    ItemType: _EnumType
    Action: _EnumType
    Achievement: _EnumType
    ProjectileType: _EnumType
    TORCH_LIGHT_MAP: object
    DIRECTIONS: object
    OBS_DIM: tuple[int, int]
    MAX_OBS_DIM: int
    MONSTERS_KILLED_TO_CLEAR_LEVEL: int
    BOSS_FIGHT_EXTRA_DAMAGE: float
    BOSS_FIGHT_SPAWN_TURNS: int


class _StaticEnvParams(Protocol):
    map_size: int
    num_levels: int
    max_melee_mobs: int
    max_passive_mobs: int
    max_ranged_mobs: int
    max_mob_projectiles: int
    max_player_projectiles: int
    max_growing_plants: int


class _EnvParams(Protocol):
    max_timesteps: int
    day_length: int
    mob_despawn_distance: int
    max_attribute: int


class _ReferenceState(Protocol):
    EnvParams: Callable[[], _EnvParams]
    StaticEnvParams: Callable[[], _StaticEnvParams]


def _reference_value(module: _ReferenceConstants, name: str) -> object:
    """Return a dynamically selected reference table."""
    value: object = getattr(module, name)  # pyright: ignore[reportAny] -- The reference module is dynamically imported.
    return value


def _ported_tensor(name: str) -> Tensor:
    """Return a ported table after narrowing the dynamic attribute lookup."""
    value: object = getattr(constants, name)  # pyright: ignore[reportAny] -- The table name is selected by pytest parameters.
    assert isinstance(value, Tensor)
    return value


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
