"""Tests that the per-floor terrain recipes match the reference."""

from __future__ import annotations

from typing import cast

import dataclasses

import numpy as np
import pytest

from priml.baselines.craftax.conftest import reference, requires_craftax
from priml.baselines.craftax.game import world_config
from priml.baselines.craftax.game.constants import NUM_LEVELS


def test_every_floor_has_a_recipe() -> None:
    assert len(world_config.LEVEL_CONFIGS) == NUM_LEVELS


def test_descent_alternates_between_the_two_generators() -> None:
    # The player meets an open floor, then a dungeon, and so on; the order is
    # what the world builder walks.
    kinds = [type(config).__name__ for config in world_config.LEVEL_CONFIGS]
    assert kinds == [
        "SmoothWorldConfig",
        "DungeonConfig",
        "SmoothWorldConfig",
        "DungeonConfig",
        "DungeonConfig",
        "SmoothWorldConfig",
        "SmoothWorldConfig",
        "SmoothWorldConfig",
        "SmoothWorldConfig",
    ]


def test_the_overworld_is_the_only_floor_without_an_ascent() -> None:
    # There is nowhere above the surface, so an up ladder there would strand
    # the player outside the world.
    assert not world_config.OVERWORLD.ladder_up
    assert world_config.GNOMISH_MINES.ladder_up


def test_the_boss_floor_is_a_dead_end() -> None:
    assert not world_config.GRAVEYARD.ladder_up
    assert not world_config.GRAVEYARD.ladder_down


def test_ore_slots_are_declared_consistently() -> None:
    for config in world_config.SMOOTH_WORLD_CONFIGS:
        assert len(config.ores) == 5
        assert len(config.ore_chances) == 5
        assert len(config.ore_requirement_blocks) == 5


def test_only_lit_floors_declare_ambient_light() -> None:
    # A dark floor is what makes torches matter; the surface and the fire realm
    # are the two that do not need them.
    lit = {
        config.default_light
        for config in world_config.SMOOTH_WORLD_CONFIGS
        if config.default_light > 0
    }
    assert lit == {1.0}
    assert world_config.GNOMISH_MINES.default_light == 0.0


@requires_craftax
@pytest.mark.parametrize(
    ("ported", "upstream_name"),
    [
        ("OVERWORLD", "OVERWORLD_CONFIG"),
        ("GNOMISH_MINES", "GNOMISH_MINES_CONFIG"),
        ("TROLL_MINES", "TROLL_MINES_CONFIG"),
        ("FIRE_REALM", "FIRE_LEVEL_CONFIG"),
        ("ICE_REALM", "ICE_LEVEL_CONFIG"),
        ("GRAVEYARD", "BOSS_LEVEL_CONFIG"),
        ("DUNGEON", "DUNGEON_CONFIG"),
        ("SEWERS", "SEWER_CONFIG"),
        ("VAULTS", "VAULTS_CONFIG"),
    ],
)
def test_recipe_matches_reference(ported: str, upstream_name: str) -> None:
    upstream = reference("craftax.world_gen.world_gen_configs")

    ours = getattr(world_config, ported)
    theirs = getattr(upstream, upstream_name)
    for field in dataclasses.fields(ours):
        actual = getattr(ours, field.name)
        expected = getattr(theirs, field.name)
        # The reference stores its probabilities as float32 while these are
        # written as ordinary Python floats, so every comparison is made at
        # the precision the game actually runs at.
        if isinstance(actual, tuple):
            values = cast("tuple[float, ...]", actual)
            assert [np.float32(value) for value in values] == np.asarray(
                expected,
                dtype=np.float32,
            ).tolist(), field.name
            continue
        assert np.float32(actual) == np.float32(expected), field.name


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
