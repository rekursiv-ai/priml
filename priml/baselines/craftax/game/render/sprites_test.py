"""Tests for the viewer's enum-indexed sprite tables."""

from __future__ import annotations

from priml.baselines.craftax.game.render import sprites


def test_block_sprites_follow_the_block_enum() -> None:
    assert len(sprites.BLOCK_SPRITES) == 37
    assert sprites.BLOCK_SPRITES[1] == ""
    assert sprites.BLOCK_SPRITES[2] == "grass.png"
    assert sprites.BLOCK_SPRITES[18] == ""
    assert sprites.BLOCK_SPRITES[36] == "necromancer_vulnerable.png"


def test_item_and_player_sprites_follow_their_indices() -> None:
    assert sprites.ITEM_SPRITES == (
        "",
        "torch_in_inventory.png",
        "ladder_down.png",
        "ladder_up.png",
        "ladder_down_blocked.png",
    )
    assert sprites.PLAYER_SPRITES == (
        "player-left.png",
        "player-right.png",
        "player-up.png",
        "player-down.png",
        "player-sleep.png",
    )


def test_every_sprite_is_sorted_unique_and_excludes_flat_tiles() -> None:
    names = sprites.every_sprite()
    assert names == tuple(sorted(set(names)))
    assert "" not in names
    assert len(names) == 68


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
