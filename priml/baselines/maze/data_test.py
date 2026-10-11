"""Tests for maze tokens, dihedral views, and canonical labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.baselines.maze.data import (
    CHARSET,
    DIHEDRAL_VIEWS,
    canonical_labels,
    canonical_path_label,
    tokenize,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_tokenize_maps_each_character_to_its_token() -> None:
    np.testing.assert_array_equal(
        tokenize("#SGo", side=2),
        np.array([[1, 3], [4, 5]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [("#S G", "needs 9 cells"), ("#S Gx####", "drawn from")],
)
def test_tokenize_rejects_a_malformed_grid(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tokenize(text, side=3)


def test_the_canonical_path_expands_down_left_right_up() -> None:
    """Of the ten shortest paths across an open room, the stored rule picks one.

    Breadth-first search reaches each cell first from the neighbour it expanded
    earliest, so the walk back from the goal hugs the left edge, then the bottom.
    """
    maze = _grid("S   ", "    ", "   G")
    np.testing.assert_array_equal(
        canonical_path_label(maze),
        _grid("S   ", "o   ", "oooG"),
    )


def test_the_canonical_path_goes_around_walls() -> None:
    maze = _grid("S#  ", " # #", "   G")
    np.testing.assert_array_equal(
        canonical_path_label(maze),
        _grid("S#  ", "o# #", "oooG"),
    )


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (("S   ", "    ", "    "), "one start and one goal"),
        (("S  G", "    ", "   G"), "one start and one goal"),
        (("S#  ", "##  ", "   G"), "No path"),
    ],
)
def test_a_maze_without_one_route_is_rejected(
    rows: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        canonical_path_label(_grid(*rows))


def test_each_view_is_relabeled_in_its_own_frame() -> None:
    """A rotated label is a valid path, but not the one the rule picks there.

    The stored rule is not invariant under a quarter turn: this maze's label,
    rotated along, is a shortest path the rule would not pick in the new frame.
    """
    maze = _grid("  S", "   ", "G  ")
    rotated = np.rot90(maze)
    first, second = np.unstack(
        canonical_labels(np.stack([maze.reshape(-1), rotated.reshape(-1)])),
    )
    np.testing.assert_array_equal(first, canonical_path_label(maze).reshape(-1))
    np.testing.assert_array_equal(second, canonical_path_label(rotated).reshape(-1))
    assert not np.array_equal(second, np.rot90(canonical_path_label(maze)).reshape(-1))


def test_the_dihedral_views_are_the_eight_symmetries_in_stored_order() -> None:
    """The order is pinned because training shuffles rows by index.

    A 2x3 grid: its six distinct cells tell all eight apart, and the two
    sides tell a quarter turn from a reflection by the shape it leaves.
    """
    grid = np.arange(6, dtype=np.uint8).reshape(2, 3)
    views = [view(grid) for view in DIHEDRAL_VIEWS]
    wide, tall = (2, 3), (3, 2)
    assert [v.shape for v in views] == [wide, tall, wide, tall, wide, wide, tall, tall]
    assert [v.reshape(-1).tolist() for v in views] == [
        [0, 1, 2, 3, 4, 5],
        [2, 5, 1, 4, 0, 3],
        [5, 4, 3, 2, 1, 0],
        [3, 0, 4, 1, 5, 2],
        [2, 1, 0, 5, 4, 3],
        [3, 4, 5, 0, 1, 2],
        [0, 3, 1, 4, 2, 5],
        [5, 2, 4, 1, 3, 0],
    ]


def _grid(*rows: str) -> NDArray[np.uint8]:
    """Tokenize equal-length rows of maze characters."""
    tokens = [[CHARSET.index(char) + 1 for char in row] for row in rows]
    return np.array(tokens, dtype=np.uint8)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
