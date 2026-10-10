"""Tests for the path-boundary loss."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.maze.data import CHARSET, OPEN, PATH, WALL
from priml.baselines.maze.loss import CanonicalBoundary, canonical_boundary_mask


SHAPE: Final = (3, 4)

ROUTED: Final = ("S#  ", "o  #", "ooG ")
"""A labeled maze: start, three path cells, goal; walls and open cells around."""

WALLED: Final = ("S###", "####", "###G")
"""A label with no path cell and no open cell, so nothing is selected."""


def test_the_mask_is_the_path_and_the_open_cells_beside_the_route() -> None:
    """Start and goal are never scored, but cells beside them are."""
    mask = canonical_boundary_mask(_labels(ROUTED, WALLED), grid_shape=SHAPE)
    expected = torch.tensor(
        [
            [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    ).reshape(2, -1)
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("offset", [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_an_open_cell_counts_from_every_side(offset: tuple[int, int]) -> None:
    labels = torch.full((2, *SHAPE), WALL)
    labels[0, 1, 1] = PATH
    beside = (1 + offset[0], 1 + offset[1])
    labels[0, beside[0], beside[1]] = OPEN
    expected = torch.zeros_like(labels, dtype=torch.bool)
    expected[0, 1, 1] = expected[0, beside[0], beside[1]] = True
    mask = canonical_boundary_mask(labels.reshape(2, -1), grid_shape=SHAPE)
    assert torch.equal(mask, expected.reshape(2, -1))


def test_the_loss_averages_each_maze_over_its_own_mask() -> None:
    """A maze with nothing selected scores zero rather than dividing by zero."""
    loss = CanonicalBoundary.Config(grid_shape=SHAPE).make()
    per_token = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    actual = loss(per_token, _labels(ROUTED, WALLED))
    # Row-major indices of the six selected cells of ROUTED.
    selected = (4, 5, 6, 8, 9, 11)
    assert torch.equal(actual, torch.tensor([sum(selected) / len(selected), 0.0]))


def test_an_unset_grid_is_rejected() -> None:
    """The grid is normally filled in from the dataset when the loop finalizes."""
    with pytest.raises(ValueError, match="grid_shape must be positive"):
        CanonicalBoundary.Config().make()


def _labels(*mazes: tuple[str, ...]) -> torch.Tensor:
    """Tokenize each maze's rows and flatten it to one label row."""
    return torch.tensor(
        [[CHARSET.index(char) + 1 for row in maze for char in row] for maze in mazes],
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
