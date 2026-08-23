"""Tests for batched grid indexing with explicit out-of-bounds behavior.

The reference implementation gets its out-of-bounds behavior from JAX for
free; here it is written out, so these tests are what hold the two
implementations to the same rule.
"""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game.indexing import (
    gather_tiles,
    local_view,
    scatter_tiles,
    scatter_tiles_where,
)


def _grid() -> Tensor:
    return torch.arange(2 * 4 * 4, dtype=torch.int32).reshape(2, 4, 4)


def test_gather_reads_the_addressed_tile() -> None:
    values = gather_tiles(_grid(), torch.tensor([[1, 2], [0, 3]]))
    assert values.tolist() == [6, 19]


def test_gather_wraps_a_negative_index_like_python() -> None:
    # Measured against the reference: row -1 is the last row, not row 0. A
    # creature stepping off the top edge reads the bottom one.
    values = gather_tiles(_grid(), torch.tensor([[-1, 0], [0, -3]]))
    assert values.tolist() == [12, 17]


def test_gather_past_the_end_reads_the_nearest_edge() -> None:
    values = gather_tiles(_grid(), torch.tensor([[4, 0], [99, 99]]))
    assert values.tolist() == [12, 31]


def test_scatter_writes_the_addressed_tile() -> None:
    updated = scatter_tiles(
        _grid(),
        torch.tensor([[1, 2], [0, 3]]),
        torch.tensor([99, 88], dtype=torch.int32),
    )
    assert int(updated[0, 1, 2]) == 99
    assert int(updated[1, 0, 3]) == 88


def test_scatter_past_the_end_is_dropped_not_clamped() -> None:
    # A clamped write would corrupt the edge tile. That is the failure this
    # whole module exists to prevent, so it is asserted directly.
    grid = _grid()
    updated = scatter_tiles(
        grid,
        torch.tensor([[4, 0], [0, 9]]),
        torch.tensor([99, 99], dtype=torch.int32),
    )
    assert torch.equal(updated, grid)


def test_scatter_wraps_a_negative_index_rather_than_dropping_it() -> None:
    # Measured against the reference: writing to row -1 lands on the last row.
    updated = scatter_tiles(
        _grid(),
        torch.tensor([[-1, 0], [0, -3]]),
        torch.tensor([99, 88], dtype=torch.int32),
    )
    assert int(updated[0, 3, 0]) == 99
    assert int(updated[1, 0, 1]) == 88


def test_scatter_leaves_its_input_untouched() -> None:
    grid = _grid()
    original = grid.clone()
    _ = scatter_tiles(
        grid,
        torch.tensor([[0, 0], [0, 0]]),
        torch.tensor([99, 99], dtype=torch.int32),
    )
    assert torch.equal(grid, original)


def test_masked_scatter_writes_only_where_asked() -> None:
    updated = scatter_tiles_where(
        _grid(),
        torch.tensor([[0, 0], [0, 0]]),
        torch.tensor([99, 99], dtype=torch.int32),
        torch.tensor([True, False]),
    )
    assert int(updated[0, 0, 0]) == 99
    assert int(updated[1, 0, 0]) == 16


def test_masked_scatter_still_drops_writes_past_the_end() -> None:
    grid = _grid()
    updated = scatter_tiles_where(
        grid,
        torch.tensor([[4, 0], [0, 0]]),
        torch.tensor([99, 99], dtype=torch.int32),
        torch.tensor([True, True]),
    )
    assert torch.equal(updated[0], grid[0])
    assert int(updated[1, 0, 0]) == 99


def test_local_view_reads_a_centered_window() -> None:
    view = local_view(_grid(), torch.tensor([[1, 1], [2, 2]]), (3, 3))
    assert view.shape == (2, 3, 3)
    assert view[0].tolist() == [[0, 1, 2], [4, 5, 6], [8, 9, 10]]


def test_local_view_keeps_its_shape_at_a_corner() -> None:
    # A shrinking window would change the observation size depending on where
    # the player stands, which no downstream layer could consume.
    view = local_view(_grid(), torch.tensor([[0, 0], [3, 3]]), (3, 3))
    assert view.shape == (2, 3, 3)
    assert view[0].tolist() == [[0, 0, 0], [0, 0, 1], [0, 4, 5]]
    assert view[1].tolist() == [[26, 27, 0], [30, 31, 0], [0, 0, 0]]


def test_local_view_pads_beyond_the_grid_with_zero() -> None:
    view = local_view(_grid(), torch.tensor([[-4, -4], [9, 9]]), (3, 3))
    assert int(view.abs().sum()) == 0


def test_local_view_reports_the_requested_value_outside_the_grid() -> None:
    # The renderer pads with the out-of-bounds block so the agent can see the
    # world's edge; zero would read as a legitimate tile.
    view = local_view(_grid(), torch.tensor([[0, 0], [0, 0]]), (3, 3), outside=1)
    assert view[0].tolist() == [[1, 1, 1], [1, 0, 1], [1, 4, 5]]


@pytest.mark.parametrize("size", [(1, 1), (3, 5), (9, 11)])
def test_local_view_honors_the_requested_window(size: tuple[int, int]) -> None:
    view = local_view(_grid(), torch.tensor([[1, 1], [2, 2]]), size)
    assert view.shape == (2, *size)


def test_local_view_matches_a_manual_slice_away_from_the_edges() -> None:
    grid = torch.arange(1 * 8 * 8, dtype=torch.int32).reshape(1, 8, 8)
    view = local_view(grid, torch.tensor([[4, 4]]), (3, 3))
    assert torch.equal(view[0], grid[0, 3:6, 3:6])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
