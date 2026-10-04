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
    _bounds,
    batch_rows,
    gather_tiles,
    local_view,
    scatter_tiles,
    scatter_tiles_where,
)


def _grid() -> Tensor:
    return torch.arange(2 * 3 * 4, dtype=torch.int32).reshape(2, 3, 4)


def test_gather_reads_the_addressed_tile() -> None:
    values = gather_tiles(_grid(), torch.tensor([[1, 2], [0, 3]]))
    assert values.tolist() == [6, 15]


def test_gather_wraps_a_negative_index_like_python() -> None:
    # Measured against the reference: row -1 is the last row, not row 0. A
    # creature stepping off the top edge reads the bottom one.
    values = gather_tiles(_grid(), torch.tensor([[-1, 0], [0, -3]]))
    assert values.tolist() == [8, 13]


def test_gather_past_the_end_reads_the_nearest_edge() -> None:
    values = gather_tiles(_grid(), torch.tensor([[4, 0], [99, 99]]))
    assert values.tolist() == [8, 23]


def test_gather_wraps_once_then_clamps_on_a_rectangular_grid() -> None:
    # Three rows, five columns, so a helper that mixed up the two extents reads
    # the wrong tile. A negative wraps ONCE: -4 rows is -1 after the wrap, which
    # then clamps to the first row rather than wrapping again.
    grid = torch.arange(2 * 3 * 5, dtype=torch.int32).reshape(2, 3, 5)
    positions = [(-4, -6), (-3, -5), (-1, -2), (2, 7), (5, 3)]
    values = [
        int(gather_tiles(grid, torch.tensor([position]))[0]) for position in positions
    ]
    assert values == [0, 0, 13, 14, 13]


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
    assert int(updated[0, 2, 0]) == 99
    assert int(updated[1, 0, 1]) == 88


def test_scatter_casts_int64_values_to_grid_dtype() -> None:
    grid = _grid()
    updated = scatter_tiles(
        grid,
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([101, 202], dtype=torch.int64),
    )
    expected = grid.clone()
    expected[0, 1, 2] = 101
    expected[1, 2, 3] = 202
    assert updated.dtype == torch.int32
    assert torch.equal(updated, expected)


def test_scatter_where_casts_int64_values_to_grid_dtype() -> None:
    grid = _grid()
    updated = scatter_tiles_where(
        grid,
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([101, 202], dtype=torch.int64),
        torch.tensor([True, False]),
    )
    expected = grid.clone()
    expected[0, 1, 2] = 101
    assert updated.dtype == torch.int32
    assert torch.equal(updated, expected)


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
    assert int(updated[1, 0, 0]) == 12


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
    view = local_view(_grid(), torch.tensor([[0, 0], [2, 4]]), (3, 3))
    assert view.shape == (2, 3, 3)
    assert view[0].tolist() == [[0, 0, 0], [0, 0, 1], [0, 4, 5]]
    assert view[1].tolist() == [[19, 0, 0], [23, 0, 0], [0, 0, 0]]


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
    grid = torch.arange(2 * 7 * 8, dtype=torch.int32).reshape(2, 7, 8)
    view = local_view(grid, torch.tensor([[4, 4], [4, 4]]), (3, 3))
    assert torch.equal(view[0], grid[0, 3:6, 3:6])


def test_writes_wrap_the_most_negative_valid_index() -> None:
    grid = _grid()
    updated = scatter_tiles(
        grid,
        torch.tensor([[-3, -4], [3, 4]]),
        torch.tensor([91, 92], dtype=torch.int32),
    )
    assert int(updated[0, 0, 0]) == 91
    assert int(updated[1, 2, 3]) == 23


def test_indexing_preserves_meta_device(monkeypatch: pytest.MonkeyPatch) -> None:
    arange = torch.arange
    tensor = torch.tensor
    requested_devices: list[object] = []

    def arange_on_requested_device(
        end: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        requested_devices.append(device)
        return arange(end, device=device, dtype=dtype)

    def tensor_on_requested_device(
        data: object,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        requested_devices.append(device)
        return tensor(
            data,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    batch_rows.cache_clear()
    _bounds.cache_clear()
    monkeypatch.setattr(torch, "arange", arange_on_requested_device)
    monkeypatch.setattr(torch, "tensor", tensor_on_requested_device)
    grid = torch.empty((3, 4, 5), device="meta")
    positions = torch.empty((3, 2), dtype=torch.int64, device="meta")
    values = torch.empty((3,), device="meta")
    assert gather_tiles(grid, positions).device.type == "meta"
    assert scatter_tiles(grid, positions, values).device.type == "meta"
    assert (
        scatter_tiles_where(
            grid,
            positions,
            values,
            torch.empty((3,), dtype=torch.bool, device="meta"),
        ).device.type
        == "meta"
    )
    assert local_view(grid, positions, (3, 5)).device.type == "meta"
    assert requested_devices
    assert all(device == torch.device("meta") for device in requested_devices)


def test_scatter_where_uses_both_spatial_extents() -> None:
    grid = torch.arange(3 * 4 * 5, dtype=torch.int32).reshape(3, 4, 5)
    positions = torch.tensor([[3, 4], [1, 2], [2, 3]])
    updated = scatter_tiles_where(
        grid,
        positions,
        torch.tensor([91, 92, 93], dtype=torch.int32),
        torch.tensor([True, True, True]),
    )
    assert int(updated[0, 3, 4]) == 91
    assert int(updated[1, 1, 2]) == 92
    assert int(updated[2, 2, 3]) == 93


def test_scatter_wraps_and_drops_at_both_axis_boundaries() -> None:
    grid = torch.zeros((5, 4, 6), dtype=torch.int32)
    updated = scatter_tiles(
        grid,
        torch.tensor([[0, 0], [-4, -6], [4, 2], [2, 6], [-5, 1]]),
        torch.tensor([11, 12, 13, 14, 15], dtype=torch.int32),
    )
    assert int(updated[0, 0, 0]) == 11
    assert int(updated[1, 0, 0]) == 12
    assert int(updated[2].sum()) == 0
    assert int(updated[3].sum()) == 0
    assert int(updated[4].sum()) == 0


def test_local_view_five_by_five_is_centered() -> None:
    grid = torch.arange(2 * 7 * 8, dtype=torch.int32).reshape(2, 7, 8)
    view = local_view(grid, torch.tensor([[3, 3], [4, 5]]), (5, 5))
    assert torch.equal(view[0], grid[0, 1:6, 1:6])
    assert torch.equal(view[1], grid[1, 2:7, 3:8])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
