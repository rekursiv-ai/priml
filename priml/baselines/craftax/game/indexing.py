"""Batched grid reads and writes with out-of-bounds semantics made explicit.

The reference implementation is written in JAX, whose indexing quietly absorbs
out-of-range coordinates. PyTorch instead raises, or -- worse -- returns a
smaller tensor. The game relies on the quiet behaviour: a creature stepping off
the edge, a torch lighting a tile beyond the wall, and a player standing in a
corner all produce coordinates outside the map.

The absorbed behaviour is not one rule but two, and the difference is
measurable rather than a matter of taste (see ``indexing_test.py``, which pins
each against the reference):

* A NEGATIVE index wraps, exactly as it does in ordinary Python. Row ``-1`` is
  the last row, and a write there lands on the far edge of the map.
* An index past the END is clamped for a read and DROPPED for a write. Writing
  a clamped value instead would silently corrupt the edge tile, which is a
  wrong answer rather than a crash.

Rather than repeat that at every one of the many call sites, and get it wrong
somewhere, these helpers state the rule once. Each is batched over a leading
environment axis.
"""

from __future__ import annotations

from torch import Tensor

import torch


def gather_tiles(grid: Tensor, positions: Tensor) -> Tensor:
    """Read one tile per environment; negatives wrap and overflow clamps.

    Args:
      grid: Values to read, ``[envs, rows, columns]``.
      positions: Row/column per environment, ``[envs, 2]``.

    Returns:
      values: The tile under each position, ``[envs]``.

    """
    rows, columns = _clamped(positions, grid.shape[-2], grid.shape[-1])
    return grid[torch.arange(grid.shape[0], device=grid.device), rows, columns]


def scatter_tiles(grid: Tensor, positions: Tensor, values: Tensor) -> Tensor:
    """Write one tile per environment, dropping writes past the end.

    Dropping rather than clamping is the load-bearing choice: a clamped write
    would silently modify the edge tile, which is a wrong answer rather than a
    crash. A negative coordinate is not out of bounds -- it wraps.

    Args:
      grid: Values to write into, ``[envs, rows, columns]``.
      positions: Row/column per environment, ``[envs, 2]``.
      values: One value per environment, ``[envs]``, or a scalar.

    Returns:
      grid: A new grid with the in-bounds writes applied.

    """
    envs, height, width = grid.shape[0], grid.shape[-2], grid.shape[-1]
    rows, columns = _clamped(positions, height, width)
    index = torch.arange(envs, device=grid.device)
    updated = grid.clone()
    updated[index, rows, columns] = torch.where(
        _inside(positions, height, width),
        values.to(grid.dtype),
        updated[index, rows, columns],
    )
    return updated


def scatter_tiles_where(
    grid: Tensor,
    positions: Tensor,
    values: Tensor,
    apply: Tensor,
) -> Tensor:
    """Write one tile per environment, only where ``apply`` is set.

    The batch has no per-environment control flow, so an action that only some
    environments took is expressed as a masked write rather than a branch.

    Args:
      grid: Values to write into, ``[envs, rows, columns]``.
      positions: Row/column per environment, ``[envs, 2]``.
      values: One value per environment, ``[envs]``.
      apply: Which environments perform the write, ``[envs]``.

    Returns:
      grid: A new grid with the selected in-bounds writes applied.

    """
    height, width = grid.shape[-2], grid.shape[-1]
    rows, columns = _clamped(positions, height, width)
    index = torch.arange(grid.shape[0], device=grid.device)
    current = grid[index, rows, columns]
    updated = grid.clone()
    updated[index, rows, columns] = torch.where(
        apply & _inside(positions, height, width),
        values.to(grid.dtype),
        current,
    )
    return updated


def local_view(
    grid: Tensor,
    centers: Tensor,
    size: tuple[int, int],
    *,
    outside: float = 0.0,
) -> Tensor:
    """Cut a fixed window around each environment's center, padding outside.

    The window is always exactly ``size``: positions near an edge read padding
    rather than a smaller view, which is what keeps the observation shape
    independent of where the player stands.

    Args:
      grid: Values to read, ``[envs, rows, columns]``.
      centers: Row/column of each window's center, ``[envs, 2]``.
      size: Window height and width, both odd.
      outside: Value reported for tiles beyond the grid. The renderer passes
        the out-of-bounds block so the agent can see the world's edge.

    Returns:
      view: The windows, ``[envs, size[0], size[1]]``.

    """
    height, width = size
    row_offsets = torch.arange(height, device=grid.device) - height // 2
    column_offsets = torch.arange(width, device=grid.device) - width // 2
    rows = centers[:, 0, None] + row_offsets[None, :]
    columns = centers[:, 1, None] + column_offsets[None, :]
    inside = (
        (rows >= 0)[:, :, None]
        & (rows < grid.shape[-2])[:, :, None]
        & (columns >= 0)[:, None, :]
        & (columns < grid.shape[-1])[:, None, :]
    )
    gathered = grid[
        torch.arange(grid.shape[0], device=grid.device)[:, None, None],
        rows.clamp(0, grid.shape[-2] - 1)[:, :, None],
        columns.clamp(0, grid.shape[-1] - 1)[:, None, :],
    ]
    return torch.where(inside, gathered, torch.full_like(gathered, outside))


def _clamped(positions: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Return usable row and column indices: negatives wrap, overflow clamps."""
    rows, columns = positions[..., 0].long(), positions[..., 1].long()
    return (
        torch.where(rows < 0, rows + height, rows).clamp(0, height - 1),
        torch.where(columns < 0, columns + width, columns).clamp(0, width - 1),
    )


def _inside(positions: Tensor, height: int, width: int) -> Tensor:
    """Return whether each position addresses a tile once negatives wrap."""
    rows, columns = positions[..., 0], positions[..., 1]
    return (rows >= -height) & (rows < height) & (columns >= -width) & (columns < width)
