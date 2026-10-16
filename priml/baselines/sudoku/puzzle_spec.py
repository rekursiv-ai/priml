"""Data-owned geometry and vocabulary for grid puzzles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from configgle import Fig

import torch


if TYPE_CHECKING:
    from torch import Tensor


class PuzzleSpec(Protocol):
    """Read-only geometry and vocabulary supplied by a dataset."""

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Axes of the puzzle's token grid."""
        ...

    @property
    def vocab_size(self) -> int:
        """Number of token IDs in the puzzle vocabulary."""
        ...


class SudokuSpec(Fig["SudokuSpec"]):
    """Geometry and token vocabulary of a standard sudoku puzzle."""

    grid_shape: tuple[int, int] = (9, 9)
    """Row and column extents."""

    box_shape: tuple[int, int] = (3, 3)
    """Row and column extents of each constraint box."""

    vocab_size: int = 11
    """Pad, blank, and nine digit tokens."""


def sudoku_group_indices(spec: SudokuSpec) -> Tensor:
    """Return the cell indices of every sudoku constraint group.

    Args:
      spec: Dataset-owned grid and constraint-box geometry.

    Returns:
      group_indices: ``[27, 9]`` long tensor by default; rows 0-8 are the 9 grid rows,
        rows 9-17 the 9 columns, rows 18-26 the 9 boxes, each listing the 9
        row-major cell indices (0-80) of that group.

    """
    rows, cols = spec.grid_shape
    box_rows, box_cols = spec.box_shape
    cells = torch.arange(rows * cols).reshape(rows, cols)
    boxes = (
        cells.reshape(rows // box_rows, box_rows, cols // box_cols, box_cols)
        .permute(0, 2, 1, 3)
        .reshape(-1, box_rows * box_cols)
    )
    return torch.cat([cells, cells.T, boxes])
