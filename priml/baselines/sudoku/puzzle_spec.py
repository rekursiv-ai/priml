"""Data-owned geometry and vocabulary for grid puzzles."""

from __future__ import annotations

from typing import Protocol

from configgle import Fig


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
