"""Maze-Hard tokens, dihedral views, the canonical path, and the loader.

A maze is a 30x30 grid of six tokens. The label is the input with the cells of
ONE shortest start-to-goal path marked. Most mazes have several shortest paths,
and the benchmark scores the whole grid exactly, so a valid path that is not the
stored one still fails. The stored labels follow one fixed rule: breadth-first
search that expands neighbours down, left, right, up, then walks the parent
pointers back from the goal (:func:`canonical_path_label` reproduces it).

That rule is not invariant under rotation or reflection. A rotated maze with its
label rotated along carries a path the rule would not pick in the rotated frame,
so training on rotated copies teaches eight conflicting tie-breaking policies.
Relabeling every view in its own frame (:func:`canonical_labels`) restores one
policy. ``scripts/prepare_data.py`` writes both forms; the recipes choose one.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Final, NotRequired, cast, override

import math

from configgle import Makes

import numpy as np

from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray


CHARSET: Final = "# SGo"
"""Source characters in token order: token ``i + 1`` is ``CHARSET[i]``; 0 pads."""

WALL: Final = 1
"""A blocked cell (``#``)."""

OPEN: Final = 2
"""A free cell off the path (space)."""

START: Final = 3
"""The start cell (``S``)."""

GOAL: Final = 4
"""The goal cell (``G``)."""

PATH: Final = 5
"""A free cell on the labeled path (``o``); appears only in labels."""

VOCAB_SIZE: Final = len(CHARSET) + 1
"""Pad plus the five source characters."""

NEIGHBOURS: Final = ((1, 0), (0, -1), (0, 1), (-1, 0))
"""Row and column offsets in the order the stored labels' search expands them:
down, left, right, up."""

DIHEDRAL_VIEWS: Final[tuple[Callable[[NDArray[np.uint8]], NDArray[np.uint8]], ...]] = (
    lambda grid: grid,
    lambda grid: np.rot90(grid, k=1),
    lambda grid: np.rot90(grid, k=2),
    lambda grid: np.rot90(grid, k=3),
    np.fliplr,
    np.flipud,
    lambda grid: grid.T,
    lambda grid: np.fliplr(np.rot90(grid, k=1)),
)
"""The eight symmetries of the square, in the order a maze's views are stored.

The order is part of the data: training shuffles rows by index, so a different
order is a different stream of batches."""

ORIGINAL: Final = "hard-1k"
"""Prepared tree of the eight views with the stored labels moved along."""

CANONICAL: Final = "hard-1k-canonical"
"""Prepared tree of the eight views, each relabeled in its own frame."""


class MazeData(PuzzleDataset):
    """Prepared maze arrays on the device, with the epoch count a loop reads.

    Batches are drawn exactly as :class:`PuzzleDataset` draws them -- the
    two-level seeded shuffle, a zero-padded final batch, and a puzzle id of 0
    for the one shared prefix row.
    """

    class Config(Makes["MazeData"], PuzzleDataset.Config):
        """Where the prepared arrays live, and how batches are drawn."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the training split; ticked by the loop."""

    class MazeStateDict(PuzzleDataset.StateDict):
        """The shuffle's epoch counter and the loop's epoch timer."""

        timer_epoch: NotRequired[CheckpointableStepTimer.StateDict]

    @override
    def state_dict(self) -> MazeStateDict:
        return {**super().state_dict(), "timer_epoch": self.timer_epoch.state_dict()}

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        super().load_state_dict(state_dict)
        state = cast(MazeData.MazeStateDict, state_dict)
        if "timer_epoch" in state:
            self.timer_epoch.load_state_dict(state["timer_epoch"])


def tokenize(text: str, *, side: int) -> NDArray[np.uint8]:
    """Map one source grid string to a ``[side, side]`` token array.

    Args:
      text: ``side * side`` characters from :data:`CHARSET`, row-major.
      side: Grid side.

    Returns:
      grid: Tokens ``1``-``5``.

    Raises:
      ValueError: The string has the wrong length or a character outside the set.

    """
    if len(text) != side * side:
        raise ValueError(
            f"A {side}x{side} maze needs {side * side} cells; got {len(text)}.",
        )
    if set(text) - set(CHARSET):
        raise ValueError(f"Maze characters must be drawn from {CHARSET!r}.")
    tokens = [CHARSET.index(char) + 1 for char in text]
    return np.array(tokens, dtype=np.uint8).reshape(side, side)


def canonical_path_label(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return ``grid`` with the canonical shortest start-to-goal path marked.

    Args:
      grid: ``[H, W]`` input tokens with exactly one start and one goal.

    Returns:
      label: A copy of ``grid`` whose path cells, start and goal excluded, are
        :data:`PATH`.

    Raises:
      ValueError: The start or goal is missing or repeated, or no path joins them.

    """
    cells = cast(list[list[int]], grid.tolist())
    starts = [
        (r, c) for r, row in enumerate(cells) for c, v in enumerate(row) if v == START
    ]
    goals = [
        (r, c) for r, row in enumerate(cells) for c, v in enumerate(row) if v == GOAL
    ]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("A maze needs exactly one start and one goal.")
    start, goal = starts[0], goals[0]
    parents = _search(cells, start=start, goal=goal)
    if goal not in parents:
        raise ValueError("No path joins the start to the goal.")
    label = grid.copy()
    cursor = parents[goal]
    while cursor != start:
        label[cursor] = PATH
        cursor = parents[cursor]
    return label


def canonical_labels(inputs: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Relabel ``[N, side * side]`` flattened square mazes, each in its own frame."""
    side = math.isqrt(inputs.size // len(inputs))
    grids = np.unstack(inputs.reshape(len(inputs), side, side))
    return np.stack([canonical_path_label(grid).reshape(-1) for grid in grids])


def _search(
    cells: list[list[int]],
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Breadth-first parents in :data:`NEIGHBOURS` order, stopping at the goal."""
    height, width = len(cells), len(cells[0])
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    seen = {start}
    queue = deque([start])
    while queue and goal not in seen:
        row, col = queue.popleft()
        for d_row, d_col in NEIGHBOURS:
            cell = (row + d_row, col + d_col)
            if (
                cell in seen
                or cell[0] < 0
                or cell[0] >= height
                or cell[1] < 0
                or cell[1] >= width
                or cells[cell[0]][cell[1]] == WALL
            ):
                continue
            seen.add(cell)
            parents[cell] = (row, col)
            queue.append(cell)
    return parents
