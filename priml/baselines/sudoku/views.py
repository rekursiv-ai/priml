"""Exact-symmetry test-time-augmentation (TTA) views for sudoku token grids.

Sudoku's exact symmetry group lets ONE frozen checkpoint act as several vote
members: evaluate a symmetry-transformed copy of each puzzle, then map the
prediction back through the inverse transform. A recorded view composes
(applied in this order):

- optional transpose of the 9x9 grid positions,
- band-preserving row and column permutations (each 3-block of lines must map
  onto a 3-block, so bands map to bands and stacks to stacks),
- digit relabeling: a permutation of digits 1-9 applied to the token values
  (tokens 2-10 encode digits 1-9; pad 0 and blank 1 are fixed points).

Every component is an exact Sudoku symmetry, so a transformed puzzle is a
legal Sudoku with the transformed unique solution: givens map to givens and
solutions to solutions. Search runs entirely in transformed space; root and
final grids are mapped back through :func:`invert_symmetry` before any
metric or vote sees them, so exact flags, dumps, and modal voting all
operate in original space against the original labels.

:data:`NINE_VIEWS` is a FROZEN evaluation table -- the recorded view order
is the committee tie-break order and must not be edited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

from torch import Tensor

import torch


if TYPE_CHECKING:
    from collections.abc import Iterable


IDENTITY_DIGITS: Final = (1, 2, 3, 4, 5, 6, 7, 8, 9)
"""Identity digit relabeling (digit d stays d)."""


IDENTITY_LINES: Final = (0, 1, 2, 3, 4, 5, 6, 7, 8)
"""Identity row/column permutation (line i stays i)."""


REVERSED_LINES: Final = (8, 7, 6, 5, 4, 3, 2, 1, 0)
"""Reversed row/column permutation (bands map onto bands in reverse)."""


class View(NamedTuple):
    """One recorded exact Sudoku symmetry.

    Attributes:
      name: Stable view label (member name in committee tables).
      digit_permutation: Relabeled digit for each of 1..9.
      transpose: Whether to transpose the 9x9 grid positions.
      row_permutation: Source row per output row; maps bands onto bands.
      col_permutation: Source column per output column; maps stacks onto
        stacks.

    """

    name: str
    digit_permutation: tuple[int, ...]
    transpose: bool
    row_permutation: tuple[int, ...] = IDENTITY_LINES
    col_permutation: tuple[int, ...] = IDENTITY_LINES

    def apply(self, grid: Tensor) -> Tensor:
        """Apply this view's recorded symmetry (see :func:`apply_symmetry`).

        Args:
            grid: Token grid to transform.

        Returns:
            transformed: Transformed token grid.

        """
        return apply_symmetry(
            grid,
            digit_permutation=self.digit_permutation,
            transpose_grid=self.transpose,
            row_permutation=self.row_permutation,
            col_permutation=self.col_permutation,
        )

    def invert(self, grid: Tensor) -> Tensor:
        """Invert this view's recorded symmetry (see :func:`invert_symmetry`).

        Args:
            grid: Transformed token grid.

        Returns:
            original: Original-space token grid.

        """
        return invert_symmetry(
            grid,
            digit_permutation=self.digit_permutation,
            transpose_grid=self.transpose,
            row_permutation=self.row_permutation,
            col_permutation=self.col_permutation,
        )


# The frozen nine-view committee table (tie-break order). "Trained class" =
# transform families sampled by train-time augmentation (digit permutations +
# dihedral); stack_rot_digit_shift2 and full_mix carry band/stack/line
# components that training never saw.
#
#   view                   transform                          class
#   canonical              identity (the plain pass)          trained
#   shift1                 digits cyclic +1                   trained
#   shift4                 digits cyclic +4                   trained
#   reversal               digits 1<->9, 2<->8, ...           trained
#   anti_transpose         reflect across the anti-diagonal   trained
#                          (transpose + reverse rows/cols)
#   stack_rot_digit_shift2 rotate the three column-stacks     untrained
#                          cyclically + digits cyclic +2      (stack perm)
#   full_mix               row + column perms + transpose     untrained
#                          + digit scramble, composed         (deep comp.)
#   dg_shift1_rot180       digits cyclic +1; rotate 180       trained
#                          (reverse rows + columns)
#   dg_shift3_pure         digits cyclic +3                   trained.
NINE_VIEWS: tuple[View, ...] = (
    View("canonical", IDENTITY_DIGITS, False),
    View("shift1", (2, 3, 4, 5, 6, 7, 8, 9, 1), False),
    View("shift4", (5, 6, 7, 8, 9, 1, 2, 3, 4), False),
    View("reversal", (9, 8, 7, 6, 5, 4, 3, 2, 1), False),
    View("anti_transpose", IDENTITY_DIGITS, True, REVERSED_LINES, REVERSED_LINES),
    View(
        "stack_rot_digit_shift2",
        (3, 4, 5, 6, 7, 8, 9, 1, 2),
        False,
        IDENTITY_LINES,
        (6, 7, 8, 0, 1, 2, 3, 4, 5),
    ),
    View(
        "full_mix",
        (7, 3, 9, 1, 8, 2, 4, 6, 5),
        True,
        (4, 5, 3, 8, 6, 7, 1, 2, 0),
        (5, 3, 4, 0, 1, 2, 7, 8, 6),
    ),
    View(
        "dg_shift1_rot180",
        (2, 3, 4, 5, 6, 7, 8, 9, 1),
        False,
        REVERSED_LINES,
        REVERSED_LINES,
    ),
    View("dg_shift3_pure", (4, 5, 6, 7, 8, 9, 1, 2, 3), False),
)
"""The frozen nine-view agreement-lock / sieve ladder, in tie-break order."""


def apply_symmetry(
    grid: Tensor,
    *,
    digit_permutation: tuple[int, ...],
    transpose_grid: bool,
    row_permutation: tuple[int, ...] = IDENTITY_LINES,
    col_permutation: tuple[int, ...] = IDENTITY_LINES,
) -> Tensor:
    """Apply a recorded exact Sudoku symmetry to token grids.

    Position components apply as transpose first, then row/column
    permutations; digit relabeling is elementwise and commutes with all
    position components.

    Args:
      grid: ``[..., 81]`` token grid (0 pad, 1 blank, 2-10 digits 1-9); any
        dtype holding integral values, negative ignore ids pass through.
      digit_permutation: Relabeled digit for each of 1..9.
      transpose_grid: Whether to transpose the 9x9 positions.
      row_permutation: Source row per output row; must map bands onto bands.
      col_permutation: Source column per output column; must map stacks onto
        stacks.

    Returns:
      transformed: Same shape and dtype, transformed tokens.

    """
    lut = _token_lut(digit_permutation, device=grid.device)
    tokens = grid.long()
    mapped = torch.where(
        (tokens >= 0) & (tokens < lut.shape[0]),
        lut[tokens.clamp(min=0, max=lut.shape[0] - 1)],
        tokens,
    )
    if _moves_positions(row_permutation, col_permutation, transpose_grid):
        index = _position_index(
            row_permutation,
            col_permutation,
            transpose_grid=transpose_grid,
            device=grid.device,
        )
        mapped = mapped[..., index]
    return mapped.to(grid.dtype)


def invert_symmetry(
    grid: Tensor,
    *,
    digit_permutation: tuple[int, ...],
    transpose_grid: bool,
    row_permutation: tuple[int, ...] = IDENTITY_LINES,
    col_permutation: tuple[int, ...] = IDENTITY_LINES,
) -> Tensor:
    """Invert :func:`apply_symmetry` exactly.

    Args:
        grid: Transformed token grid.
        digit_permutation: Digit mapping used for the forward transform.
        transpose_grid: Whether the forward transform transposed positions.
        row_permutation: Source row per output row in the forward transform.
        col_permutation: Source column per output column in the forward transform.

    Returns:
        original: Original-space token grid.

    """
    inverse = [0] * 9
    for source, target in enumerate(digit_permutation):
        inverse[target - 1] = source + 1
    lut = _token_lut(tuple(inverse), device=grid.device)
    positioned = grid.long()
    if _moves_positions(row_permutation, col_permutation, transpose_grid):
        index = _position_index(
            row_permutation,
            col_permutation,
            transpose_grid=transpose_grid,
            device=grid.device,
        )
        positioned = positioned[..., torch.argsort(index)]
    restored = torch.where(
        (positioned >= 0) & (positioned < lut.shape[0]),
        lut[positioned.clamp(min=0, max=lut.shape[0] - 1)],
        positioned,
    )
    return restored.to(grid.dtype)


def validate_grid_permutation(name: str, permutation: tuple[int, ...]) -> None:
    """Require a 0..8 permutation whose 3-blocks map onto 3-blocks.

    Args:
      name: Field name used in the error message.
      permutation: Source line per output line.

    Raises:
      ValueError: If the permutation is not band-structure preserving.

    """
    if tuple(sorted(permutation)) != tuple(range(9)):
        raise ValueError(f"{name} must be a permutation of 0..8, got {permutation}.")
    for block in range(3):
        sources = {permutation[block * 3 + offset] // 3 for offset in range(3)}
        if len(sources) != 1:
            raise ValueError(
                f"{name} must map each band/stack onto one band/stack, got "
                f"{permutation}.",
            )


def validate_views(views: Iterable[View]) -> None:
    """Require every recorded view to be an exact Sudoku symmetry.

    Args:
      views: Recorded views to check.

    Raises:
      ValueError: A digit relabeling is not a permutation of 1..9, or a line
        permutation does not map bands onto bands.

    """
    for view in views:
        if tuple(sorted(view.digit_permutation)) != tuple(range(1, 10)):
            raise ValueError(
                f"view {view.name!r}: digit_permutation must be a permutation "
                f"of 1..9, got {view.digit_permutation}.",
            )
        validate_grid_permutation("row_permutation", view.row_permutation)
        validate_grid_permutation("col_permutation", view.col_permutation)


def _moves_positions(
    row_permutation: tuple[int, ...],
    col_permutation: tuple[int, ...],
    transpose_grid: bool,
) -> bool:
    """Return whether any position component is non-identity."""
    identity = tuple(range(9))
    return (
        transpose_grid
        or tuple(row_permutation) != identity
        or tuple(col_permutation) != identity
    )


def _token_lut(digit_permutation: tuple[int, ...], device: torch.device) -> Tensor:
    """Token lookup: pad 0 and blank 1 fixed; token 1+d relabels digit d."""
    table = [0, 1] + [digit_permutation[digit] + 1 for digit in range(9)]
    return torch.tensor(table, device=device, dtype=torch.long)


def _position_index(
    row_permutation: tuple[int, ...],
    col_permutation: tuple[int, ...],
    *,
    transpose_grid: bool,
    device: torch.device,
) -> Tensor:
    """Flat source index per output cell for the position components."""
    base = torch.arange(81, device=device).reshape(9, 9)
    if transpose_grid:
        base = base.T
    rows = torch.tensor(row_permutation, device=device, dtype=torch.long)
    cols = torch.tensor(col_permutation, device=device, dtype=torch.long)
    return base[rows][:, cols].reshape(-1)
