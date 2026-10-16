"""Extra supervision where a maze path could take a wrong turn.

Exact match fails a grid that marks a valid shortest path other than the stored
one, and most cells cannot cause that: a wall or a far-off open cell is easy.
The hard cells are the path itself and the open cells beside it -- the places a
different route would branch off. :class:`CanonicalBoundary` adds a token loss
over just those cells, so they are not drowned out by the hundreds of easy
cells the full-grid loss averages them with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Fig

import torch

from priml.baselines.maze.data import GOAL, OPEN, PATH, START


if TYPE_CHECKING:
    from torch import Tensor


def canonical_boundary_mask(labels: Tensor, *, grid_shape: tuple[int, int]) -> Tensor:
    """Select the labeled path and the open cells four-adjacent to it.

    Args:
      labels: ``[B, H * W]`` label tokens.
      grid_shape: ``(H, W)``.

    Returns:
      mask: ``[B, H * W]`` bool; true on :data:`PATH` cells and on :data:`OPEN`
        cells beside a start, goal, or path cell.

    """
    grid = labels.reshape(-1, *grid_shape)
    route = (grid == START) | (grid == GOAL) | (grid == PATH)
    pad = torch.nn.functional.pad
    beside = (
        pad(route[:, :, 1:], (0, 1))
        | pad(route[:, :, :-1], (1, 0))
        | pad(route[:, 1:, :], (0, 0, 0, 1))
        | pad(route[:, :-1, :], (0, 0, 1, 0))
    )
    return ((grid == PATH) | ((grid == OPEN) & beside)).reshape(labels.shape)


class CanonicalBoundary:
    """Average the token loss over :func:`canonical_boundary_mask`, per maze."""

    class Config(Fig["CanonicalBoundary"]):
        """Weight and the grid the mask is laid out on."""

        weight: float = 0.1
        """Multiplier on the boundary loss before it joins the token loss."""

        grid_shape: tuple[int, int] = (0, 0)
        """Rows and columns; inherited from the dataset."""

    def __init__(self, config: Config) -> None:
        if min(config.grid_shape) < 1:
            raise ValueError(
                f"grid_shape must be positive; got {config.grid_shape}. It is "
                "normally inherited from the dataset during finalize.",
            )
        self.weight = config.weight
        self.grid_shape = config.grid_shape

    def __call__(self, per_token: Tensor, labels: Tensor) -> Tensor:
        """Return the ``[B]`` mean of ``per_token`` over each maze's boundary cells.

        Args:
          per_token: ``[B, H * W]`` token losses.
          labels: ``[B, H * W]`` label tokens.

        Returns:
          per_sample: Boundary mean; 0 for a maze with no boundary cell.

        """
        focus = canonical_boundary_mask(labels, grid_shape=self.grid_shape)
        return (per_token * focus).sum(dim=-1) / focus.sum(dim=-1).clamp(min=1)
