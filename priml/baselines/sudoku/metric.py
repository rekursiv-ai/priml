"""Whole-grid and per-cell accuracy for structured-output puzzles.

A puzzle is solved or it is not: 80 of 81 correct cells is a wrong answer. So
the headline number is the fraction of puzzles solved EXACTLY, and per-cell
accuracy is reported beside it as a progress signal -- early in training exact
accuracy sits at zero for a long time while cell accuracy climbs, and watching
only the former makes a learning run look dead.

Counts accumulate as integers and are all-reduced once at ``compute``, so a
distributed run reports the same number as a single process rather than a mean
of per-rank ratios (which would weight a short final shard equally with a full
one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor

import torch
import torch.distributed as dist

from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Mapping


class GridAccuracy:
    """Exact-grid and per-cell accuracy over a batched grid prediction.

    Consumes the packed evaluation output a puzzle train step emits: the grid
    predictions are the LAST ``grid_len`` columns, so any leading diagnostic
    columns (a halt logit, per-step traces) are ignored without this metric
    needing to know how many there are.
    """

    class Config(Fig["GridAccuracy"]):
        """Which label value marks a cell as not counting."""

        ignore_label_id: int = -100
        """Label value excluded from both accuracies; a row of only this value
        counts as no puzzle at all."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Zero every count."""
        self.solved = 0
        self.puzzles = 0
        self.cells_correct = 0
        self.cells = 0

    def update(self, logits: Tensor, **batch: object) -> None:
        """Accumulate one batch.

        Args:
          logits: Packed model output; the last ``grid_len`` columns are the
            predicted tokens.
          **batch: Must carry ``label`` and ``valid_count``. The loaders pad a
            short final batch with zero labels, which only ``valid_count``
            tells apart from real puzzles, so it is required.

        """
        label_raw = batch["label"]
        assert isinstance(label_raw, Tensor)
        labels = label_raw.detach().to(torch.int64)
        grid_len = labels.shape[1]
        predictions = logits.detach()[:, -grid_len:].to(torch.int64)
        labels = labels.to(predictions.device)
        valid_count = batch["valid_count"]
        assert isinstance(valid_count, int)
        predictions = predictions[:valid_count]
        labels = labels[:valid_count]

        counted = labels != self.config.ignore_label_id
        correct = (predictions == labels) & counted
        per_puzzle = counted.sum(dim=1)
        # A puzzle counts as solved only if every counted cell is right, and
        # only if it had cells to begin with -- an all-ignored row is padding.
        solved = (correct.sum(dim=1) == per_puzzle) & (per_puzzle > 0)
        self.solved += int(solved.sum().item())
        self.puzzles += int((per_puzzle > 0).sum().item())
        self.cells_correct += int(correct.sum().item())
        self.cells += int(per_puzzle.sum().item())

    def compute(self) -> dict[str, float]:
        """Return exact and cell accuracy, summed across ranks first.

        Returns:
          result: Dict with exact and cell accuracy floats.

        """
        counts = torch.tensor(
            [self.solved, self.puzzles, self.cells_correct, self.cells],
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            # NCCL reduces only CUDA tensors; gloo only CPU ones. Move for the
            # former and come back, so ``.tolist()`` works either way.
            if dist.get_backend() != "gloo":
                counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            counts = counts.cpu()
        solved, puzzles, cells_correct, cells = [
            from_plain(count, float) for count in counts.tolist()
        ]
        return {
            "exact": solved / max(1.0, puzzles),
            "cell": cells_correct / max(1.0, cells),
        }

    class StateDict(TypedDict):
        """The four running counts."""

        solved: int
        puzzles: int
        cells_correct: int
        cells: int

    def state_dict(self) -> StateDict:
        """Return the accumulated counts.

        Returns:
          result: Counters for solved puzzles, total, correct cells, total cells.

        """
        return {
            "solved": self.solved,
            "puzzles": self.puzzles,
            "cells_correct": self.cells_correct,
            "cells": self.cells,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore counts produced by :meth:`state_dict`.

        Args:
          state_dict: State dict.

        """
        state = cast(GridAccuracy.StateDict, state_dict)
        self.solved = _read_count(state, "solved")
        self.puzzles = _read_count(state, "puzzles")
        self.cells_correct = _read_count(state, "cells_correct")
        self.cells = _read_count(state, "cells")


def _read_count(state: Mapping[str, object], key: str) -> int:
    """Read one checkpoint count, preserving absent-field defaults."""
    if key not in state:
        return 0
    value = state[key]
    if isinstance(value, str):
        return from_plain(value, int, strict=False)
    if isinstance(value, float):
        converted = from_plain(value, float)
        if converted.is_integer():
            return int(converted)
    return from_plain(value, int)
