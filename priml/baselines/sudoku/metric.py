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

from typing import Any

from configgle import Fig
from torch import Tensor

import torch
import torch.distributed as dist


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
        """Label value excluded from both accuracies.

        Padding rows appended to square off a short final batch carry it, so
        they neither count as solved nor as failed."""

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
          **batch: Must carry ``label``; ``valid_count`` truncates the padded
            tail when present.

        """
        label_raw = batch["label"]
        assert isinstance(label_raw, Tensor)
        labels = label_raw.detach().to(torch.int64)
        grid_len = labels.shape[-1]
        predictions = logits.detach()[:, -grid_len:].to(torch.int64)
        labels = labels.to(predictions.device)
        raw_count = batch.get("valid_count", labels.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        predictions = predictions[:valid_count]
        labels = labels[:valid_count]

        counted = labels != self.config.ignore_label_id
        correct = (predictions == labels) & counted
        per_puzzle = counted.sum(dim=-1)
        # A puzzle counts as solved only if every counted cell is right, and
        # only if it had cells to begin with -- an all-ignored row is padding.
        solved = (correct.sum(dim=-1) == per_puzzle) & (per_puzzle > 0)
        self.solved += int(solved.sum().item())
        self.puzzles += int((per_puzzle > 0).sum().item())
        self.cells_correct += int(correct.sum().item())
        self.cells += int(per_puzzle.sum().item())

    def compute(self) -> dict[str, float]:
        """Return exact and cell accuracy, summed across ranks first."""
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
        solved, puzzles, cells_correct, cells = counts.tolist()
        return {
            "exact": solved / max(1.0, puzzles),
            "cell": cells_correct / max(1.0, cells),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the accumulated counts."""
        return {
            "solved": self.solved,
            "puzzles": self.puzzles,
            "cells_correct": self.cells_correct,
            "cells": self.cells,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore counts produced by :meth:`state_dict`."""
        self.solved = state_dict.get("solved", 0)
        self.puzzles = state_dict.get("puzzles", 0)
        self.cells_correct = state_dict.get("cells_correct", 0)
        self.cells = state_dict.get("cells", 0)
