"""pass@K voting over a puzzle's augmented views.

Each ARC puzzle is evaluated many times -- once per augmented view -- and the
model may answer differently on each. The score is the consensus: group every
prediction for one puzzle, rank the distinct answers, and count the puzzle
solved if the true grid is among the top K.

Ranking is by vote count, with mean halt confidence breaking ties. Count
dominates because agreement across independent views is the stronger signal;
confidence only separates answers that tied.

Predictions are stored as hashes, not grids. A full evaluation is hundreds of
thousands of 900-cell grids, and only equality between them matters.
"""

from __future__ import annotations

from typing import Any

import hashlib

from configgle import Fig
from torch import Tensor

import torch
import torch.distributed as dist


class PassK:
    """Consensus accuracy over each puzzle's augmented views.

    Consumes the packed evaluation output the puzzle train step emits: a halt
    logit in column 0 and the predicted tokens in the last ``grid_len``
    columns, so any diagnostic columns between them are ignored.
    """

    class Config(Fig["PassK"]):
        """Which K values to report, and how a vote is counted."""

        pass_ks: tuple[int, ...] = (1, 2, 5, 10)
        """Report the true grid appearing in the top K ranked answers.

        pass@1 is the headline -- the model's single best guess. Larger K
        measures whether the right answer was present but outvoted, which
        separates a model that cannot solve a task from one that cannot pick
        its own best attempt."""

        ignore_label_id: int = -100
        """Label value marking cells excluded from the comparison.

        Padding rows appended to square off a short batch carry it, so they
        neither count as solved nor as failed."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Drop every accumulated vote."""
        # puzzle id -> answer hash -> [votes, summed confidence]
        self._votes: dict[int, dict[str, list[float]]] = {}
        # puzzle id -> the true answer's hash
        self._truth: dict[int, str] = {}

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record one batch of predictions as votes.

        Args:
          logits: Packed model output; column 0 is the halt logit and the last
            ``grid_len`` columns are the predicted tokens.
          **batch: Must carry ``label`` and ``puzzle_identifiers``;
            ``valid_count`` truncates the padded tail when present.

        """
        label_raw = batch["label"]
        assert isinstance(label_raw, Tensor)
        labels = label_raw.detach().to(torch.int64)
        puzzle_identifiers_raw = batch["puzzle_identifiers"]
        assert isinstance(puzzle_identifiers_raw, Tensor)
        identifiers = puzzle_identifiers_raw.detach().to(torch.int64)
        grid_len = labels.shape[-1]
        packed = logits.detach()
        predictions = packed[:, -grid_len:].to(torch.int64)
        # Confidence in [0, 1] so ties break on a comparable scale.
        confidence = torch.sigmoid(packed[:, 0].float())

        raw_count = batch.get("valid_count", labels.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        labels = labels[:valid_count].to(predictions.device)
        predictions = predictions[:valid_count]
        identifiers = identifiers[:valid_count].to(predictions.device)
        confidence = confidence[:valid_count]

        counted = labels != self.config.ignore_label_id
        for row in range(predictions.shape[0]):
            keep = counted[row]
            if not bool(keep.any()):
                continue  # an all-ignored row is padding, not a puzzle
            puzzle = int(identifiers[row])
            answer = _digest(predictions[row][keep])
            truth = _digest(labels[row][keep])
            self._truth.setdefault(puzzle, truth)
            tally = self._votes.setdefault(puzzle, {}).setdefault(answer, [0.0, 0.0])
            tally[0] += 1.0
            tally[1] += float(confidence[row])

    def compute(self) -> dict[str, float]:
        """Rank each puzzle's answers and score every K."""
        solved = dict.fromkeys(self.config.pass_ks, 0)
        for puzzle, tally in self._votes.items():
            truth = self._truth[puzzle]
            # Count first, then mean confidence: agreement across independent
            # views outranks a single confident view.
            ranked = sorted(
                tally.items(),
                key=lambda item: (item[1][0], item[1][1] / item[1][0]),
                reverse=True,
            )
            for k in self.config.pass_ks:
                if any(answer == truth for answer, _ in ranked[:k]):
                    solved[k] += 1
        counts = torch.tensor(
            [float(len(self._votes)), *(float(solved[k]) for k in self.config.pass_ks)],
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            # NCCL reduces only CUDA tensors; gloo only CPU ones. Move for the
            # former and come back, so ``.tolist()`` works either way.
            if dist.get_backend() != "gloo":
                counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            counts = counts.cpu()
        total, *hits = counts.tolist()
        return {
            f"pass@{k}": hit / max(1.0, total)
            for k, hit in zip(self.config.pass_ks, hits, strict=True)
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the accumulated votes."""
        return {"votes": self._votes, "truth": self._truth}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore votes produced by :meth:`state_dict`."""
        self._votes = state_dict.get("votes", {})
        self._truth = state_dict.get("truth", {})


def _digest(grid: Tensor) -> str:
    """Hash one grid's tokens.

    Only equality between grids matters, and an evaluation holds hundreds of
    thousands of them, so a digest is stored instead of the grid.
    """
    return hashlib.blake2b(
        grid.to(torch.int16).cpu().numpy().tobytes(),
        digest_size=16,
    ).hexdigest()
