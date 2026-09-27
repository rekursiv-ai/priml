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

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypedDict, cast, override

import hashlib
import json

from configgle import Fig, Makeable
from torch import Tensor

import numpy as np
import torch
import torch.distributed

from priml.baselines.arcagi1.augmentation import (
    ColorDihedral,
    canonicalize_arc_grid,
    grid_hash,
)
from priml.lib.custom_json import FloatCodec
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Mapping


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
        # Puzzle id -> answer hash -> [votes, summed confidence].
        self._votes: dict[int, dict[str, list[float]]] = {}
        # Puzzle id -> the true answer's hash.
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
                continue  # An all-ignored row is padding, not a puzzle.
            puzzle = int(identifiers[row])
            answer = _digest(predictions[row][keep])
            truth = _digest(labels[row][keep])
            self._truth.setdefault(puzzle, truth)
            tally = self._votes.setdefault(puzzle, {}).setdefault(answer, [0.0, 0.0])
            tally[0] += 1.0
            tally[1] += float(confidence[row])

    def compute(self) -> dict[str, float]:
        """Rank each puzzle's answers and score every K.

        Returns:
          metrics: Accuracy at each pass@K threshold, keyed as "pass@{k}".

        """
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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # NCCL reduces only CUDA tensors; gloo only CPU ones. Move for the
            # former and come back, so ``.tolist()`` works either way.
            if torch.distributed.get_backend() != "gloo":
                counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            counts = counts.cpu()
        total, *hits = [FloatCodec.coerce(count) for count in counts.tolist()]
        return {
            f"pass@{k}": hit / max(1.0, total)
            for k, hit in zip(self.config.pass_ks, hits, strict=True)
        }

    class StateDict(TypedDict):
        """Per-puzzle vote tallies and the digest of each puzzle's answer."""

        votes: dict[int, dict[str, list[float]]]
        truth: dict[int, str]

    def state_dict(self) -> StateDict:
        """Return the accumulated votes.

        Returns:
          state: The vote tallies and answer digests, keyed by puzzle.

        """
        return {"votes": self._votes, "truth": self._truth}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore votes produced by :meth:`state_dict`.

        Args:
          state_dict: State dict.

        """
        state = cast(PassK.StateDict, state_dict)
        self._votes = state.get("votes", {})
        self._truth = state.get("truth", {})


class _TestPair(TypedDict):
    input: list[list[int]]
    output: list[list[int]]


class _TestTask(TypedDict):
    test: list[_TestPair]


type _CanonicalVotes = dict[str, dict[str, list[tuple[str, float]]]]


class CanonicalPassK:
    """Vote on restored ARC grids, grouping all augmented views by source task."""

    class Config(Fig["CanonicalPassK"]):
        """Source dataset and the spatial view budget used for pass@K."""

        base_dir: Path | str | None = None
        working_dir: Path | str = "/datasets/arc1concept-aug-1000"
        pass_ks: tuple[int, ...] = (1, 2, 5, 10, 100, 1_000)
        spatial_views: Literal["all", "non_spatial"] = "all"
        max_views_per_input: int = 0
        transform: Makeable[ColorDihedral] = field(default_factory=ColorDihedral.Config)

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self._root = Path(config.working_dir)
        self._transform = config.transform.make()
        self._identifiers: list[str] | None = None
        self._test_tasks: dict[str, _TestTask] | None = None
        self.reset()

    def reset(self) -> None:
        """Clear votes while retaining the immutable source metadata."""
        self._votes: _CanonicalVotes = {}

    def update(self, logits: Tensor, **batch: object) -> None:
        """Add each restored prediction as a vote for its canonical test input.

        Args:
          logits: Halt score followed by predicted grid tokens for each row.
          **batch: Input grids, puzzle ids, spatial tags, and valid row count.

        """
        media = batch["media"]
        identifiers = batch["puzzle_identifiers"]
        assert isinstance(media, Tensor)
        assert isinstance(identifiers, Tensor)
        tags = batch.get("spatial_tags")
        if tags is None:
            tags = torch.zeros((media.shape[0], 3), dtype=torch.int64)
            tags[:, 0] = 1
        assert isinstance(tags, Tensor)
        raw_count = batch.get("valid_count", media.shape[0])
        assert isinstance(raw_count, int)
        names, _ = self._load_source()
        if logits.ndim != 2 or logits.shape[1] != media.shape[1] + 1:
            raise ValueError("ARC vote output must contain one halt logit and a grid.")
        predictions = logits[:, -media.shape[1] :].detach().to(torch.int64).cpu()
        confidence = torch.sigmoid(logits[:, 0].detach().to(torch.float64)).cpu()
        media = media.detach().cpu()
        identifiers = identifiers.detach().cpu()
        tags = tags.detach().cpu()
        for row in range(raw_count):
            puzzle_id = int(identifiers[row])
            if puzzle_id == 0:
                continue
            if puzzle_id < 0 or puzzle_id >= len(names):
                raise ValueError(f"Unknown ARC puzzle identifier {puzzle_id}.")
            tag = tags[row]
            if self.config.spatial_views == "non_spatial" and tuple(tag.tolist()) != (
                1,
                0,
                0,
            ):
                continue
            name = names[puzzle_id]
            source_name, source_grid = canonicalize_arc_grid(
                media[row],
                name=name,
                spatial_tags=tag,
                transform=self._transform,
            )
            prediction_name, answer_grid = canonicalize_arc_grid(
                predictions[row],
                name=name,
                spatial_tags=tag,
                transform=self._transform,
            )
            if source_name != prediction_name:
                raise ValueError(
                    "Input and prediction resolved to different ARC tasks.",
                )
            input_hash = _canonical_digest(source_grid)
            answer_hash = _canonical_digest(answer_grid)
            self._votes.setdefault(source_name, {}).setdefault(input_hash, []).append(
                (answer_hash, float(confidence[row])),
            )

    def compute(self) -> dict[str, float]:
        """Report mean per-task pass@K over every known public test pair.

        Returns:
          scores: Mean per-task pass@K scores for the configured ranks.

        """
        _, tasks = self._load_source()
        votes = self._gather_votes()
        solved = dict.fromkeys(self.config.pass_ks, 0.0)
        for name, task in tasks.items():
            pairs = task["test"]
            if not pairs:
                continue
            for pair in pairs:
                source = np.asarray(pair["input"], dtype=np.uint8)
                answer = np.asarray(pair["output"], dtype=np.uint8)
                records = votes.get(name, {}).get(grid_hash(source), [])
                if (
                    self.config.max_views_per_input > 0
                    and len(records) > self.config.max_views_per_input
                ):
                    records = sorted(
                        records,
                        key=lambda item: (-item[1], item[0]),
                    )[: self.config.max_views_per_input]
                tallies: dict[str, list[float]] = {}
                for answer_hash, weight in records:
                    count_and_weight = tallies.setdefault(answer_hash, [0.0, 0.0])
                    count_and_weight[0] += 1.0
                    count_and_weight[1] += weight
                ranked = sorted(
                    tallies,
                    key=lambda digest: (
                        tallies[digest][0],
                        tallies[digest][1] / tallies[digest][0],
                    ),
                    reverse=True,
                )
                truth = grid_hash(answer)
                for k in self.config.pass_ks:
                    solved[k] += float(truth in ranked[:k]) / len(pairs)
        return {
            f"pass@{k}": solved[k] / max(1, len(tasks)) for k in self.config.pass_ks
        }

    class StateDict(TypedDict):
        votes: _CanonicalVotes

    def state_dict(self) -> StateDict:
        """Return votes for checkpointed evaluation."""
        return {"votes": self._votes}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore votes from a previous evaluation."""
        self._votes = cast("_CanonicalVotes", state_dict["votes"])

    def _load_source(self) -> tuple[list[str], dict[str, _TestTask]]:
        if self._identifiers is None:
            self._identifiers = cast(
                "list[str]",
                json.loads((self._root / "identifiers.json").read_text()),
            )
        if self._test_tasks is None:
            self._test_tasks = cast(
                "dict[str, _TestTask]",
                json.loads((self._root / "test_puzzles.json").read_text()),
            )
        return self._identifiers, self._test_tasks

    def _gather_votes(self) -> _CanonicalVotes:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return self._votes
        gathered: list[object] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, self._votes)
        merged: _CanonicalVotes = {}
        for shard in gathered:
            local = cast("_CanonicalVotes", shard)
            for name, by_input in local.items():
                target = merged.setdefault(name, {})
                for input_hash, records in by_input.items():
                    target.setdefault(input_hash, []).extend(records)
        return merged


def _canonical_digest(grid: Tensor) -> str:
    return grid_hash(grid.to(torch.uint8).cpu().numpy())


# Only equality between grids matters, and an evaluation holds hundreds of thousands of
# them, so a digest is stored instead of the grid.
def _digest(grid: Tensor) -> str:
    """Hash one grid's tokens."""
    return hashlib.blake2b(
        grid.to(torch.int16).cpu().numpy().tobytes(),
        digest_size=16,
    ).hexdigest()
