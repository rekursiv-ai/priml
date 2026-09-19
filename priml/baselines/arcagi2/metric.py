"""Canonical augmentation voting with all three ARC2 scoring rules."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, override

import hashlib
import math

from configgle.fig import Fig
from torch import Tensor

import torch
import torch.distributed as dist

from priml.lib.custom_json import (
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    StrCodec,
    loads,
)
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Mapping


class PassK:
    """Count-dominant voting over canonical inputs, normalized by task."""

    class Config(Fig["PassK"]):
        """Prepared test manifest and scoring cutoffs."""

        base_dir: Path | str | None = None
        """Resource root supplied by the training loop."""

        working_dir: Path | str = "/datasets/arcagi2/arc2concept-aug-1000"
        """Directory containing identifiers and canonical test puzzles."""

        pass_ks: tuple[int, ...] = (1, 2, 5, 10, 100, 1000)
        """Attempt budgets to report."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = Path(config.working_dir)
        self.votes: dict[str, dict[str, list[tuple[str, float]]]] = {}

    @cached_property
    def identifiers(self) -> list[str]:
        """Prepared identifier table, loaded only when evaluation starts."""
        return ListCodec.coerce(
            loads((self.root / "identifiers.json").read_text()),
            str,
        )

    @cached_property
    def puzzles(self) -> dict[str, object]:
        """Canonical evaluation tasks, including tasks without predictions."""
        return DictCodec.coerce(loads((self.root / "test_puzzles.json").read_text()))

    @cached_property
    def blank_identifier(self) -> int:
        """The same padding identifier honored by the prepared loader."""
        path = self.root / "test" / "dataset.json"
        metadata = DictCodec.coerce(loads(path.read_text())) if path.is_file() else {}
        return IntCodec.coerce(metadata.get("blank_identifier_id"), 0)

    def reset(self) -> None:
        """Discard the previous evaluation's ballots."""
        self.votes.clear()

    def update(self, logits: Tensor, **batch: object) -> None:
        """Invert each view and retain its canonical input/prediction ballot.

        Args:
          logits: Packed halt logits followed by predicted grid token ids.
          **batch: Input grids in ``media`` and their ``puzzle_identifiers``.

        """
        media, identifiers = batch["media"], batch["puzzle_identifiers"]
        assert isinstance(media, Tensor)
        assert isinstance(identifiers, Tensor)
        output = logits.detach().cpu()
        inputs = media.detach().cpu()
        header = output.shape[1] - inputs.shape[1]
        if header != 1:
            raise ValueError("Expected one halt column followed by the grid")
        confidence = ListCodec.coerce(output[:, 0].double().sigmoid().tolist(), float)
        predictions = output[:, 1:].to(torch.uint8)
        for index, identifier in enumerate(identifiers.detach().cpu().tolist()):
            ident = IntCodec.coerce(identifier)
            if ident == self.blank_identifier:
                continue
            if ident < 0 or ident >= len(self.identifiers):
                raise ValueError(f"Puzzle identifier {ident} is outside the manifest")
            name = self.identifiers[ident]
            task, canonical_input = _canonical(name, _crop(inputs[index]))
            _, canonical_prediction = _canonical(name, _crop(predictions[index]))
            records = self.votes.setdefault(task, {}).setdefault(
                _hash(canonical_input),
                [],
            )
            records.append(
                (_hash(canonical_prediction), confidence[index]),
            )

    def compute(self) -> dict[str, float]:
        """Report task-mean, all-inputs-strict, and pooled-output pass rates.

        Returns:
          results: Pass rates keyed by ranking or scoring rule and attempt budget.

        """
        votes = self._global_votes()
        names = (
            "pass",
            "strict",
            "per_output",
            "votes_times_mean_q",
            "votes_times_max_q",
        )
        results = {f"{name}@{k}": 0.0 for name in names for k in self.config.pass_ks}
        output_count = 0
        for name, raw_puzzle in self.puzzles.items():
            pairs = ListCodec.mappings(DictCodec.coerce(raw_puzzle)["test"])
            output_count += len(pairs)
            counts = {
                f"{ranking}@{k}": 0
                for ranking in ("pass", "votes_times_mean_q", "votes_times_max_q")
                for k in self.config.pass_ks
            }
            for pair in pairs:
                records = votes.get(name, {}).get(
                    _hash(_json_grid(pair["input"])),
                    [],
                )
                stats: dict[str, list[float]] = {}
                for digest, confidence in records:
                    tally = stats.setdefault(digest, [0.0, 0.0, 0.0])
                    tally[0] += 1.0
                    tally[1] += confidence
                    tally[2] = max(tally[2], confidence)
                for tally in stats.values():
                    tally[1] /= max(1.0, tally[0])
                ranked = {
                    "pass": sorted(
                        stats,
                        key=lambda digest: stats[digest][:2],
                        reverse=True,
                    ),
                    "votes_times_mean_q": sorted(
                        stats,
                        key=lambda digest: (
                            -stats[digest][0] * stats[digest][1],
                            digest,
                        ),
                    ),
                    "votes_times_max_q": sorted(
                        stats,
                        key=lambda digest: (
                            -stats[digest][0] * stats[digest][2],
                            digest,
                        ),
                    ),
                }
                truth = _hash(_json_grid(pair["output"]))
                for ranking, candidates in ranked.items():
                    for k in self.config.pass_ks:
                        counts[f"{ranking}@{k}"] += truth in candidates[:k]
            for key, count in counts.items():
                results[key] += count / max(1, len(pairs))
            for k in self.config.pass_ks:
                count = counts[f"pass@{k}"]
                results[f"strict@{k}"] += count == len(pairs)
                results[f"per_output@{k}"] += count
        for key in results:
            denominator = (
                output_count if key.startswith("per_output@") else len(self.puzzles)
            )
            results[key] /= max(1, denominator)
        return results

    class StateDict(TypedDict):
        """Canonical ballots accumulated so far."""

        votes: dict[str, dict[str, list[list[str | float]]]]

    def state_dict(self) -> StateDict:
        """Return the accumulated canonical ballots.

        Returns:
          state: Rank-local ballots, without replicated gathered state.

        """
        return {
            "votes": {
                name: {
                    input_hash: [[digest, confidence] for digest, confidence in records]
                    for input_hash, records in by_input.items()
                }
                for name, by_input in self.votes.items()
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Validate and restore the nested checkpoint ballot boundary.

        Args:
          state_dict: Checkpoint containing per-task and per-input ballots.

        """
        votes: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for name, raw_inputs in DictCodec.coerce(
            state_dict["votes"],
            default=None,
        ).items():
            by_input = votes.setdefault(name, {})
            for input_hash, raw_records in DictCodec.coerce(
                raw_inputs,
                default=None,
            ).items():
                records: list[tuple[str, float]] = []
                for raw_record in ListCodec.coerce(raw_records, default=None):
                    digest, confidence = ListCodec.coerce(raw_record, default=None)
                    records.append(
                        (
                            StrCodec.coerce(digest, default=None),
                            FloatCodec.coerce(confidence, default=None),
                        ),
                    )
                by_input[input_hash] = records
        self.votes = votes

    def _global_votes(self) -> dict[str, dict[str, list[tuple[str, float]]]]:
        """Merge rank-major ballots without modifying any rank's local state."""
        if not dist.is_initialized():
            return self.votes
        gathered: list[dict[str, dict[str, list[tuple[str, float]]]]] = [
            {} for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, self.votes)
        merged: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for part in gathered:
            for name, by_input in part.items():
                target = merged.setdefault(name, {})
                for input_hash, records in by_input.items():
                    target.setdefault(input_hash, []).extend(records)
        return merged


def _hash(grid: Tensor) -> str:
    """Hash shape and uint8 grid bytes in the reference representation."""
    shape = bytes(grid.shape)
    return hashlib.sha256(shape + grid.to(torch.uint8).numpy().tobytes()).hexdigest()


def _crop(tokens: Tensor) -> Tensor:
    """Recover the largest top-left rectangle containing only color tokens."""
    side = math.isqrt(tokens.numel())
    if side * side != tokens.numel():
        raise ValueError("ARC packed grids must be square")
    grid = tokens.reshape(side, side)
    values = ListCodec.coerce(grid.flatten().tolist(), int)
    area = height = width = 0
    columns = side
    for rows in range(1, side + 1):
        for column in range(1, columns + 1):
            value = values[(rows - 1) * side + column - 1]
            if value < 2 or value >= 12:
                columns = column - 1
                break
        if rows * columns > area:
            area, height, width = rows * columns, rows, columns
    return (grid[:height, :width] - 2).to(torch.uint8)


def _canonical(name: str, grid: Tensor) -> tuple[str, Tensor]:
    """Undo the identifier's dihedral transform and color permutation."""
    if "|||" not in name:
        return name, grid
    original, transform, permutation = name.split("|||")
    if len(permutation) != 10 or set(permutation) != set("0123456789"):
        raise ValueError(f"Invalid ARC color permutation: {permutation!r}")
    tid = int(transform[1:])
    if 0 <= tid <= 3:
        grid = torch.rot90(grid, -tid, (0, 1))
    elif tid == 4:
        grid = grid.flip(1)
    elif tid == 5:
        grid = grid.flip(0)
    elif tid == 6:
        grid = grid.T
    elif tid == 7:
        grid = torch.rot90(grid, 1, (0, 1)).flip(1)
    else:
        raise ValueError(f"Invalid ARC dihedral transform: {tid}")
    inverse = torch.tensor([int(color) for color in permutation]).argsort()
    return original, inverse[grid.long()].to(torch.uint8)


def _json_grid(value: object) -> Tensor:
    """Decode a raw ARC grid from its JSON boundary."""
    rows = [ListCodec.coerce(row, int) for row in ListCodec.coerce(value)]
    return torch.tensor(rows, dtype=torch.uint8)
