"""Canonical augmentation voting with all three ARC2 scoring rules."""

from __future__ import annotations

from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, cast, override

from configgle import Fig, Makeable
from torch import Tensor

import torch
import torch.distributed as dist

from priml.baselines.arcagi1.augmentation import (
    ArcSpec,
    ColorDihedral,
    arc_grid_to_np,
    crop_grid,
    grid_hash,
)
from priml.baselines.arcagi1.metric import PerOutputPass, StrictPass, TaskScore
from priml.lib.codec import from_plain, loads
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray

    import numpy as np


class PassK:
    """Count-dominant voting over canonical inputs, normalized by task."""

    class Config(Fig["PassK"]):
        """Prepared test manifest and scoring cutoffs."""

        base_dir: Path | str | None = None
        """Resource root supplied by the training loop."""

        working_dir: Path | str = "/datasets/arc2concept-aug-1000"
        """Directory containing identifiers and canonical test puzzles."""

        pass_ks: tuple[int, ...] = (1, 2, 5, 10, 100, 1000)
        """Attempt budgets to report."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Prepared dataset's packed-grid geometry and vocabulary."""

        transform: Makeable[ColorDihedral] = field(default_factory=ColorDihedral.Config)
        """Color/dihedral policy that encoded the prepared identifiers."""

        @override
        def finalize(self) -> Self:
            if (
                isinstance(self.transform, ColorDihedral.Config)
                and not self.transform.separator
            ):
                self.transform.separator = self.spec.puzzle_id_separator
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.spec = config.spec
        self._transform = config.transform.make()
        self.root = Path(config.working_dir)
        self.votes: dict[str, dict[str, list[tuple[str, float]]]] = {}

    @cached_property
    def identifiers(self) -> list[str]:
        """Prepared identifier table, loaded only when evaluation starts."""
        return from_plain(
            loads((self.root / "identifiers.json").read_text()),
            list[str],
        )

    @cached_property
    def puzzles(self) -> dict[str, object]:
        """Canonical evaluation tasks, including tasks without predictions."""
        return from_plain(
            loads((self.root / "test_puzzles.json").read_text()),
            dict[str, object],
        )

    @cached_property
    def blank_identifier(self) -> int:
        """The same padding identifier honored by the prepared loader.

        Returns:
          blank_identifier: ``dataset.json``'s ``blank_identifier_id``, else 0.

        """
        path = self.root / "test" / "dataset.json"
        metadata = (
            from_plain(loads(path.read_text()), dict[str, object])
            if path.is_file()
            else {}
        )
        return from_plain(metadata.get("blank_identifier_id"), int, default=0)

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
        confidence = from_plain(output[:, 0].double().sigmoid().tolist(), list[float])
        input_rows = _uint8_rows(inputs)
        prediction_rows = _uint8_rows(output[:, 1:])
        idents = from_plain(identifiers.detach().cpu().tolist(), list[int])
        for index, ident in enumerate(idents):
            if ident == self.blank_identifier:
                continue
            if ident < 0 or ident >= len(self.identifiers):
                raise ValueError(f"Puzzle identifier {ident} is outside the manifest")
            task, inverse = self._transform.inverse(self.identifiers[ident])
            canonical_input = inverse(crop_grid(input_rows[index], spec=self.spec))
            canonical_prediction = inverse(
                crop_grid(prediction_rows[index], spec=self.spec),
            )
            self.votes.setdefault(task, {}).setdefault(
                grid_hash(canonical_input),
                [],
            ).append((grid_hash(canonical_prediction), confidence[index]))

    def compute(self) -> dict[str, float]:
        """Report task-mean, all-inputs-strict, and pooled-output pass rates.

        Returns:
          results: Pass rates keyed by ranking or scoring rule and attempt budget.

        """
        votes = self._global_votes()
        rankings = ("pass", "votes_times_mean_q", "votes_times_max_q")
        results = {f"{name}@{k}": 0.0 for name in rankings for k in self.config.pass_ks}
        task_scores: list[TaskScore] = []
        for name, raw_puzzle in self.puzzles.items():
            pairs = from_plain(
                from_plain(raw_puzzle, dict[str, object]).get("test"),
                list[dict[str, object]],
            )
            counts = {
                f"{ranking}@{k}": 0
                for ranking in ("pass", "votes_times_mean_q", "votes_times_max_q")
                for k in self.config.pass_ks
            }
            for pair in pairs:
                records = votes.get(name, {}).get(
                    grid_hash(self._json_grid(pair["input"])),
                    [],
                )
                stats: dict[str, list[float]] = {}
                for digest, confidence in records:
                    tally = stats.get(digest)
                    if tally is None:
                        stats[digest] = [1.0, confidence, confidence]
                    else:
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
                truth = grid_hash(self._json_grid(pair["output"]))
                for ranking, candidates in ranked.items():
                    for k in self.config.pass_ks:
                        counts[f"{ranking}@{k}"] += truth in candidates[:k]
            for key, count in counts.items():
                results[key] += count / max(1, len(pairs))
            task_scores.append(
                TaskScore(
                    tuple(counts[f"pass@{k}"] for k in self.config.pass_ks),
                    len(pairs),
                ),
            )
        for key in results:
            results[key] /= max(1, len(self.puzzles))
        for rule in (StrictPass.Config().make(), PerOutputPass.Config().make()):
            results.update(
                {
                    f"{rule.name}@{k}": rule(task_scores, index)
                    for index, k in enumerate(self.config.pass_ks)
                },
            )
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
        for name, raw_inputs in from_plain(
            state_dict.get("votes"),
            dict[str, object],
        ).items():
            by_input = votes.setdefault(name, {})
            for input_hash, raw_records in from_plain(
                raw_inputs,
                dict[str, object],
            ).items():
                records: list[tuple[str, float]] = []
                for raw_record in from_plain(raw_records, list[list[object]]):
                    digest, confidence = from_plain(raw_record, list[object])
                    records.append(
                        (from_plain(digest, str), from_plain(confidence, float)),
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

    def _json_grid(self, value: object) -> NDArray[np.uint8]:
        """Decode a raw ARC grid from its JSON boundary."""
        return arc_grid_to_np(
            from_plain(value, list[list[int]]),
            max_grid=self.spec.max_grid,
        )


# An integer token tensor narrows to uint8 by wrapping, as the reference voted on.
def _uint8_rows(values: Tensor) -> list[NDArray[np.uint8]]:
    return list(cast("Iterable[NDArray[np.uint8]]", values.to(torch.uint8).numpy()))
