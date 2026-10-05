"""Canonicalized ARC2 voting, checked against source-minted records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import dihedral_transform
from priml.baselines.arcagi2.metric import (
    PassK,
    _canonical,
    _crop,
    _hash,
    _json_grid,
)
from priml.baselines.arcagi2.record_test import assert_matches, reduce
from priml.lib.custom_json import ReadError


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


class Voting(Protocol):
    """The metric surface the recorder drives."""

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record one batch of ballots."""
        ...

    def compute(self) -> Mapping[str, object]:
        """Score the ballots."""
        ...


def write_manifest(root: Path) -> None:
    """Two tasks: one with two test outputs under eight views, one never voted."""
    identifiers = ["<blank>", *(f"multi|||t{i}|||0213456789" for i in range(8))]
    (root / "identifiers.json").write_text(json.dumps(identifiers))
    (root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "multi": {
                    "test": [
                        {"input": [[1, 2]], "output": [[2], [1]]},
                        {"input": [[3]], "output": [[4]]},
                    ],
                },
                "missing": {"test": [{"input": [[5]], "output": [[6]]}]},
            },
        ),
    )


def record_votes(
    root: Path,
    build: Callable[[Path], Voting],
    dihedral: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, Tensor]:
    """Vote eight transformed views and record the inputs and every score.

    Args:
      root: Directory holding the manifest.
      build: Constructs the metric over ``root``.
      dihedral: Applies dihedral transform ``tid`` to a grid.

    Returns:
      record: Input tensors and the computed scores.

    """
    permutation = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
    # Eight views are required to cover every dihedral transform.
    media = torch.zeros(8, 2, 2, dtype=torch.int32)
    predictions = torch.zeros_like(media)
    for index in range(8):
        for storage, grid in ((media, [[1, 2]]), (predictions, [[2], [1]])):
            transformed = dihedral(permutation[np.array(grid)], index).copy()
            transformed_tensor = torch.from_numpy(transformed)
            rows, cols = transformed_tensor.shape
            storage[index, :rows, :cols] = transformed_tensor + 2
    # PassK reserves one leading halt-logit column in packed scores.
    packed = torch.cat([torch.zeros(8, 1), predictions.flatten(1).float()], dim=1)
    batch = {"media": media.flatten(1), "puzzle_identifiers": torch.arange(1, 9)}
    metric = build(root)
    metric.update(packed, **batch)
    return reduce({"packed": packed, **batch, "scores": dict(metric.compute())})


def test_reference_metric(tmp_path: Path) -> None:
    """Eight transformed votes solve one of three outputs across two tasks."""
    write_manifest(tmp_path)
    record = record_votes(
        tmp_path,
        lambda root: PassK.Config(working_dir=root).make(),
        lambda grid, tid: dihedral_transform(grid, tid=tid),
    )
    assert_matches("metric", "votes", record)
    assert record["scores/pass@1"].item() == 0.25
    assert record["scores/strict@1"].item() == 0.0
    assert torch.equal(record["scores/per_output@1"], torch.tensor(1 / 3))


def test_reference_metric_golden_bites(tmp_path: Path) -> None:
    """A vote cast with the wrong transform changes the recorded scores."""
    write_manifest(tmp_path)
    record = record_votes(
        tmp_path,
        lambda root: PassK.Config(working_dir=root).make(),
        lambda grid, tid: dihedral_transform(grid, tid=(tid + 1) % 8),
    )
    with pytest.raises(AssertionError, match="mismatches"):
        assert_matches("metric", "votes", record)


def test_metric_error_and_canonical_helpers() -> None:
    metric = PassK.Config(working_dir="/opt/scratch/absent").make()
    with pytest.raises(ValueError, match="Expected one halt column") as header_error:
        metric.update(
            torch.zeros(2, 3),
            media=torch.zeros(2, 3),
            puzzle_identifiers=torch.zeros(2, dtype=torch.long),
        )
    assert str(header_error.value) == "Expected one halt column followed by the grid"
    with pytest.raises(
        ValueError,
        match="ARC packed grids must be square",
    ) as crop_error:
        _crop(torch.zeros(3, dtype=torch.uint8))
    assert str(crop_error.value) == "ARC packed grids must be square"
    with pytest.raises(ValueError, match="Invalid ARC color"):
        _canonical("task|||t0|||bad", torch.zeros(2, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="Invalid ARC dihedral"):
        _canonical("task|||t9|||0123456789", torch.zeros(2, 3, dtype=torch.uint8))
    assert _hash(torch.zeros(2, 3, dtype=torch.uint8))
    assert _json_grid([[1, 2, 3], [4, 5, 6]]).shape == (2, 3)


def test_canonical_inverts_every_dihedral_view_of_rectangular_grid() -> None:
    grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    permutation = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
    permuted = permutation[grid]
    for tid in range(8):
        augmented = dihedral_transform(permuted, tid=tid)
        task, restored = _canonical(
            f"task|||t{tid}|||0213456789",
            torch.from_numpy(augmented.copy()),
        )
        assert task == "task"
        assert torch.equal(restored, torch.from_numpy(grid))


def test_crop_preserves_color_boundaries_and_stops_at_padding() -> None:
    packed = torch.tensor(
        [[2, 11, 12], [3, 4, 11], [0, 0, 0]],
        dtype=torch.uint8,
    )
    assert torch.equal(
        _crop(packed.flatten()),
        torch.tensor([[0, 9], [1, 2]], dtype=torch.uint8),
    )
    assert (
        _crop(torch.tensor([[0, 2], [3, 4]], dtype=torch.uint8).flatten()).numel() == 0
    )
    widest = torch.tensor([[2, 3, 4], [5, 0, 0], [6, 0, 0]], dtype=torch.uint8)
    assert torch.equal(
        _crop(widest.flatten()),
        torch.tensor([[0, 1, 2]], dtype=torch.uint8),
    )
    assert torch.equal(
        _crop(torch.tensor([[2, 3], [4, 5]], dtype=torch.uint8).flatten()),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8),
    )
    wide_input = torch.tensor([[2, 3], [4, 5]], dtype=torch.int64)
    assert _crop(wide_input.flatten()).dtype == torch.uint8
    with pytest.raises(ReadError):
        _crop(torch.tensor([[2.0, 3.0], [4.0, 5.0]]).flatten())


def test_hash_and_json_grid_use_uint8_boundary() -> None:
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
    assert _hash(values) == _hash(values.to(torch.uint8))
    grid = _json_grid([[1, 2], [3, 4]])
    assert grid.dtype == torch.uint8
    assert torch.equal(grid, values.to(torch.uint8))
    with pytest.raises(ReadError):
        _json_grid([[1, 2], [3, 4.5]])


def test_canonical_rejects_non_permutations_and_returns_uint8() -> None:
    with pytest.raises(ValueError, match="Invalid ARC color permutation: 'abcdefghij'"):
        _canonical("task|||t0|||abcdefghij", torch.zeros(2, 3, dtype=torch.uint8))
    _, restored = _canonical(
        "task|||t0|||0123456789",
        torch.ones(2, 3, dtype=torch.int64),
    )
    assert restored.dtype == torch.uint8


def test_vote_rankings_resolve_ties_by_the_documented_keys(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    input_hash = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    truth_hash = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    other_hash = _hash(torch.tensor([[7]], dtype=torch.uint8))
    metric.votes = {
        "multi": {input_hash: [(other_hash, 0.5), (truth_hash, 0.5)]},
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.0
    assert scores["pass@2"] == 0.25
    assert scores["votes_times_mean_q@1"] == float(truth_hash < other_hash) / 2
    assert scores["votes_times_max_q@1"] == float(truth_hash < other_hash) / 2
    assert scores["votes_times_mean_q@2"] == 0.25
    assert scores["votes_times_max_q@2"] == 0.25


def test_pass_tie_ignores_the_maximum_confidence(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    truth_hash = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    other_hash = _hash(torch.tensor([[7]], dtype=torch.uint8))
    metric.votes = {
        "multi": {
            input_hash: [
                (truth_hash, 0.5),
                (truth_hash, 0.5),
                (other_hash, 0.4),
                (other_hash, 0.6),
            ],
        },
    }
    assert metric.compute()["pass@1"] == 0.25


def test_rankings_use_count_mean_and_max_confidence(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    input_hash = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    truth_hash = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    other_hash = _hash(torch.tensor([[7]], dtype=torch.uint8))
    metric.votes = {
        "multi": {
            input_hash: [
                (truth_hash, 0.6),
                (truth_hash, 0.6),
                (other_hash, 0.2),
                (other_hash, 0.95),
            ],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.25
    assert scores["votes_times_max_q@1"] == 0.0
    assert scores["pass@2"] == 0.25
    assert scores["strict@1"] == 0.0
    assert scores["strict@2"] == 0.0
    assert scores["per_output@1"] == 1 / 3


def test_vote_aggregation_distinguishes_count_from_confidence(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    input_hash = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    truth_hash = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    other_hash = _hash(torch.tensor([[7]], dtype=torch.uint8))
    metric.votes = {
        "multi": {
            input_hash: [
                (truth_hash, 0.7),
                (other_hash, 0.34),
                (other_hash, 0.34),
            ],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.0
    assert scores["pass@2"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.25
    assert scores["votes_times_max_q@1"] == 0.25
    assert scores["per_output@1"] == 0.0


def test_vote_rankings_use_confidence_sum_and_maximum(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_a = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    input_b = _hash(torch.tensor([[3]], dtype=torch.uint8))
    truth_a = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    truth_b = _hash(torch.tensor([[4]], dtype=torch.uint8))
    wrong_a = _hash(torch.tensor([[7]], dtype=torch.uint8))
    wrong_b = _hash(torch.tensor([[6]], dtype=torch.uint8))
    metric.votes = {
        "multi": {
            input_a: [(truth_a, 0.9), (wrong_a, 0.425), (wrong_a, 0.425)],
            input_b: [(truth_b, 0.3), (truth_b, 0.3), (wrong_b, 0.5)],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.5
    assert scores["votes_times_max_q@1"] == 0.5


def test_update_records_halt_confidence_and_validates_identifiers(
    tmp_path: Path,
) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path).make()
    media = torch.tensor([[4, 3, 0, 0]], dtype=torch.int32)
    logits = torch.tensor([[-2.0, 3.0, 0.0, 4.0, 0.0]])
    metric.update(logits, media=media, puzzle_identifiers=torch.tensor([1]))
    input_hash = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    output_hash = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    confidence = torch.sigmoid(torch.tensor(-2.0, dtype=torch.float64)).item()
    assert metric.votes["multi"][input_hash] == [(output_hash, confidence)]

    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "dataset.json").write_text(
        json.dumps({"blank_identifier_id": 1}),
    )
    metric = PassK.Config(working_dir=tmp_path).make()
    metric.update(
        torch.tensor([[-2.0, 4.0, 0.0, 3.0, 0.0]]),
        media=torch.tensor([[3, 4, 0, 0]], dtype=torch.int32),
        puzzle_identifiers=torch.tensor([0]),
    )
    identity_input = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    identity_output = _hash(torch.tensor([[2], [1]], dtype=torch.uint8))
    assert metric.votes["<blank>"][identity_input][0][0] == identity_output

    for invalid_identifier in (-1, len(metric.identifiers)):
        with pytest.raises(
            ValueError,
            match=f"Puzzle identifier {invalid_identifier} is outside the manifest",
        ):
            metric.update(
                logits,
                media=media,
                puzzle_identifiers=torch.tensor([invalid_identifier]),
            )


def test_update_continues_after_blank_identifier_in_same_batch(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    (tmp_path / "identifiers.json").write_text(
        json.dumps(["<blank>", "padding", "task"]),
    )
    dataset_dir = tmp_path / "test"
    dataset_dir.mkdir()
    (dataset_dir / "dataset.json").write_text(
        json.dumps({"blank_identifier_id": 1}),
    )
    metric = PassK.Config(working_dir=tmp_path).make()
    media = torch.tensor([[3, 4, 0, 0], [3, 4, 0, 0]], dtype=torch.int32)
    logits = torch.tensor([[-2.0, 4.0, 3.0, 0.0, 0.0]]).expand(2, -1)

    metric.update(
        logits,
        media=media,
        puzzle_identifiers=torch.tensor([1, 2]),
    )

    input_hash = _hash(torch.tensor([[1, 2]], dtype=torch.uint8))
    output_hash = _hash(torch.tensor([[2, 1]], dtype=torch.uint8))
    confidence = torch.sigmoid(torch.tensor(-2.0, dtype=torch.float64)).item()
    assert metric.votes == {
        "task": {input_hash: [(output_hash, confidence)]},
    }


def test_strict_score_accumulates_across_tasks(tmp_path: Path) -> None:
    write_manifest(tmp_path)
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.votes = {
        "multi": {
            _hash(torch.tensor([[1, 2]], dtype=torch.uint8)): [
                (_hash(torch.tensor([[2], [1]], dtype=torch.uint8)), 0.5),
            ],
            _hash(torch.tensor([[3]], dtype=torch.uint8)): [
                (_hash(torch.tensor([[4]], dtype=torch.uint8)), 0.5),
            ],
        },
    }
    assert metric.compute()["strict@1"] == 0.5


def test_single_task_single_output_scores_keep_unit_denominators(
    tmp_path: Path,
) -> None:
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "task"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {
                "task": {
                    "test": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
                },
            },
        ),
    )
    metric = PassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_grid = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    output_grid = torch.tensor([[4, 3], [2, 1]], dtype=torch.uint8)
    metric.votes = {"task": {_hash(input_grid): [(_hash(output_grid), 0.5)]}}

    scores = metric.compute()

    assert scores["pass@1"] == 1.0
    assert scores["strict@1"] == 1.0
    assert scores["per_output@1"] == 1.0
    assert scores["votes_times_mean_q@1"] == 1.0
    assert scores["votes_times_max_q@1"] == 1.0


def test_state_dict_round_trip_preserves_independent_ballots() -> None:
    metric = PassK.Config().make()
    metric.votes = {
        "task": {
            "input-a": [("output-a", 0.25), ("output-b", 0.75)],
            "input-b": [("output-c", 0.5)],
        },
    }
    state = metric.state_dict()
    metric.reset()
    metric.load_state_dict(state)
    assert metric.state_dict() == state
    state["votes"]["task"]["input-a"][0][0] = "changed"
    assert metric.votes["task"]["input-a"][0][0] == "output-a"


def test_load_state_dict_rejects_malformed_vote_levels() -> None:
    metric = PassK.Config().make()
    malformed_states: tuple[dict[str, object], ...] = (
        {"votes": None},
        {"votes": {"task": None}},
        {"votes": {"task": {"input": None}}},
        {"votes": {"task": {"input": [None]}}},
        {"votes": {"task": {"input": [["digest", None]]}}},
        {"votes": {"task": {"input": [[None, 0.5]]}}},
    )
    for state in malformed_states:
        with pytest.raises(TypeError):
            metric.load_state_dict(state)


def test_global_vote_gather(monkeypatch: pytest.MonkeyPatch) -> None:

    candidate = PassK.Config().make()
    candidate.votes = {"task": {"input": [("output", 0.5)]}}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    def gather(output: list[object], value: object) -> None:
        output[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert candidate._global_votes() == candidate.votes


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
