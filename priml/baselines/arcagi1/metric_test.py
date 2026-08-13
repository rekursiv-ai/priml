"""Tests for the pass@K voting metric."""

from __future__ import annotations

import torch

from priml.baselines.arcagi1.metric import PassK


def _packed(predictions: torch.Tensor, halt: torch.Tensor) -> torch.Tensor:
    """Pack a halt logit ahead of the predicted tokens, as the step emits."""
    return torch.cat([halt.reshape(-1, 1), predictions.float()], dim=-1)


def _metric(**overrides: object) -> PassK:
    config = PassK.Config(pass_ks=(1, 2))
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_a_puzzle_solved_by_every_view_passes_at_one() -> None:
    metric = _metric()
    labels = torch.full((3, 9), 3, dtype=torch.int64)
    metric.update(
        _packed(labels.clone(), torch.zeros(3)),
        label=labels,
        puzzle_identifiers=torch.zeros(3, dtype=torch.int64),
    )
    assert metric.compute() == {"pass@1": 1.0, "pass@2": 1.0}


def test_the_majority_answer_wins() -> None:
    """Agreement across views is the signal, so two votes beat one."""
    metric = _metric()
    labels = torch.full((3, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[:2] = 7  # two views agree on a WRONG answer
    metric.update(
        _packed(predictions, torch.zeros(3)),
        label=labels,
        puzzle_identifiers=torch.zeros(3, dtype=torch.int64),
    )
    # The truth was outvoted, but it is still the second-ranked answer.
    assert metric.compute() == {"pass@1": 0.0, "pass@2": 1.0}


def test_confidence_only_breaks_a_tie() -> None:
    """One vote each: the more confident answer ranks first."""
    metric = _metric()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[0] = 7  # a wrong answer, but stated with low confidence
    metric.update(
        _packed(predictions, torch.tensor([-5.0, 5.0])),
        label=labels,
        puzzle_identifiers=torch.zeros(2, dtype=torch.int64),
    )
    assert metric.compute()["pass@1"] == 1.0


def test_votes_are_grouped_per_puzzle() -> None:
    """One puzzle's views must not vote in another's ballot."""
    metric = _metric()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[2:] = 7  # the second puzzle is answered wrongly
    metric.update(
        _packed(predictions, torch.zeros(4)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 0, 1, 1]),
    )
    assert metric.compute()["pass@1"] == 0.5


def test_padding_rows_are_not_puzzles() -> None:
    """Rows squaring off a short batch must not enter the denominator."""
    metric = _metric()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    labels[2:] = -100
    metric.update(
        _packed(labels.clone(), torch.zeros(4)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1, 2, 3]),
    )
    assert metric.compute()["pass@1"] == 1.0


def test_valid_count_truncates_before_voting() -> None:
    metric = _metric()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[2:] = 7
    metric.update(
        _packed(predictions, torch.zeros(4)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1, 2, 3]),
        valid_count=2,
    )
    assert metric.compute()["pass@1"] == 1.0


def test_votes_accumulate_across_batches() -> None:
    """Views of one puzzle arrive in different batches and must still group."""
    metric = _metric()
    labels = torch.full((1, 9), 3, dtype=torch.int64)
    wrong = labels.clone()
    wrong[0, 0] = 7
    identifiers = torch.zeros(1, dtype=torch.int64)
    metric.update(
        _packed(wrong, torch.zeros(1)), label=labels, puzzle_identifiers=identifiers
    )
    metric.update(
        _packed(wrong, torch.zeros(1)), label=labels, puzzle_identifiers=identifiers
    )
    metric.update(
        _packed(labels.clone(), torch.zeros(1)),
        label=labels,
        puzzle_identifiers=identifiers,
    )
    # Two wrong votes against one right: outvoted at K=1, present at K=2.
    assert metric.compute() == {"pass@1": 0.0, "pass@2": 1.0}


def test_grid_is_read_from_the_end() -> None:
    """Diagnostic columns between the halt logit and the grid are ignored."""
    metric = _metric()
    labels = torch.full((1, 9), 3, dtype=torch.int64)
    padded = torch.cat([torch.zeros(1, 6), labels.float()], dim=-1)
    metric.update(
        padded,
        label=labels,
        puzzle_identifiers=torch.zeros(1, dtype=torch.int64),
    )
    assert metric.compute()["pass@1"] == 1.0


def test_empty_metric_reports_zero() -> None:
    assert _metric().compute() == {"pass@1": 0.0, "pass@2": 0.0}


def test_state_round_trips() -> None:
    metric = _metric()
    labels = torch.full((1, 9), 3, dtype=torch.int64)
    metric.update(
        _packed(labels.clone(), torch.zeros(1)),
        label=labels,
        puzzle_identifiers=torch.zeros(1, dtype=torch.int64),
    )
    restored = _metric()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
