"""Tests for the grid-accuracy metric."""

from __future__ import annotations

from torch import Tensor

import torch

from priml.baselines.sudoku.metric import GridAccuracy


def _packed(predictions: Tensor, prefix: int = 1) -> Tensor:
    """Pack predictions behind ``prefix`` diagnostic columns."""
    lead = torch.zeros(predictions.shape[0], prefix)
    return torch.cat([lead, predictions.float()], dim=-1)


def test_one_wrong_cell_fails_the_whole_puzzle() -> None:
    """A puzzle is solved or it is not; 80 of 81 is a wrong answer."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[1, 0] = 5
    metric.update(_packed(predictions), label=labels)
    result = metric.compute()
    assert result["exact"] == 0.5
    assert result["cell"] == 17 / 18


def test_grid_is_read_from_the_end() -> None:
    """Leading diagnostic columns must not shift the prediction window."""
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    scores = [
        _score(_packed(labels.clone(), prefix=width), labels) for width in (1, 5, 40)
    ]
    assert scores == [1.0, 1.0, 1.0]


def _score(packed: Tensor, labels: Tensor) -> float:
    metric = GridAccuracy.Config().make()
    metric.update(packed, label=labels)
    return metric.compute()["exact"]


def test_padding_counts_for_neither_side() -> None:
    """Rows squaring off a short batch must not be scored as solved or failed."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    labels[2:] = -100  # the padded tail
    metric.update(_packed(labels.clone()), label=labels)
    assert metric.compute()["exact"] == 1.0
    assert metric.puzzles == 2


def test_valid_count_truncates_before_scoring() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[2:] = 7  # wrong, but past the valid rows
    metric.update(_packed(predictions), label=labels, valid_count=2)
    assert metric.compute()["exact"] == 1.0


def test_counts_accumulate_across_batches() -> None:
    """Ratios are computed once at the end, not averaged per batch."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((1, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels.clone()), label=labels)  # solved
    wrong = labels.clone()
    wrong[0, 0] = 5
    metric.update(_packed(wrong), label=labels)  # not solved
    metric.update(_packed(wrong), label=labels)  # not solved
    assert metric.compute()["exact"] == 1 / 3


def test_empty_metric_reports_zero_not_a_division_error() -> None:
    assert GridAccuracy.Config().make().compute() == {"exact": 0.0, "cell": 0.0}


def test_state_round_trips() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((1, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels.clone()), label=labels)
    state = metric.state_dict()

    restored = GridAccuracy.Config().make()
    restored.load_state_dict(state)
    assert restored.compute() == metric.compute()


def test_reset_clears_every_count() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((1, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels.clone()), label=labels)
    metric.reset()
    assert metric.compute() == {"exact": 0.0, "cell": 0.0}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
