"""Tests for the best-and-final accuracy metric."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy


CLASSES: Final = 3
"""Width of the test logits."""


def feed(metric: BestAccuracy, *, correct: int, total: int) -> None:
    """Report an evaluation batch with exactly ``correct`` of ``total`` right."""
    logits = torch.zeros(total, CLASSES)
    logits[:correct, 0] = 1.0
    logits[correct:, 1] = 1.0
    metric.update(logits, label=torch.zeros(total, dtype=torch.int64))


def test_final_is_the_percentage_of_correct_predictions() -> None:
    metric = BestAccuracy.Config().make()
    feed(metric, correct=3, total=4)
    assert metric.compute()["final"] == 100.0 * 3 / 4


def test_counts_accumulate_across_batches() -> None:
    metric = BestAccuracy.Config().make()
    feed(metric, correct=2, total=3)
    feed(metric, correct=0, total=2)
    assert metric.compute()["final"] == 100.0 * 2 / 5


def test_best_survives_a_reset_and_final_does_not() -> None:
    metric = BestAccuracy.Config().make()
    feed(metric, correct=3, total=4)
    _ = metric.compute()
    metric.reset()
    feed(metric, correct=1, total=4)
    assert metric.compute() == {"final": 25.0, "best": 75.0}


def test_best_follows_a_later_improvement() -> None:
    metric = BestAccuracy.Config().make()
    for correct in (1, 4, 2):
        metric.reset()
        feed(metric, correct=correct, total=5)
        result = metric.compute()
    assert result == {"final": 40.0, "best": 80.0}


def test_best_starts_at_zero() -> None:
    metric = BestAccuracy.Config().make()
    feed(metric, correct=0, total=2)
    assert metric.compute() == {"final": 0.0, "best": 0.0}


def test_ties_between_classes_resolve_to_the_first() -> None:
    metric = BestAccuracy.Config().make()
    metric.update(torch.zeros(2, CLASSES), label=torch.tensor([0, 1]))
    assert metric.compute()["final"] == 50.0


def test_state_dict_round_trips_counts_and_best() -> None:
    metric = BestAccuracy.Config().make()
    feed(metric, correct=3, total=4)
    _ = metric.compute()
    metric.reset()
    feed(metric, correct=1, total=2)
    restored = BestAccuracy.Config().make()
    restored.load_state_dict(metric.state_dict())
    assert restored.state_dict() == {"correct": 1, "total": 2, "best": 75.0}
    assert restored.compute() == {"final": 50.0, "best": 75.0}


def test_an_empty_evaluation_is_refused() -> None:
    metric = BestAccuracy.Config().make()
    with pytest.raises(ValueError, match="no examples"):
        metric.compute()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
