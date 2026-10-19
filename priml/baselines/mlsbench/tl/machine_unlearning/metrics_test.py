"""Tests for the evaluation: accuracy, confidence and the membership AUC."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.tl.machine_unlearning.metrics import (
    Report,
    auc_from_scores,
    evaluate_accuracy,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray


def scores(*values: int) -> NDArray[np.floating]:
    """Return eighths as float32 scores."""
    return np.array(values, dtype=np.float32) * np.float32(1 / 8)


def test_auc_is_one_when_every_member_outscores_every_nonmember() -> None:
    assert auc_from_scores(scores(5, 6), scores(1, 2, 3)) == 1.0
    assert auc_from_scores(scores(1, 2, 3), scores(5, 6)) == 0.0


def test_auc_breaks_ties_by_the_sort_order_not_by_averaging() -> None:
    # Which of a tied pair argsort ranks first is the host's SIMD kernel's
    # choice, but one of them wins it outright; averaging would give 0.5.
    assert auc_from_scores(scores(3), scores(3)) in {0.0, 1.0}
    # Ties within a set leave its rank sum alone: 16 of 20 pairs won.
    assert auc_from_scores(scores(5, 3, 7, 3, 6), scores(2, 4, 4, 1)) == 16 / 20


def test_auc_of_an_empty_set_divides_by_one() -> None:
    assert auc_from_scores(scores(), scores(1, 2)) == 0.0


class Fixed(torch.nn.Module):
    """Returns its stored logits in order, one row per image, and records its mode."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.served = 0
        self.modes: list[bool] = []

    @override
    def forward(self, media: torch.Tensor) -> torch.Tensor:
        self.modes.append(self.training)
        start, self.served = self.served, self.served + len(media)
        return self.logits[start : self.served]


def test_evaluate_accuracy_counts_argmax_hits_and_keeps_top_probabilities() -> None:
    # Five images in four classes; the predictions are classes 0, 1, 2, 3, 0.
    logits = torch.eye(4)[[0, 1, 2, 3, 0]] * torch.tensor(
        [[2.0], [3.0], [1.0], [4.0], [5.0]],
    )
    model = Fixed(logits)
    batches = [
        (torch.zeros(2, 4, 5), torch.tensor([0, 2])),
        (torch.zeros(3, 4, 5), torch.tensor([2, 3, 1])),
    ]
    accuracy, confidence = evaluate_accuracy(model, batches, device=torch.device("cpu"))
    # Labels 0, 2, 2, 3, 1 against predictions 0, 1, 2, 3, 0: three hits.
    assert accuracy == 3 / 5
    probs = (
        torch.cat(
            [torch.softmax(logits[:2], dim=1), torch.softmax(logits[2:], dim=1)],
        )
        .max(dim=1)
        .values
    )
    assert confidence.dtype == np.float32
    assert np.array_equal(confidence, probs.numpy())
    assert model.modes == [False, False]


def test_evaluate_accuracy_of_nothing_is_zero_with_an_empty_float64_array() -> None:
    accuracy, confidence = evaluate_accuracy(
        Fixed(torch.zeros(2, 3)),
        [],
        device=torch.device("cpu"),
    )
    assert accuracy == 0.0
    assert (confidence.dtype, confidence.shape) == (np.float64, (0,))


def test_the_score_rewards_retained_accuracy_and_forgetting() -> None:
    report = Report(
        retain_acc=0.75,
        forget_acc=0.25,
        forget_mia_auc=0.5,
        forget_train_confidence=scores(1),
        forget_test_confidence=scores(2),
    )
    assert report.unlearn_score == pytest.approx((0.75 + 0.75 + 0.5) / 3)
    assert report.metrics_line() == (
        "TEST_METRICS retain_acc=0.7500 forget_acc=0.2500 "
        "forget_mia_auc=0.5000 unlearn_score=0.6667"
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
