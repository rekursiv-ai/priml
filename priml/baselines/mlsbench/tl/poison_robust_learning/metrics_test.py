"""Tests for the poison-robustness metric."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.tl.poison_robust_learning.data import SPLITS
from priml.baselines.mlsbench.tl.poison_robust_learning.metrics import (
    PoisonRobustness,
)


CLASSES: Final = 3
"""Width of the test logits."""

TEST: Final = torch.tensor(0)
"""The ``split`` tag of a test batch."""

POISONED: Final = torch.tensor(1)
"""The ``split`` tag of a poisoned batch."""


def predicting(classes: list[int]) -> torch.Tensor:
    """Return logits whose argmax is each of ``classes``."""
    return torch.nn.functional.one_hot(torch.tensor(classes), CLASSES).float()


def fed() -> PoisonRobustness:
    """Return a metric fed two test batches and two poisoned ones."""
    metric = PoisonRobustness.Config().make()
    metric.update(predicting([0, 1]), label=torch.tensor([0, 2]), split=TEST)
    metric.update(predicting([2, 2]), label=torch.tensor([2, 2]), split=TEST)
    metric.update(
        predicting([1, 0, 2]),
        label=torch.tensor([1, 1, 2]),
        poison=torch.tensor([True, True, False]),
        split=POISONED,
    )
    metric.update(
        predicting([2, 0]),
        label=torch.tensor([2, 0]),
        poison=torch.tensor([False, False]),
        split=POISONED,
    )
    return metric


def test_the_metric_reports_the_references_three_fractions() -> None:
    test_acc, poison_fit = 3 / 4, 1 / 2
    assert fed().compute() == {
        "test_acc": test_acc,
        "poison_fit": poison_fit,
        "robust_score": (test_acc + (1.0 - poison_fit)) / 2.0,
    }


def test_only_flipped_images_count_toward_the_poison_fit() -> None:
    metric = fed()
    assert (metric.correct[SPLITS[1]], metric.total[SPLITS[1]]) == (1, 2)


def test_ties_resolve_to_the_first_class_as_argmax_does() -> None:
    metric = PoisonRobustness.Config().make()
    metric.update(torch.zeros(2, CLASSES), label=torch.tensor([0, 1]), split=TEST)
    assert metric.correct[SPLITS[0]] == 1


def test_an_empty_split_counts_as_zero_as_the_references_max_one_does() -> None:
    metric = PoisonRobustness.Config().make()
    metric.update(predicting([0]), label=torch.tensor([0]), split=TEST)
    assert metric.compute()["poison_fit"] == 0.0


def test_reset_and_state_dict_round_trip() -> None:
    metric = fed()
    restored = PoisonRobustness.Config().make()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()
    restored.reset()
    assert restored.total == dict.fromkeys(SPLITS, 0)


def test_a_batch_without_its_split_is_refused() -> None:
    metric = PoisonRobustness.Config().make()
    with pytest.raises(KeyError):
        metric.update(predicting([0]), label=torch.tensor([0]))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
