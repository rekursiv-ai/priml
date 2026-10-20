"""The official score gives each authoring model equal weight."""

from __future__ import annotations

import numpy as np
import pytest

from priml.baselines.name_that_llm.metric import balanced_accuracy


def test_balanced_accuracy_is_mean_recall_not_row_accuracy() -> None:
    targets = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    predictions = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    assert (
        balanced_accuracy(targets, predictions=predictions, classes=3)
        == (1 + 0.5 + 0) / 3
    )


def test_missing_class_cannot_silently_change_the_metric() -> None:
    with pytest.raises(ValueError, match="every class"):
        balanced_accuracy(
            np.array([0, 0]),
            predictions=np.array([0, 0]),
            classes=3,
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
