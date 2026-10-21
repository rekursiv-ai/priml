"""Greedy subject folds and fixed cohorts that must leave a training set."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.motor_imagery.data import Trial
from priml.baselines.motor_imagery.folds import FixedCohort, GroupedFolds


def test_subject_folds_place_the_largest_people_first() -> None:
    cfg = GroupedFolds.Config()
    cfg.count = 3
    trials = [
        Trial(
            subject=f"S00{index + 1}",
            trial_id=index,
            label=index % 2,
            windows=torch.zeros(count, 3, 100),
        )
        for index, count in enumerate((2, 4, 5, 7))
    ]
    # Windows 7 and 5 open two folds; 4 then 2 both join the lightest third.
    assert list(cfg.make()(trials)) == [["S004"], ["S003"], ["S001", "S002"]]


@pytest.mark.parametrize("reverse", argvalues=[False, True])
def test_subject_folds_pair_equal_count_people(reverse: bool) -> None:
    cfg = GroupedFolds.Config()
    cfg.count = 3
    counts = (3, 6, 3, 6, 3, 6)
    trials = [
        Trial(
            subject=f"S{index:03d}",
            trial_id=index,
            label=index % 2,
            windows=torch.zeros(count, 2, 4),
        )
        for index, count in enumerate(counts)
    ]
    assert list(cfg.make()(trials[::-1] if reverse else trials)) == [
        ["S004", "S005"],
        ["S002", "S003"],
        ["S000", "S001"],
    ]


@pytest.mark.parametrize("count", [1, 5])
def test_subject_folds_reject_invalid_counts(count: int) -> None:
    cfg = GroupedFolds.Config()
    cfg.count = count
    trials = [
        Trial(
            subject=f"S{index:03d}",
            trial_id=index,
            label=index % 2,
            windows=torch.zeros(3, 2, 4),
        )
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="count must be between 2"):
        list(cfg.make()(trials))


def fixed_cohort_trials() -> list[Trial]:
    """Four people, one cue each."""
    return [
        Trial(
            subject=f"S{index:03d}",
            trial_id=index,
            label=index % 2,
            windows=torch.zeros(3, 2, 4),
        )
        for index in range(4)
    ]


def test_fixed_cohort_is_the_only_fold() -> None:
    cfg = FixedCohort.Config()
    cfg.subjects = ["S001", "S003"]
    assert list(cfg.make()(fixed_cohort_trials())) == [["S001", "S003"]]


@pytest.mark.parametrize(
    "subjects",
    [[], ["S099"], ["S000", "S001", "S002", "S003"]],
    ids=["empty", "unknown", "everyone"],
)
def test_fixed_cohort_must_name_present_people_and_leave_a_training_cohort(
    subjects: list[str],
) -> None:
    cfg = FixedCohort.Config()
    cfg.subjects = subjects
    with pytest.raises(
        ValueError,
        match=r"^Fixed validation people must leave a training cohort\.$",
    ):
        list(cfg.make()(fixed_cohort_trials()))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
