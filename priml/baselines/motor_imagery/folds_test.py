"""Replay the source's subject folds and reject cohorts that leave no training set."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from treekle import from_plain

import pytest
import torch

from priml.baselines.motor_imagery.data import Trial
from priml.baselines.motor_imagery.folds import FixedCohort, GroupedFolds
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent


def test_subject_folds_match_reference_for_every_unequal_group_window() -> None:
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
    windows = [
        (trial.trial_id, window)
        for trial in trials
        for window in range(len(trial.windows))
    ]
    record = read_tensors(_CWD / "testdata" / "subject_folds.pt")
    expected = [
        {windows[index] for index in from_plain(indices.tolist(), list[int])}
        for indices in (record[f"unequal/{fold}"] for fold in range(cfg.count))
    ]
    actual = [
        {
            (trial.trial_id, window)
            for trial in trials
            if trial.subject in people
            for window in range(len(trial.windows))
        }
        for people in cfg.make()(trials)
    ]
    assert actual == expected
    assert {window for fold in actual for window in fold} == set(windows)


@pytest.mark.parametrize("reverse", argvalues=[False, True])
def test_subject_folds_replay_source_equal_count_ties(reverse: bool) -> None:
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
    windows = [
        (trial.subject, window)
        for trial in trials
        for window in range(len(trial.windows))
    ]
    record = read_tensors(_CWD / "testdata" / "subject_folds.pt")
    for fold, people in enumerate(cfg.make()(trials[::-1] if reverse else trials)):
        actual = {
            index for index, (subject, _) in enumerate(windows) if subject in people
        }
        assert actual == set(from_plain(record[f"ties/{fold}"].tolist(), list[int]))


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
