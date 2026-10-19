"""Pin group balancing, tie resolution, and complete row order."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from priml.data.folds import grouped_folds


def test_greedy_grouped_folds_preserves_all_rows_and_first_minimum_bins() -> None:
    groups = np.asarray(["b", "a", "c", "b", "d", "a", "b", "e", "d"])
    actual = grouped_folds(groups, count=3)
    expected = [
        ([1, 2, 4, 5, 7, 8], [0, 3, 6]),
        ([0, 2, 3, 4, 6, 8], [1, 5, 7]),
        ([0, 1, 3, 5, 6, 7], [2, 4, 8]),
    ]
    assert [(train.tolist(), valid.tolist()) for train, valid in actual] == expected
    for train, valid in actual:
        assert train.dtype == valid.dtype == np.int64
        assert set(groups[train]).isdisjoint(groups[valid])
        assert sorted([*train.tolist(), *valid.tolist()]) == list(range(len(groups)))


def test_equal_size_groups_follow_numpy_reverse_argsort_even_for_long_ties() -> None:
    groups = np.asarray([f"g{index:02d}" for index in range(40)] * 2)[::-1]
    people, counts = np.unique(groups, return_counts=True)
    assignment = np.empty(len(people), dtype=np.int64)
    for index, group in enumerate(cast("list[int]", np.argsort(counts)[::-1].tolist())):
        assignment[group] = index % 3
    folds = grouped_folds(groups, count=3)
    for fold, (train, valid) in enumerate(folds):
        expected_groups = people[assignment == fold]
        assert np.array_equal(np.unique(groups[valid]), expected_groups)
        assert np.array_equal(valid, np.flatnonzero(np.isin(groups, expected_groups)))
        assert np.array_equal(train, np.flatnonzero(~np.isin(groups, expected_groups)))


@pytest.mark.parametrize("count", [-1, 0, 1, 4])
def test_invalid_fold_counts_are_rejected(count: int) -> None:
    with pytest.raises(
        ValueError,
        match=r"^count must be between 2 and the number of groups\.$",
    ):
        grouped_folds(np.asarray(["a", "b", "c"]), count=count)


def test_empty_groups_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"^count must be between 2 and the number of groups\.$",
    ):
        grouped_folds(np.asarray([], dtype=np.str_), count=2)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
