"""Greedy group-balanced cross-validation partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


def grouped_folds[T: np.generic](
    groups: NDArray[T],
    *,
    count: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Balance whole groups across folds, returning rows in their original order.

    Args:
      groups: One group identifier per row.
      count: Number of folds, between two and the number of distinct groups.

    Returns:
      folds: Training and validation row indices for each fold in bin order.

    Raises:
      ValueError: The fold count is outside the supported range.

    """
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if count < 2 or count > len(sizes):
        raise ValueError("count must be between 2 and the number of groups.")
    loads = np.zeros(count)
    assignments = np.empty(len(sizes), dtype=np.int64)
    # NumPy's default argsort tie order is part of the released fold assignment.
    for group in cast("list[int]", np.argsort(sizes)[::-1].tolist()):
        fold = int(np.argmin(loads))
        loads[fold] += sizes[group]
        assignments[group] = fold
    sample_folds = assignments[inverse]
    return [
        (
            np.flatnonzero(cast("NDArray[np.bool_]", sample_folds != fold)),
            np.flatnonzero(cast("NDArray[np.bool_]", sample_folds == fold)),
        )
        for fold in range(count)
    ]
