"""Official balanced accuracy with an explicit fixed class vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


def balanced_accuracy(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    classes: int,
) -> float:
    """Return mean per-class recall, requiring every official class."""
    counts = np.bincount(targets, minlength=classes)
    if len(counts) != classes or np.any(cast("NDArray[np.bool_]", counts == 0)):
        raise ValueError("Balanced accuracy requires every class in the split.")
    correct = np.bincount(targets[targets == predictions], minlength=classes)
    return float(np.mean(correct / counts))
