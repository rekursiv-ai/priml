"""Rank, standardization and Tanimoto transforms shared by both engines."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from priml.math import stats
from priml.memory import convert_to_tensor


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.casmi2026.custom_types import FloatArray


@overload
def tanimoto(
    intersection: np.float32,
    left: np.float32,
    right: np.float32,
) -> np.float32: ...


@overload
def tanimoto(
    intersection: NDArray[np.float32],
    left: NDArray[np.float32] | np.float32,
    right: NDArray[np.float32] | np.float32,
) -> NDArray[np.float32]: ...


def tanimoto(
    intersection: NDArray[np.float32] | np.float32,
    left: NDArray[np.float32] | np.float32,
    right: NDArray[np.float32] | np.float32,
) -> NDArray[np.float32] | np.float32:
    """Divide shared bits by the union with the source's additive guard.

    Args:
      intersection: Bits set in both fingerprints.
      left: Bits set in the first fingerprint.
      right: Bits set in the second fingerprint.

    Returns:
      similarity: ``intersection / (left + right - intersection + 1e-9)`` in the
        operands' binary32 arithmetic.

    """
    return intersection / (left + right - intersection + 1e-9)  # pyrefly: ignore[bad-return] -- numpy's stubs widen float32 true division to `floating`; it stays float32 at runtime.


def ordinal_ranks(values: FloatArray) -> NDArray[np.float64]:
    """Normalize descending ordinal positions; ties keep encounter order.

    Args:
      values: Candidate scores in encounter order.

    Returns:
      ranks: Zero is the best position and one the worst.

    """
    order = np.argsort(-values, kind="stable")
    ranks = np.asarray(
        stats.ordinal_ranks(values.copy(), order=convert_to_tensor(order)).numpy(),
        np.float64,
    )
    return ranks / max(1, len(values) - 1)


def average_ranks(values: FloatArray) -> NDArray[np.float64]:
    """Assign descending average-tie ranks on a zero-to-one scale.

    Args:
      values: Candidate scores in any encounter order.

    Returns:
      ranks: Average ordinal positions, normalized by the available rank span.

    """
    values = np.asarray(values, np.float64)
    count = len(values)
    present = ~np.isnan(values)
    result = np.empty(count, np.float64)
    result[present] = stats.average_ranks(values[present], descending=True).numpy()
    result[~present] = np.arange(np.count_nonzero(present), count)
    return result / max(1, count - 1)


def population_z(values: FloatArray) -> FloatArray:
    """Standardize float32 engine-2 scores using population variance.

    Args:
      values: Candidate scores; reduction order is their encounter order.

    Returns:
      scores: Centered and scaled values, or zeros below the variance cutoff.

    """
    deviation = values.std()
    return (
        (values - values.mean()) / deviation
        if deviation > 1e-9
        else np.zeros_like(values)
    )
