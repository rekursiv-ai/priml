"""Round a fractional ConvexTok solution to a vocabulary.

Upstream ranks only candidates whose indicator is positive, using Python's
stable sort, so equal values keep candidate order. Each scheme returns the
chosen candidates' positions in ascending (candidate) order; the byte alphabet
is added separately when the tokenizer is exported.
"""

from collections.abc import Sequence
from typing import Protocol

from torch import Tensor

import torch


class RoundingFn(Protocol):
    """Choose candidate positions from their indicator values under a budget."""

    def __call__(
        self,
        indicators: Tensor,
        candidates: Sequence[str],
        /,
        *,
        budget: int,
    ) -> Tensor:
        """Return the chosen candidate positions, ascending."""
        ...


def deterministic_rounding(
    indicators: Tensor,
    candidates: Sequence[str],
    /,
    *,
    budget: int,
) -> Tensor:
    """Keep the ``budget`` largest positive indicators (upstream's ``det``).

    Args:
      indicators: Solution value per candidate.
      candidates: Candidate tokens, in indicator order.
      budget: Most candidates to keep.

    Returns:
      chosen: Kept candidate positions, ascending.

    """
    del candidates
    return _top_positive(indicators, indicators, budget=budget)


def biased_rounding(
    indicators: Tensor,
    candidates: Sequence[str],
    /,
    *,
    budget: int,
) -> Tensor:
    """Keep the ``budget`` positive candidates with the largest value per byte (``bias``).

    Args:
      indicators: Solution value per candidate.
      candidates: Candidate tokens, in indicator order; their lengths divide the values.
      budget: Most candidates to keep.

    Returns:
      chosen: Kept candidate positions, ascending.

    """
    lengths = torch.tensor([len(token) for token in candidates])
    return _top_positive(indicators, indicators / lengths, budget=budget)


def integral_rounding(
    indicators: Tensor,
    candidates: Sequence[str],
    /,
    *,
    budget: int,
) -> Tensor:
    """Keep candidates whose indicator is at least 0.99 (upstream's ``all_ones``).

    Args:
      indicators: Solution value per candidate.
      candidates: Candidate tokens, in indicator order.
      budget: Unused: the kept set is whatever the solution already decided.

    Returns:
      chosen: Kept candidate positions, ascending.

    """
    del candidates, budget
    return torch.nonzero(indicators >= 0.99).squeeze(1)


def _top_positive(indicators: Tensor, keys: Tensor, *, budget: int) -> Tensor:
    """Rank positive indicators by ``keys``, stably, and keep the first ``budget``."""
    positive = torch.nonzero(indicators > 0).squeeze(1)
    order = torch.sort(keys[positive], descending=True, stable=True).indices
    return torch.sort(positive[order[:budget]]).values
