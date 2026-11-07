"""Rank ties, standardization and binary32 Tanimoto."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.scores import (
    average_ranks,
    ordinal_ranks,
    population_z,
    tanimoto,
)
from priml.math import stats


if TYPE_CHECKING:
    from priml.math.custom_types import Tensorable


@pytest.mark.parametrize("count", [0, 1, 7, 33, 129])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ordinal_ranks_reuses_shared_inversion_with_stable_order(
    count: int,
    dtype: type[np.float32 | np.float64],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integers = (np.arange(count) % 5) - 2
    values = (
        integers.astype(np.float32)
        if dtype is np.float32
        else integers.astype(np.float64)
    )[::-1]
    if count > 2:
        values[0] = np.nan
        values[1] = -0.0
        values[2] = 0.0
    order = np.argsort(-values, kind="stable")
    expected = np.empty(count, np.float64)
    expected[order] = np.arange(count)
    observed: list[Tensor | None] = []
    original = stats.ordinal_ranks

    def capture(x: Tensorable, *, order: Tensor | None = None) -> Tensor:
        observed.append(order)
        return original(x, order=order)

    monkeypatch.setattr(stats, "ordinal_ranks", capture)
    actual = ordinal_ranks(values)
    np.testing.assert_array_equal(
        actual.view(np.uint64),
        (expected / max(1, count - 1)).view(np.uint64),
    )
    assert len(observed) == 1
    assert observed[0] is not None
    assert torch.equal(observed[0], torch.from_numpy(order))


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1.0],
        [0.0, -0.0, 3.0, 3.0, -2.0],
        [float("nan"), 3.0, float("nan"), -1.0],
        [float("inf"), -float("inf"), 0.0],
    ],
)
def test_average_ranks_preserves_numpy_missing_and_normalization_policy(
    values: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.asarray(values, np.float64)[::-1]
    present = ~np.isnan(source)
    below = (source[None, :] > source[:, None]).sum(1)
    equal = np.equal(source[None, :], source[:, None]).sum(1)
    expected = below + (equal - 1) / 2
    expected[~present] = np.count_nonzero(present) + np.arange(
        np.count_nonzero(~present),
    )
    observed: list[bool] = []
    original = stats.average_ranks

    def capture(x: Tensorable, *, descending: bool = False) -> Tensor:
        observed.append(descending)
        return original(x, descending=descending)

    monkeypatch.setattr(stats, "average_ranks", capture)
    actual = average_ranks(source)
    np.testing.assert_array_equal(
        actual.view(np.uint64),
        (expected / max(1, len(source) - 1)).view(np.uint64),
    )
    assert observed == [True]


def test_average_ranks_supports_the_empty_population() -> None:
    ranked = average_ranks(np.zeros(0, np.float32))
    assert ranked.dtype == np.float64
    assert ranked.size == 0


def test_population_z_uses_population_variance_in_float32() -> None:
    values = np.array([1.0, 2.0, 3.0, 6.0], np.float32)
    scores = population_z(values)
    assert scores.dtype == np.float32
    # ``ddof=0``: the sample deviation would shrink every score.
    np.testing.assert_allclose(
        scores,
        (values - 3.0) / cast(float, 3.5**0.5),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "values",
    [[5.0, 5.0, 5.0], [1.0, 1.0 + 1e-10], [7.0]],
    ids=["constant", "below-cutoff", "single"],
)
def test_population_z_is_zero_below_the_variance_cutoff(values: list[float]) -> None:
    source = np.asarray(values, np.float64)
    scores = population_z(source)
    np.testing.assert_array_equal(scores, np.zeros_like(source))
    assert scores.dtype == source.dtype


def test_population_z_uses_population_variance_without_changing_input() -> None:
    values = np.array([2.0, 4.0], np.float32)
    actual = population_z(values)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.array([-1.0, 1.0], np.float32))
    np.testing.assert_array_equal(values, np.array([2.0, 4.0], np.float32))


@pytest.mark.parametrize("spread", [0.0, 1e-10, 1e-9])
def test_population_z_returns_zeros_at_or_below_the_cutoff(spread: float) -> None:
    values = np.array([-spread, spread], np.float64)
    actual = population_z(values)

    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(actual, np.zeros_like(values))
    np.testing.assert_array_equal(values, np.array([-spread, spread], np.float64))


def test_tanimoto_keeps_the_additive_guard_in_binary32() -> None:
    shared = np.array([2.0, 0.0], np.float32)
    left = np.array([4.0, 0.0], np.float32)
    right = np.float32(3.0)
    pairs = tanimoto(shared, left, right)
    assert pairs.dtype == np.float32
    np.testing.assert_array_equal(pairs, np.array([np.float32(0.4), 0.0], np.float32))
    single = tanimoto(np.float32(2.0), np.float32(4.0), right)
    assert single == np.float32(0.4)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
