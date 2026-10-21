"""Pin sparse classifier scores, shuffle draws and configuration checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from priml.baselines.name_that_llm.sparse import CsrMatrix
from priml.baselines.name_that_llm.svm import DualCoordinate, LinearSVC, _bounded


if TYPE_CHECKING:
    from numpy.random import RandomState


def test_sparse_scores_round_each_product_before_adding_it() -> None:
    model = LinearSVC.Config().make()
    model.n_features_in_ = 3
    model.classes_ = np.arange(4)
    model.coef_ = np.zeros((4, 3))
    model.coef_[0, :2] = [-1, 1 - 2**-27]
    model.intercept_ = np.arange(4, dtype=np.float64)
    scores = model.decision_function(
        CsrMatrix.from_dense(np.array([[1, 1 + 2**-27, 0], [0, 0, 2]])),
    )
    # A fused second step would leave -2**-54, the rounding error of the product.
    assert np.array_equal(scores, np.tile(model.intercept_, reps=(2, 1)))


@pytest.mark.parametrize(
    ("limit", "threshold", "seed", "rejected"),
    argvalues=[
        # Seed 0's accepted product has a low word exactly at the threshold.
        (3 * 2**30, 2**30, 0, 1),
        (2**31 + 1, 2**31 - 1, 8, 3),
    ],
)
def test_bounded_draw_rejects_biased_products_and_consumes_each_draw(
    limit: int,
    threshold: int,
    seed: int,
    rejected: int,
) -> None:
    # The threshold is 2**32 mod limit: low words below it would bias the draw.
    draws = np.random.RandomState(seed).randint(
        0,
        1 << 32,
        dtype=np.uint32,
        size=rejected + 2,
    )
    raw = cast("list[int]", draws.tolist())
    low = [(value * limit) & 0xFFFFFFFF for value in raw[: rejected + 1]]
    assert [word < threshold for word in low] == [True] * rejected + [False]
    random = np.random.RandomState(seed)
    assert _bounded(random, limit=limit) == raw[rejected] * limit >> 32
    assert random.randint(0, 1 << 32, dtype=np.uint32) == raw[rejected + 1]


def test_solver_shrinks_a_wide_margin_row_then_rechecks_every_row() -> None:
    draws = np.random.RandomState(1308).randint(0, 1 << 32, dtype=np.uint32, size=12)
    # Each bounded draw is accepted with offset zero, so four sweeps keep row order.
    assert all(
        0 < draw * limit < 1 << 32
        for draw, limit in zip(
            cast("list[int]", draws.tolist()),
            [3, 2, 1] * 4,
            strict=True,
        )
    )
    weights, sweeps = (
        DualCoordinate.Config()
        .make()
        .minimize(
            [[(0, 2.0)], [(0, 0.5)], [(0, 4.0)]],
            signs=[1.0, 1.0, 1.0],
            width=1,
            inverse_regularization=2,
            random=np.random.RandomState(1308),
        )
    )
    # C = 2 gives the diagonal 1/4. Sweep 2 ends at the largest gradient 26/17; sweep
    # 3 drops the last row (gradient 3) and finds the rest flat, so sweep 4 rechecks
    # every row. The optimum keeps only the middle row: w = 2 * 1/2 = 1.
    assert (weights, sweeps) == ([1.0], 4)


def test_classifier_reports_the_most_sweeps_any_class_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DualCoordinate, "minimize", _Sweeps(counts=[2, 5, 3]))
    model = LinearSVC.Config().make()
    model.fit(CsrMatrix.from_dense(np.eye(3, 4)), targets=np.arange(3, dtype=np.int64))
    assert model.n_iter_ == 5


@pytest.mark.parametrize("inverse", argvalues=[0, -1, np.nan, np.inf])
def test_sparse_classifier_rejects_invalid_regularization(inverse: float) -> None:
    config = LinearSVC.Config()
    config.inverse_regularization = inverse
    with pytest.raises(ValueError, match="positive finite inverse regularization"):
        config.make()


def test_optimizer_rejects_negative_sweep_count() -> None:
    config = DualCoordinate.Config()
    config.max_iter = -1
    with pytest.raises(ValueError, match="nonnegative sweep budget"):
        config.make()


@pytest.mark.parametrize("tolerance", argvalues=[0, -1, np.nan, np.inf])
def test_optimizer_rejects_invalid_gradient_tolerance(tolerance: float) -> None:
    config = DualCoordinate.Config()
    config.tolerance = tolerance
    with pytest.raises(
        ValueError,
        match="positive finite projected-gradient tolerance",
    ):
        config.make()


@pytest.mark.parametrize("classes", argvalues=[1, 2])
def test_sparse_classifier_rejects_nonmulticlass_targets(classes: int) -> None:
    features = CsrMatrix.from_dense(np.arange(20, dtype=np.float64).reshape(4, 5))
    with pytest.raises(ValueError, match="at least three classes"):
        LinearSVC.Config().make().fit(
            features,
            targets=np.arange(4, dtype=np.int64) % classes,
        )


def test_sparse_classifier_rejects_unaligned_rows() -> None:
    features = CsrMatrix.from_dense(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="feature rows aligned with targets"):
        LinearSVC.Config().make().fit(features, targets=np.arange(4, dtype=np.int64))


@pytest.mark.parametrize("unfitted_columns", argvalues=[0, 5])
def test_sparse_classifier_rejects_unfitted_or_mismatched_feature_columns(
    unfitted_columns: int,
) -> None:
    config = LinearSVC.Config()
    config.optimizer.max_iter = 3
    model = config.make()
    features = CsrMatrix.from_dense(np.arange(15, dtype=np.float64).reshape(3, 5))
    with pytest.raises(
        ValueError,
        match=r"^Expected fitted classifier feature columns\.$",
    ):
        model.decision_function(CsrMatrix.from_dense(np.zeros((3, unfitted_columns))))
    model.fit(features, targets=np.arange(3, dtype=np.int64))
    with pytest.raises(
        ValueError,
        match=r"^Expected fitted classifier feature columns\.$",
    ):
        model.decision_function(features.take_columns(np.arange(2)))


@dataclass(slots=True, kw_only=True)
class _Sweeps:
    """Return zero weights and the next sweep count; a non-function stays unbound."""

    counts: list[int]

    def __call__(
        self,
        rows: list[list[tuple[int, float]]],
        signs: list[float],
        *,
        width: int,
        inverse_regularization: float,
        random: RandomState,
    ) -> tuple[list[float], int]:
        del rows, signs, inverse_regularization, random
        return [0.0] * width, self.counts.pop(0)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
