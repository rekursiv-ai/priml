"""Pin the sigmoid calibration's likelihood, its starting prior and its branches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from priml.baselines.name_that_llm.calibration import (
    SigmoidCalibration,
    _binary_points,
    _Objective,
)
from priml.baselines.name_that_llm.logistic import LBFGS


def test_smoothed_likelihood_matches_its_closed_form() -> None:
    objective = _Objective(
        scores=np.array([1.0, 2.0, 4.0]),
        targets=np.array([0, 1, 0.25]),
    )
    # Zero logits give each row the gradient (1 - 2t) / 2 in its logit.
    _, gradient = objective(np.zeros(2))
    assert gradient.tolist() == [-0.5, -0.25]
    # An offset of 1000 underflows exp(-1000): each row costs 1000 t.
    loss, gradient = objective(np.array([0.0, 1000.0]))
    assert loss == 1000 * 1.25
    assert gradient.tolist() == [1 * 0 + 2 * 1 + 4 * 0.25, 1.25]


def test_sigmoid_without_updates_starts_at_the_smoothed_class_prior() -> None:
    config = SigmoidCalibration.Config()
    config.optimizer.max_iter = 0
    model = config.make().fit(
        np.sin(np.arange(7) / 3),
        targets=np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.int64),
    )
    # Four negatives and three positives: the offset starts at log((4 + 1) / (3 + 1)).
    assert (model.a_, model.b_) == (0.0, math.log(5 / 4))


def test_binary_likelihood_takes_the_stable_branch_on_each_side_of_every_edge() -> None:
    # The branches agree mathematically, so only exact float64 bits tell them apart.
    scores = np.array(
        [-37, np.nextafter(-37, 0), -2, np.nextafter(-2, 0), 18, np.nextafter(18, 19)],
    )
    targets = np.full(6, 6 / 7)
    loss, gradient = _binary_points(scores, targets=targets)
    below, above = np.exp(scores[:3]), np.exp(-scores[3:])
    expected_loss = np.concatenate(
        [
            [below[0] - targets[0] * scores[0]],
            np.log1p(below[1:]) - targets[1:3] * scores[1:3],
            np.log1p(above[:2]) + (1 - targets[3:5]) * scores[3:5],
            [above[2] + (1 - targets[5]) * scores[5]],
        ],
    )
    expected_gradient = np.concatenate(
        [
            [below[0] - targets[0]],
            ((1 - targets[1:3]) * below[1:] - targets[1:3]) / (1 + below[1:]),
            ((1 - targets[3:]) - targets[3:] * above) / (1 + above),
        ],
    )
    np.testing.assert_array_equal(loss, expected_loss)
    np.testing.assert_array_equal(gradient, expected_gradient)


def test_unfitted_calibration_predicts_one_half_everywhere() -> None:
    model = SigmoidCalibration.Config().make()
    assert (model.a_, model.b_) == (0.0, 0.0)
    np.testing.assert_array_equal(model.predict(np.array([-3.0, 4.0])), [0.5, 0.5])


def test_sigmoid_predictions_follow_the_logistic_on_both_sides_of_zero() -> None:
    model = SigmoidCalibration.Config().make()
    model.a_ = -1.0
    exponential = np.exp(-0.5)
    np.testing.assert_array_equal(
        model.predict(np.array([-0.5, 0.5])),
        [exponential / (1 + exponential), 1 / (1 + exponential)],
    )


def test_sigmoid_rescales_scores_whose_magnitude_reaches_the_threshold() -> None:
    scores = np.sin(np.arange(7) / 3)
    targets = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.int64)
    scale = float(np.abs(scores).max())
    config = SigmoidCalibration.Config()
    config.max_magnitude = scale
    rescaled = config.make().fit(scores, targets=targets)
    unscaled = SigmoidCalibration.Config().make().fit(scores / scale, targets=targets)
    assert (rescaled.a_, rescaled.b_) == (unscaled.a_ / scale, unscaled.b_)


def test_sigmoid_fits_a_strided_score_column_like_its_copy() -> None:
    scores = np.stack([np.sin(np.arange(7) / 3), np.cos(np.arange(7) / 5)], axis=1)
    targets = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.int64)
    assert not scores[:, 1].flags.c_contiguous
    strided, copied = (
        SigmoidCalibration.Config().make().fit(column, targets=targets)
        for column in (scores[:, 1], scores[:, 1].copy())
    )
    assert (strided.a_, strided.b_) == (copied.a_, copied.b_)


def test_sigmoid_accepts_a_threshold_below_one() -> None:
    config = SigmoidCalibration.Config()
    config.max_magnitude = 0.5
    assert config.make().config.max_magnitude == 0.5


@pytest.mark.parametrize("maximum", argvalues=[0, -1, np.nan, np.inf])
def test_sigmoid_rejects_invalid_rescaling_threshold(maximum: float) -> None:
    config = SigmoidCalibration.Config()
    config.max_magnitude = maximum
    with pytest.raises(
        ValueError,
        match=r"^Expected a positive finite calibration rescaling threshold\.$",
    ):
        config.make()


@pytest.mark.parametrize(("rows", "labels"), argvalues=[(0, 0), (2, 3)])
def test_sigmoid_rejects_empty_or_unaligned_scores(rows: int, labels: int) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Expected nonempty calibration scores aligned with targets\.$",
    ):
        SigmoidCalibration.Config().make().fit(
            np.zeros(rows, dtype=np.float64),
            targets=np.zeros(labels, dtype=np.int64),
        )


def test_sigmoid_predictions_saturate_without_overflow_and_keep_shape() -> None:
    model = SigmoidCalibration.Config().make()
    model.a_, model.b_ = -1.0, 0.0
    scores = np.array([[-np.inf, -1000, 0], [1000, np.inf, np.nan]])
    np.testing.assert_array_equal(
        model.predict(scores),
        np.array([[0, 0, 0.5], [1, 1, np.nan]]),
    )


def test_default_calibration_keeps_the_shared_solver_and_source_budgets() -> None:
    config = SigmoidCalibration.Config()
    assert isinstance(config.optimizer.make(), LBFGS)
    assert config.optimizer.max_iter == 15_000
    assert config.optimizer.tolerance_gradient == 1e-6


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
