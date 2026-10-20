"""Replay complete source-minted sparse classifier states and predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from scipy.sparse import csr_array
from treekle import from_plain

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm.svm import DualCoordinate, LinearSVC, _bounded
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("max_iter", argvalues=[3, 5000])
@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_sparse_classifier_replays_complete_source_state(
    max_iter: int,
    corrupt: bool,
) -> None:
    fit = np.sin(np.arange(35).reshape(5, 7) / 7)
    fit[np.arange(5), np.arange(5)] = 0
    predict = csr_array(np.cos(np.arange(21).reshape(3, 7) / 11))
    targets = np.array([4, 2, 8, 4, 2], dtype=np.int64)
    config = LinearSVC.Config()
    config.optimizer.max_iter = max_iter
    model = config.make().fit(csr_array(fit), targets=targets)
    if corrupt:
        model.coef_[:1, :1] = np.nextafter(model.coef_[:1, :1], np.inf)
    actual = {"predict": torch.from_numpy(model.decision_function(predict))}
    for name, value in cast("dict[str, object]", vars(model)).items():
        if name.endswith("_"):
            assert isinstance(value, (int, np.generic, np.ndarray))
            actual[name] = torch.from_numpy(np.asarray(value))
    prefix = f"{max_iter}/"
    expected = {
        name.removeprefix(prefix): value
        for name, value in read_tensors(_CWD / "testdata" / "linear_svc.pt").items()
        if name.startswith(prefix)
    }
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("coef_:") for item in report), report
    else:
        assert not report, "\n".join(report)


@pytest.mark.parametrize("max_iter", argvalues=[3, 5000])
def test_sparse_prediction_replays_from_complete_captured_fitted_state(
    max_iter: int,
) -> None:
    record = read_tensors(_CWD / "testdata" / "linear_svc.pt")
    model = LinearSVC.Config().make()
    model.n_features_in_ = int(record[f"{max_iter}/n_features_in_"])
    model.classes_ = record[f"{max_iter}/classes_"].numpy()
    model.coef_ = record[f"{max_iter}/coef_"].numpy()
    model.intercept_ = record[f"{max_iter}/intercept_"].numpy()
    # The original fixture fixes three prediction rows and three classes.
    scores = model.decision_function(
        csr_array(np.cos(np.arange(21).reshape(3, 7) / 11)),
    )
    report = mismatches(
        {"scores": record[f"{max_iter}/predict"]},
        actual={"scores": torch.from_numpy(scores)},
    )
    assert not report, report


def test_coordinate_shuffle_matches_complete_source_bounded_draws() -> None:
    golden = read_tensors(_CWD / "testdata" / "linear_svc.pt")
    random = np.random.RandomState(5489)
    limits = from_plain(golden["rng/limits"].tolist(), list[int])
    actual = torch.tensor([_bounded(random, limit=limit) for limit in limits])
    assert not mismatches({"draws": golden["rng/draws"]}, actual={"draws": actual})


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
    features = csr_array(np.arange(20, dtype=np.float64).reshape(4, 5))
    with pytest.raises(ValueError, match="at least three classes"):
        LinearSVC.Config().make().fit(
            features,
            targets=np.arange(4, dtype=np.int64) % classes,
        )


def test_sparse_classifier_rejects_unaligned_rows() -> None:
    features = csr_array(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="feature rows aligned with targets"):
        LinearSVC.Config().make().fit(features, targets=np.arange(4, dtype=np.int64))


@pytest.mark.parametrize("unfitted_columns", argvalues=[0, 5])
def test_sparse_classifier_rejects_unfitted_or_mismatched_feature_columns(
    unfitted_columns: int,
) -> None:
    config = LinearSVC.Config()
    config.optimizer.max_iter = 3
    model = config.make()
    features = csr_array(np.arange(15, dtype=np.float64).reshape(3, 5))
    with pytest.raises(
        ValueError,
        match=r"^Expected fitted classifier feature columns\.$",
    ):
        model.decision_function(csr_array((3, unfitted_columns), dtype=np.float64))
    model.fit(features, targets=np.arange(3, dtype=np.int64))
    with pytest.raises(
        ValueError,
        match=r"^Expected fitted classifier feature columns\.$",
    ):
        model.decision_function(features[:, :2])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
