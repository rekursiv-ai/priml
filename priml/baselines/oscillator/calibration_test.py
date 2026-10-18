"""Check force fitting, held-out forecasts, and the training-data boundary."""

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.oscillator.calibration import (
    CalibratedSimulator,
    CalibrationReport,
)
from priml.baselines.oscillator.data import ForecastData, Windows
from priml.baselines.oscillator.experiments import exp003
from priml.testing.golden import assert_pprint_golden


@pytest.mark.parametrize("cubic", [0.0, 0.17, 0.73])
@pytest.mark.parametrize("substeps", [1, 4])
def test_fit_recovers_force_and_predicts_larger_motion(
    cubic: float,
    substeps: int,
) -> None:
    cfg = _data_config()
    cfg.reference.cubic = cubic
    cfg.reference.substeps = substeps
    cfg.cheap.substeps = substeps
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    method = CalibratedSimulator.Config().make()
    report = method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=data.cheap,
    )
    assert report["calibrated_cubic"] == pytest.approx(cubic, abs=2e-6)
    assert report["training_scaled_mse"] < 1e-12
    assert 1 < report["simulator_evaluations"] <= 65
    assert report["updates"] == report["best_update"] == report["pretrain_updates"] == 0
    assert set(report) == CalibrationReport.__required_keys__
    assert data.cheap.config.cubic == cfg.cheap.cubic == 0.0
    shifted = data.windows(cfg.shifted, simulator=data.reference)
    prediction = method.predict(shifted.media)
    assert prediction.shape == shifted.label.shape
    assert prediction.dtype == shifted.media.dtype
    torch.testing.assert_close(prediction, shifted.label, atol=3e-6, rtol=3e-6)
    if cubic == 0:
        assert torch.equal(prediction, shifted.label)
    else:
        uncorrected = data.cheap.forecast(shifted.media[:, -1], steps=cfg.pred_len)
        assert (prediction - shifted.label).square().mean() < (
            (uncorrected - shifted.label).square().mean() * 1e-6
        )


def test_fit_does_not_read_validation_or_synthetic_data() -> None:
    cfg = _data_config()
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    method = CalibratedSimulator.Config().make()
    expected = method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=data.cheap,
    )
    unused = Windows(
        media=torch.full_like(train.media, float("nan")),
        label=torch.full_like(train.label, float("nan")),
    )
    actual = method.fit(
        train,
        validation=unused,
        synthetic=unused,
        simulator=data.cheap,
    )
    assert actual == expected


def test_test_changes_do_not_change_the_fitted_coefficient() -> None:
    cfg = exp003()
    cfg.data = _data_config()
    cfg.seed = 7
    before = cfg.make().run()
    cfg.data.test.seed = 800
    cfg.data.shifted.seed = 900
    cfg.data.shifted.amplitude = (2.5, 3.0)
    after = cfg.make().run()
    assert before["training"] == after["training"]
    assert before["reference_budget"] == after["reference_budget"]


def test_config_and_search_are_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp003", config=exp003())


def test_search_budget_is_configurable_and_failure_is_reported() -> None:
    cfg = CalibratedSimulator.Config()
    search = cfg.search
    assert isinstance(search, PartialConfig)
    search.options = {"maxiter": 1, "xatol": 1e-6}
    method = cfg.make()
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    with pytest.raises(RuntimeError, match="converged"):
        method.fit(
            train,
            validation=train,
            synthetic=train,
            simulator=data.cheap,
        )
    with pytest.raises(RuntimeError, match="Fit the simulator"):
        method.predict(train.media)


def test_nonfinite_training_windows_fail_the_search() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    nonfinite = Windows(
        media=train.media,
        label=torch.full_like(train.label, float("nan")),
    )
    method = CalibratedSimulator.Config().make()
    with pytest.raises(RuntimeError, match="finite converged"):
        method.fit(nonfinite, validation=train, synthetic=train, simulator=data.cheap)
    with pytest.raises(RuntimeError, match="Fit the simulator"):
        method.predict(train.media)


def test_predict_requires_a_fit() -> None:
    method = CalibratedSimulator.Config().make()
    with pytest.raises(RuntimeError, match="Fit the simulator"):
        method.predict(torch.zeros(3, 4, 2))


def _data_config() -> ForecastData.Config:
    cfg = ForecastData.Config()
    cfg.seq_len = 4
    cfg.pred_len = 5
    cfg.train.count = 3
    cfg.shifted.count = 7
    return cfg


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
