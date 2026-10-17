"""Check held-out evaluation and its scores."""

import math

import pytest
import torch

from priml.baselines.oscillator.benchmark import forecast_scores
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection


@pytest.mark.parametrize(
    "ridge",
    [True, pytest.param(False, marks=pytest.mark.compute_training)],
)
def test_test_changes_do_not_change_training_or_reference_budget(ridge: bool) -> None:
    cfg = exp_smoke()
    if ridge:
        cfg.method = RidgeCorrection.Config()
    before_rng = torch.get_rng_state().clone()
    first = cfg.make().run()
    assert torch.equal(before_rng, torch.get_rng_state())
    cfg.data.test.seed += 17
    cfg.data.shifted.amplitude = (3.0, 3.5)
    second = cfg.make().run()
    assert first["training"] == second["training"]
    assert first["scores"] != second["scores"]
    assert first["reference_budget"] == cfg.data.train.count + cfg.data.validation.count
    assert first["timesteps_per_query"] == cfg.data.seq_len + cfg.data.pred_len


def test_study_supplies_geometry_only_to_methods_that_declare_it() -> None:
    cfg = exp_smoke().copy_tree().finalize()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert (cfg.method.seq_len, cfg.method.pred_len) == (5, 3)
    ridge = exp_smoke()
    ridge.method = RidgeCorrection.Config()
    method = ridge.copy_tree().finalize().method
    assert isinstance(method, RidgeCorrection.Config)
    assert not hasattr(method, "seq_len")


def test_scores_weight_all_cells_and_keep_original_units() -> None:
    prediction = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float() / 7
    target = torch.zeros_like(prediction)
    # Per-channel scaling broadcasts over trajectories and forecast timesteps.
    scale = torch.tensor([[[2.0, 4.0]]])
    scores = forecast_scores(prediction, target=target, scale=scale)
    assert scores["mse"] == float(prediction.double().square().mean())
    assert scores["mae"] == float(prediction.double().abs().mean())
    assert scores["scaled_mse"] == float((prediction.double() / scale).square().mean())


def test_scores_reject_broadcasting_and_propagate_nonfinite_outputs() -> None:
    prediction = torch.zeros(4, 3, 2)
    scale = torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="same nonempty shape"):
        forecast_scores(prediction, target=prediction[:1], scale=scale)
    with pytest.raises(ValueError, match="one training scale per channel"):
        forecast_scores(prediction, target=prediction, scale=scale[..., :1])
    target = prediction.clone()
    prediction[0, 0, 0] = float("nan")
    scores = forecast_scores(prediction, target=target, scale=scale)
    assert all(math.isnan(value) for value in scores.values())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
