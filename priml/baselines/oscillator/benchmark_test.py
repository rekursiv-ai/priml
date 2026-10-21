"""Check held-out evaluation and its scores."""

from unittest import mock

import math

import pytest
import torch

from priml.baselines.oscillator.benchmark import (
    ForecastScores,
    StudyRecord,
    forecast_scores,
)
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection


@pytest.mark.parametrize("ridge", [True, False])
def test_test_changes_do_not_change_training_or_reference_budget(ridge: bool) -> None:
    cfg = exp_smoke()
    cfg.seed = 7
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


def test_unseeded_run_records_the_seed_it_drew() -> None:
    cfg = exp_smoke()
    cfg.method = RidgeCorrection.Config()
    drawn = cfg.make().run()["seed"]
    assert drawn != cfg.make().run()["seed"]
    cfg.seed = drawn
    assert cfg.make().run()["seed"] == drawn


@pytest.mark.parametrize("nested_seed", [None, 19])
def test_study_seed_leaves_gpu_generators_alone(nested_seed: int | None) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.step.seed = nested_seed
    cfg.seed = 7
    with mock.patch.object(torch.cuda, "manual_seed_all") as seed_all:
        if nested_seed is None:
            cfg.make().run()
        else:
            with pytest.raises(ValueError, match="seed"):
                cfg.make().run()
    seed_all.assert_not_called()


def test_run_record_matches_its_schema() -> None:
    cfg = exp_smoke()
    cfg.method = RidgeCorrection.Config()
    record = cfg.make().run()
    assert set(record) == StudyRecord.__required_keys__
    assert set(record["scores"]) == {"familiar", "shifted"}
    for split in record["scores"].values():
        assert set(split) == {"method", "cheap_simulator", "last_value"}
        for scores in split.values():
            assert set(scores) == ForecastScores.__required_keys__


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
    # The unit dims are forced: forecast_scores rejects any scale not shaped
    # (1, 1, channels), since one fitted scale serves every trajectory and step.
    scale = torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="same nonempty shape"):
        forecast_scores(prediction, target=prediction[:1], scale=scale)
    with pytest.raises(ValueError, match="one training scale per channel"):
        forecast_scores(prediction, target=prediction, scale=scale[..., :1])
    target = prediction.clone()
    prediction[0, 0, 0] = float("nan")
    scores = forecast_scores(prediction, target=target, scale=scale)
    assert math.isnan(scores["mse"])
    assert math.isnan(scores["mae"])
    assert math.isnan(scores["scaled_mse"])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
