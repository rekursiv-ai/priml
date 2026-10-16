"""Check held-out evaluation and experiment definitions."""

from pathlib import Path

import pytest
import torch

from priml.baselines.etth1.experiments import exp000 as etth1_exp000
from priml.baselines.oscillator.benchmark import forecast_scores
from priml.baselines.oscillator.experiments import exp000, exp001, exp002, exp_smoke
from priml.baselines.oscillator.scripts.comparison import main
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection
from priml.testing.golden import assert_pprint_golden


def test_config_is_printable_without_running_a_simulator() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_pretraining_fork_changes_only_the_data_allocation() -> None:
    before = exp000()
    after = exp001()
    assert isinstance(before.method, DLinearForecast.Config)
    assert isinstance(after.method, DLinearForecast.Config)
    assert after.method.pretrain_updates == before.method.updates // 2
    after.method.pretrain_updates = before.method.pretrain_updates
    after.experiment_name = before.experiment_name
    assert after == before
    correction = exp002()
    assert isinstance(correction.method, RidgeCorrection.Config)
    assert correction.data == before.data


def test_new_geometry_does_not_change_etth1() -> None:
    old = etth1_exp000().copy_tree().finalize()
    cfg = exp000().copy_tree().finalize()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert cfg.method.step.model.seq_len == cfg.data.seq_len
    assert cfg.method.step.model.pred_len == cfg.data.pred_len
    assert cfg.method.step.model.channels == 2
    assert old.step.model.seq_len == 336
    assert old.step.model.pred_len == 96
    assert old.step.model.channels == 7


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


def test_scores_weight_all_cells_and_keep_original_units() -> None:
    prediction = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float() / 7
    target = torch.zeros_like(prediction)
    # Per-channel scaling broadcasts over trajectories and forecast timesteps.
    scale = torch.tensor([[[2.0, 4.0]]])
    scores = forecast_scores(prediction, target=target, scale=scale)
    assert scores["mse"] == float(prediction.double().square().mean())
    assert scores["mae"] == float(prediction.double().abs().mean())
    assert scores["scaled_mse"] == float((prediction.double() / scale).square().mean())


def test_scores_reject_broadcasting_and_nonfinite_outputs() -> None:
    prediction = torch.zeros(4, 3, 2)
    scale = torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="same nonempty shape"):
        forecast_scores(prediction, target=prediction[:1], scale=scale)
    prediction[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        forecast_scores(prediction, target=prediction, scale=scale)
    with pytest.raises(ValueError, match="positive training scale"):
        forecast_scores(prediction, target=prediction, scale=scale * 0)


def test_cli_refuses_to_overwrite_a_run(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        main(["--directory", str(tmp_path), "--smoke"])


def test_cli_help_and_unknown_flags() -> None:
    with pytest.raises(SystemExit) as help_result:
        main(["--help"])
    assert help_result.value.code == 0
    with pytest.raises(SystemExit) as bad_flag:
        main(["--unknown"])
    assert bad_flag.value.code == 2
