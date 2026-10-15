"""Check training budgets, normalization, and simulator correction."""

import pytest
import torch

from priml.baselines.oscillator.data import Windows
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection


@pytest.mark.compute_training
@pytest.mark.parametrize("pretrain_updates", [0, 1])
def test_budget_includes_pretraining_and_final_validation(
    pretrain_updates: int,
) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.pretrain_updates = pretrain_updates
    cfg = cfg.finalize()
    assert isinstance(cfg.method, DLinearForecast.Config)
    data = cfg.data.make()
    train = data.windows(cfg.data.train, simulator=data.reference)
    validation = data.windows(cfg.data.validation, simulator=data.reference)
    synthetic = data.windows(cfg.data.synthetic, simulator=data.cheap)
    method = cfg.method.make()
    report = method.fit(
        train,
        validation=validation,
        synthetic=synthetic,
        simulator=data.cheap,
    )
    assert report["updates"] == cfg.method.updates
    assert report["best_update"] == cfg.method.updates
    values = torch.cat((train.media, train.label), dim=1)
    assert method.mean is not None
    assert method.scale is not None
    assert torch.equal(method.mean, values.mean(dim=(0, 1), keepdim=True))
    assert torch.equal(method.scale, values.std(dim=(0, 1), correction=0, keepdim=True))
    before = {
        name: value.clone() for name, value in method.step.model.state_dict().items()
    }
    assert torch.isfinite(method.predict(validation.media)).all()
    for name, value in method.step.model.state_dict().items():
        assert torch.equal(before[name], value)
    with pytest.raises(RuntimeError, match="fresh forecaster"):
        method.fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )


def test_ridge_correction_is_zero_when_simulator_matches() -> None:
    cfg = exp_smoke()
    cfg.data.reference = cfg.data.cheap.copy_tree()
    data = cfg.data.make()
    train = data.windows(cfg.data.train, simulator=data.reference)
    validation = data.windows(cfg.data.validation, simulator=data.reference)
    synthetic = data.windows(cfg.data.synthetic, simulator=data.cheap)
    method = RidgeCorrection.Config().make()
    method.fit(train, validation=validation, synthetic=synthetic, simulator=data.cheap)
    assert torch.equal(method.predict(validation.media), validation.label)


def test_ridge_correction_learns_a_constant_residual_without_validation_targets() -> (
    None
):
    cfg = exp_smoke()
    data = cfg.data.make()
    train = data.windows(cfg.data.train, simulator=data.cheap)
    validation = data.windows(cfg.data.validation, simulator=data.cheap)
    train = Windows(media=train.media, label=train.label + 0.25)
    validation.label.fill_(float("nan"))
    method = RidgeCorrection.Config().make()
    method.fit(train, validation=validation, synthetic=train, simulator=data.cheap)
    expected = (
        data.cheap.forecast(validation.media[:, -1], steps=cfg.data.pred_len) + 0.25
    )
    torch.testing.assert_close(method.predict(validation.media), expected)


def test_pretraining_cannot_exceed_total_budget() -> None:
    cfg = DLinearForecast.Config()
    cfg.pretrain_updates = cfg.updates
    with pytest.raises(ValueError, match="leave at least one"):
        cfg.make()


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("nan")])
def test_ridge_rejects_invalid_penalty(penalty: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        RidgeCorrection.Config(penalty=penalty).make()


def test_unfitted_models_and_invalid_training_counts_fail() -> None:
    cfg = exp_smoke().copy_tree().finalize()
    media = torch.zeros(4, cfg.data.seq_len, 2)
    with pytest.raises(RuntimeError, match="before predicting"):
        cfg.method.make().predict(media)
    with pytest.raises(RuntimeError, match="before predicting"):
        RidgeCorrection.Config().make().predict(media)
    with pytest.raises(ValueError, match="positive"):
        DLinearForecast.Config(updates=0).make()


@pytest.mark.compute_training
@pytest.mark.parametrize("split", ["train", "validation", "shape"])
def test_bad_training_data_cannot_produce_a_report(split: str) -> None:
    cfg = exp_smoke().copy_tree().finalize()
    data = cfg.data.make()
    train = data.windows(cfg.data.train, simulator=data.reference)
    validation = data.windows(cfg.data.validation, simulator=data.reference)
    synthetic = data.windows(cfg.data.synthetic, simulator=data.cheap)
    if split == "train":
        train.label.fill_(float("nan"))
        message = "nonfinite loss"
    elif split == "validation":
        validation.label.fill_(float("nan"))
        message = "finite validation checkpoint"
    else:
        train.media = train.media[:, :-1]
        message = "dimensions"
    with pytest.raises(ValueError, match=message):
        cfg.method.make().fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )
