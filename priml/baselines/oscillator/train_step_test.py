"""Check training budgets, normalization, and simulator correction."""

from typing import TYPE_CHECKING
from unittest import mock

import dataclasses
import math

from torch import Tensor

import pytest
import torch

from priml.baselines.oscillator.data import ForecastData, Windows
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.train_step import (
    DLinearForecast,
    FitReport,
    RidgeCorrection,
)
from priml.train.custom_types import TrainStepOutput


if TYPE_CHECKING:
    from collections.abc import Callable


class _ScriptedUpdates:
    """Replace optimizer updates with fixed weights, recording each batch."""

    def __init__(self, method: DLinearForecast) -> None:
        self.method = method
        self.train_loss: Callable[..., TrainStepOutput] = method.step.train_loss
        self.batches: list[tuple[Tensor, Tensor]] = []

    def __call__(self, **batch: object) -> TrainStepOutput:
        media, label = batch["media"], batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        self.batches.append((media.clone(), label.clone()))
        step = self.method.step
        step.timer_step.global_count += 1
        # Make the last weights worse so keeping them would fail this test.
        value = 100.0 if step.global_step == self.method.config.updates else 0.0
        with torch.no_grad():
            for parameter in step.model.parameters():
                parameter.fill_(value)
        return self.train_loss(**batch)


@pytest.mark.parametrize("pretrain_updates", [0, 1])
def test_fit_restores_best_weights_and_checks_the_final_update(
    monkeypatch: pytest.MonkeyPatch,
    pretrain_updates: int,
) -> None:
    cfg = exp_smoke()
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.updates = pretrain_updates + 3
    cfg.method.pretrain_updates = pretrain_updates
    cfg.method.validation_every = 2
    cfg = cfg.finalize()
    assert isinstance(cfg.method, DLinearForecast.Config)
    train, validation, synthetic, data = _splits(cfg.data)
    method = cfg.method.make()
    scripted = _ScriptedUpdates(method)
    validation_steps: list[int] = []
    predict = method.predict

    def record_validation(media: Tensor) -> Tensor:
        validation_steps.append(method.step.global_step)
        return predict(media)

    monkeypatch.setattr(method.step, "train_step", scripted)
    monkeypatch.setattr(method, "predict", record_validation)
    report = method.fit(
        train,
        validation=validation,
        synthetic=synthetic,
        simulator=data.cheap,
    )
    assert validation_steps == [pretrain_updates + 2, cfg.method.updates]
    assert report["updates"] == cfg.method.updates
    assert report["best_update"] == pretrain_updates + 2
    assert report["pretrain_updates"] == pretrain_updates
    assert "validation_scaled_mse" in report
    assert all(
        torch.count_nonzero(value) == 0 for value in method.step.model.parameters()
    )
    mean, scale = train.channel_stats()
    assert torch.equal(predict(validation.media), mean.expand_as(validation.label))
    assert report["validation_scaled_mse"] == pytest.approx(
        float(((mean - validation.label) / scale).square().mean()),
    )
    for offset, source in ((0, train), (1, synthetic)):
        generator = torch.Generator().manual_seed(cfg.method.sample_seed + offset)
        selected = (
            scripted.batches[pretrain_updates:]
            if offset == 0
            else scripted.batches[:pretrain_updates]
        )
        for media, label in selected:
            indices = torch.randint(
                len(source.media),
                (cfg.method.batch_size,),
                generator=generator,
            )
            assert torch.equal(media, (source.media[indices] - mean) / scale)
            assert torch.equal(label, (source.label[indices] - mean) / scale)
    with pytest.raises(RuntimeError, match="fresh forecaster"):
        method.fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )


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
    train, validation, synthetic, data = _splits(cfg.data)
    method = cfg.method.make()
    report = method.fit(
        train,
        validation=validation,
        synthetic=synthetic,
        simulator=data.cheap,
    )
    assert report["updates"] == cfg.method.updates
    assert report["best_update"] == cfg.method.updates
    mean, scale = train.channel_stats()
    assert method.mean is not None
    assert method.scale is not None
    assert torch.equal(method.mean, mean)
    assert torch.equal(method.scale, scale)
    before = {
        name: value.clone() for name, value in method.step.model.state_dict().items()
    }
    assert torch.isfinite(method.predict(validation.media)).all()
    for name, value in method.step.model.state_dict().items():
        assert torch.equal(before[name], value)


def test_fit_reports_share_one_schema() -> None:
    assert FitReport.__required_keys__ == {
        "updates",
        "best_update",
        "pretrain_updates",
    }
    assert FitReport.__optional_keys__ == {"validation_scaled_mse"}


def test_building_a_forecaster_does_not_reseed_global_rng() -> None:
    cfg = exp_smoke().copy_tree().finalize()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert cfg.method.step.seed is None
    torch.manual_seed(1)
    first = cfg.method.make().step.model.state_dict()
    torch.manual_seed(2)
    second = cfg.method.make().step.model.state_dict()
    assert any(not torch.equal(first[key], second[key]) for key in first)


def test_ridge_correction_is_zero_when_simulator_matches() -> None:
    cfg = exp_smoke()
    cfg.data.reference = cfg.data.cheap.copy_tree()
    train, validation, synthetic, data = _splits(cfg.data)
    method = RidgeCorrection.Config().make()
    report = method.fit(
        train,
        validation=validation,
        synthetic=synthetic,
        simulator=data.cheap,
    )
    assert report == {"updates": 0, "best_update": 0, "pretrain_updates": 0}
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


def test_geometry_lives_only_on_the_model() -> None:
    names = {item.name for item in dataclasses.fields(DLinearForecast.Config)}
    assert names.isdisjoint({"seq_len", "pred_len"})
    cfg = DLinearForecast.Config()
    cfg.seq_len = 5
    cfg.pred_len = 3
    assert (cfg.step.model.seq_len, cfg.step.model.pred_len) == (5, 3)
    assert (cfg.seq_len, cfg.pred_len) == (5, 3)


def test_update_budget_lives_only_on_the_step() -> None:
    names = {item.name for item in dataclasses.fields(DLinearForecast.Config)}
    assert "updates" not in names
    cfg = DLinearForecast.Config()
    assert cfg.step.train_budget_steps == 300
    assert cfg.updates == 300
    cfg.updates = 7
    assert cfg.step.train_budget_steps == 7
    cfg.step.train_budget_steps = 9
    assert cfg.updates == 9


def test_finalize_keeps_explicit_step_settings() -> None:
    cfg = DLinearForecast.Config()
    assert cfg.step.model.channels == 2
    cfg.step.model.channels = 3
    assert cfg.copy_tree().finalize().step.model.channels == 3


@pytest.mark.parametrize("budget", [math.inf, 2.9])
def test_update_budget_must_be_a_finite_whole_count(budget: float) -> None:
    cfg = DLinearForecast.Config()
    cfg.step.train_budget_steps = budget
    with pytest.raises(ValueError, match="finite whole update count"):
        cfg.make()


def test_training_batches_pass_through_the_step_preprocessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp_smoke().copy_tree().finalize()
    train, validation, synthetic, data = _splits(cfg.data)
    method = cfg.method.make()
    assert isinstance(method, DLinearForecast)
    monkeypatch.setattr(method.step, "train_step", method.step.train_loss)
    with mock.patch.object(
        method.step,
        "preprocess_batch",
        wraps=method.step.preprocess_batch,
    ) as preprocess:
        method.fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )
    assert preprocess.call_count == method.config.updates


@pytest.mark.parametrize("pretrain_updates", [-1, 300])
def test_pretraining_must_leave_a_reference_update(pretrain_updates: int) -> None:
    cfg = DLinearForecast.Config()
    cfg.pretrain_updates = pretrain_updates
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
    zero = DLinearForecast.Config()
    zero.updates = 0
    with pytest.raises(ValueError, match="positive"):
        zero.make()


@pytest.mark.parametrize("split", ["validation", "shape"])
def test_bad_development_data_cannot_produce_a_report(
    split: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = exp_smoke().copy_tree().finalize()
    train, validation, synthetic, data = _splits(cfg.data)
    if split == "validation":
        validation.label.fill_(float("nan"))
        message = "finite validation checkpoint"
    else:
        train = Windows(media=train.media[:, :-1], label=train.label)
        message = "dimensions"
    method = cfg.method.make()
    assert isinstance(method, DLinearForecast)
    # Input checks also run without an optimizer update.
    monkeypatch.setattr(method.step, "train_step", method.step.train_loss)
    with pytest.raises(ValueError, match=message):
        method.fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )
    with pytest.raises(RuntimeError, match="before predicting"):
        method.predict(validation.media)


@pytest.mark.compute_training
def test_nonfinite_training_data_leaves_no_finite_checkpoint() -> None:
    cfg = exp_smoke().copy_tree().finalize()
    train, validation, synthetic, data = _splits(cfg.data)
    train.label.fill_(float("nan"))
    method = cfg.method.make()
    with pytest.raises(ValueError, match="finite validation checkpoint"):
        method.fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )


def _splits(
    cfg: ForecastData.Config,
) -> tuple[Windows, Windows, Windows, ForecastData]:
    data = cfg.make()
    return (
        data.windows(cfg.train, simulator=data.reference),
        data.windows(cfg.validation, simulator=data.reference),
        data.windows(cfg.synthetic, simulator=data.cheap),
        data,
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
