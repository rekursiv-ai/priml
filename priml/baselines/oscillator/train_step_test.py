"""Check training budgets, normalization, and simulator correction."""

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch

from priml.baselines.oscillator.data import Windows
from priml.baselines.oscillator.experiments import exp_smoke
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection
from priml.train.custom_types import TrainStepOutput


if TYPE_CHECKING:
    from collections.abc import Callable


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
    data = cfg.data.make()
    train = data.windows(cfg.data.train, simulator=data.reference)
    validation = data.windows(cfg.data.validation, simulator=data.reference)
    synthetic = data.windows(cfg.data.synthetic, simulator=data.cheap)
    method = cfg.method.make()
    batches: list[tuple[Tensor, Tensor]] = []
    validation_steps: list[int] = []
    predict = method.predict
    train_loss: Callable[..., TrainStepOutput] = method.step.train_loss

    def scripted_update(**batch: object) -> TrainStepOutput:
        media, label = batch["media"], batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        batches.append((media.clone(), label.clone()))
        method.step.timer_step.global_count += 1
        # Make the last weights worse so keeping them would fail this test.
        value = 100.0 if method.step.global_step == method.config.updates else 0.0
        with torch.no_grad():
            for parameter in method.step.model.parameters():
                parameter.fill_(value)
        return train_loss(**batch)

    def record_validation(media: Tensor) -> Tensor:
        validation_steps.append(method.step.global_step)
        return predict(media)

    monkeypatch.setattr(method.step, "train_step", scripted_update)
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
    assert all(
        torch.count_nonzero(value) == 0 for value in method.step.model.parameters()
    )
    values = torch.cat((train.media, train.label), dim=1)
    mean = values.mean(dim=(0, 1), keepdim=True)
    scale = values.std(dim=(0, 1), correction=0, keepdim=True)
    assert torch.equal(predict(validation.media), mean.expand_as(validation.label))
    assert report["validation_scaled_mse"] == pytest.approx(
        float(((mean - validation.label) / scale).square().mean()),
    )
    for offset, source in ((0, train), (1, synthetic)):
        generator = torch.Generator().manual_seed(cfg.method.sample_seed + offset)
        selected = (
            batches[pretrain_updates:] if offset == 0 else batches[:pretrain_updates]
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


@pytest.mark.parametrize(
    "optimize",
    [False, pytest.param(True, marks=pytest.mark.compute_training)],
)
@pytest.mark.parametrize("split", ["train", "validation", "shape"])
def test_bad_training_data_cannot_produce_a_report(
    split: str,
    optimize: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    method = cfg.method.make()
    assert isinstance(method, DLinearForecast)
    if not optimize:
        # Input checks also run without an optimizer update.
        monkeypatch.setattr(method.step, "train_step", method.step.train_loss)
    with pytest.raises(ValueError, match=message):
        method.fit(
            train,
            validation=validation,
            synthetic=synthetic,
            simulator=data.cheap,
        )
