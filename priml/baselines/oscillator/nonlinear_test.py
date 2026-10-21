"""Check nonlinear capacity, checkpoint selection, and history masking."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, override
from unittest import mock

from configgle import Makes, PartialConfig
from torch import Tensor

import pytest
import torch

from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.data import Oscillator, Windows
from priml.baselines.oscillator.experiments import exp014, exp029
from priml.baselines.oscillator.nonlinear import (
    HistoryAblation,
    HistoryNetwork,
    KernelCorrection,
    NeuralForecast,
    PolynomialKernel,
)
from priml.cost import cost
from priml.model.init import unit_fan_in_uniform
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.testing.golden import assert_pprint_golden
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.loss.custom_types import LossOutput
    from priml.train.custom_types import TrainStepOutput


def test_quadratic_kernel_learns_an_interaction_affine_features_cannot() -> None:
    train = _windows()
    method = KernelCorrection.Config().make()
    unused = Windows(
        media=torch.full_like(train.media, float("nan")),
        label=train.label,
    )
    method.fit(
        train,
        validation=unused,
        synthetic=unused,
        simulator=Oscillator.Config().make(),
    )
    prediction = method.predict(train.media * 1.1)
    target = train.label * 1.21
    assert prediction.shape == target.shape
    assert prediction.dtype == target.dtype
    assert (prediction - target).square().mean() < 0.01 * target.square().mean()


def test_kernel_gram_is_symmetric_positive_and_degree_changes_features() -> None:
    values = _windows().media.flatten(start_dim=1).double()
    cfg = PolynomialKernel.Config()
    nonlinear = cfg.make()(values, values)
    assert torch.equal(nonlinear, nonlinear.T)
    directions = torch.arange(len(values) * 3).reshape(len(values), 3).double() - 5
    assert ((nonlinear @ directions) * directions).sum(dim=0).min() >= 0
    cfg.degree = 1
    assert not torch.equal(nonlinear, cfg.make()(values, values))
    left, right = values[:2], values[2:5]
    torch.testing.assert_close(
        PolynomialKernel.Config(degree=3, offset=0.5).make()(left, right),
        (0.5 + left @ right.T / values.shape[1]).pow(3),
    )


@pytest.mark.parametrize("bad", ["degree", "offset", "penalty"])
def test_bad_kernel_settings_are_rejected(bad: str) -> None:
    cfg = PolynomialKernel.Config()
    if bad == "penalty":
        with pytest.raises(ValueError, match="penalty"):
            KernelCorrection.Config(penalty=0).make()
    else:
        if bad == "degree":
            cfg.degree = 0
        else:
            cfg.offset = -1
        with pytest.raises(ValueError, match="positive degree"):
            cfg.make()


def test_kernel_prediction_requires_a_fit() -> None:
    with pytest.raises(RuntimeError, match="Fit the kernel"):
        KernelCorrection.Config().make().predict(_windows().media)


def test_network_geometry_reaches_the_injected_block() -> None:
    cfg = _neural_config().step.model.copy_tree().finalize()
    assert isinstance(cfg.block, SwiGLU.Config)
    assert (cfg.block.channels_in, cfg.block.channels_out) == (3 * 2, 4 * 2)
    assert cfg.block.channels_hidden == 4
    output = cfg.make()(media=_windows().media)
    assert output.shape == _windows().label.shape
    with pytest.raises(ValueError, match="geometry"):
        cfg.make()(media=_windows().media[:, :2])
    cfg.seq_len = 0
    with pytest.raises(ValueError, match="positive"):
        cfg.make()


@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_network_costs_one_flattened_history_per_trajectory(
    composed: bool,
    dtype: torch.dtype,
) -> None:
    cfg = _neural_config().step.model.copy_tree()
    if composed:
        cfg.block = Sequential.Config(
            elements=[
                SwiGLU.Config(channels_hidden=5, channels_out=7),
                SwiGLU.Config(channels_hidden=4),
            ],
        )
    else:
        assert isinstance(cfg.block, SwiGLU.Config)
        cfg.block.bias = True
    cfg = cfg.finalize()
    counted = cost(cfg, seq_len=3, batch_size=2, dtype=dtype)
    assert counted == cost(cfg.block, seq_len=1, batch_size=2, dtype=dtype)
    assert counted.params == sum(
        parameter.numel() for parameter in cfg.make().parameters()
    )
    assert counted.params > 0
    single = cost(cfg, seq_len=3, batch_size=1, dtype=dtype)
    assert counted.params == single.params
    assert counted["primal", "flops", "matmul", dtype] == (
        2 * single["primal", "flops", "matmul", dtype]
    )
    with pytest.raises(ValueError, match="geometry"):
        cost(cfg, seq_len=5, batch_size=2, dtype=dtype)


@pytest.mark.parametrize("accumulate", [1, 3])
def test_network_uses_common_updates_and_restores_the_selected_full_state(
    accumulate: int,
) -> None:
    cfg = _neural_config()
    cfg.step.accumulate_grad_batches = accumulate
    cfg.step.gradient_clip_norm = 0.1
    method = cfg.make()
    train = _windows()
    report = method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert report["updates"] == 3
    assert 0 < report["best_update"] <= 3
    assert method.step.global_step == report["best_update"]
    assert method.step.accumulation_steps == 0
    assert "validation_scaled_mse" in report
    _, scale = train.channel_stats()
    actual = float(
        ((method.predict(train.media) - train.label) / scale).square().mean(),
    )
    assert actual == pytest.approx(report["validation_scaled_mse"], abs=1e-7)
    assert method.step.last_grad_norm is not None
    assert float(method.step.last_grad_norm) >= 0
    with pytest.raises(RuntimeError, match="fresh network"):
        method.fit(
            train,
            validation=train,
            synthetic=train,
            simulator=Oscillator.Config().make(),
        )


def test_skipped_network_attempts_report_zero_actual_optimizer_updates() -> None:
    cfg = _neural_config()
    cfg.updates = 2
    cfg.step.skip_step_on_nonfinite_grad = True
    cfg.step.loss = PartialConfig(_finite_loss_nan_gradient)
    method = cfg.make()
    before = [parameter.clone() for parameter in method.step.model.parameters()]
    train = _windows()
    report = method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert report["updates"] == 0
    assert report["best_update"] == 0
    assert report.get("attempted_updates") == 2
    assert report.get("skipped_updates") == 2
    assert report.get("best_attempt") == 1
    assert method.step.global_step == method.step.skipped_steps == 1
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(
            method.step.model.parameters(),
            before,
            strict=True,
        )
    )


@pytest.mark.parametrize("bad", ["budget", "batch", "interval"])
def test_bad_network_training_settings_are_rejected(bad: str) -> None:
    cfg = _neural_config()
    if bad == "budget":
        cfg.step.train_budget_steps = float("inf")
    elif bad == "batch":
        cfg.batch_size = 0
    else:
        cfg.validation_every = 0
    with pytest.raises(ValueError, match=r"budget|positive"):
        cfg.make()


@pytest.mark.parametrize(
    "control",
    ["dtype_autocast", "train_budget_sec", "train_budget_epochs"],
)
def test_network_rejects_training_controls_it_cannot_honor(control: str) -> None:
    cfg = _neural_config()
    if control == "dtype_autocast":
        cfg.step.dtype_autocast = torch.bfloat16
    elif control == "train_budget_sec":
        cfg.step.train_budget_sec = 1.0
    else:
        cfg.step.train_budget_epochs = 1.0
    with pytest.raises(ValueError, match=control):
        cfg.make()


def test_network_rejects_wrong_geometry_and_prediction_before_fitting() -> None:
    method = _neural_config().make()
    train = _windows()
    with pytest.raises(RuntimeError, match="Fit the network"):
        method.predict(train.media)
    invalid = Windows(media=train.media, label=train.label[:, :3])
    with pytest.raises(ValueError, match="geometry"):
        method.fit(
            invalid,
            validation=train,
            synthetic=train,
            simulator=Oscillator.Config().make(),
        )


def test_network_requires_a_finite_validation_checkpoint() -> None:
    cfg = _neural_config()
    step = _ScriptedStep.Config()
    step.update(cfg.step)
    cfg.step = step
    method = cfg.make()
    train = _windows()
    invalid = Windows(
        media=train.media,
        label=torch.full_like(train.label, float("nan")),
    )
    with pytest.raises(ValueError, match="finite validation"):
        method.fit(
            train,
            validation=invalid,
            synthetic=train,
            simulator=Oscillator.Config().make(),
        )
    with pytest.raises(RuntimeError, match="Fit the network"):
        method.predict(train.media)


def test_history_mask_is_applied_consistently_and_does_not_mutate_inputs() -> None:
    train = _windows()
    original = train.media.clone()
    cfg = HistoryAblation.Config()
    cfg.visible_steps = 1
    method = cfg.make()
    masked = method.mask(train.media)
    assert torch.count_nonzero(masked) == 0
    assert torch.equal(train.media, original)
    method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    prediction = method.predict(train.media)
    torch.testing.assert_close(
        prediction,
        torch.zeros_like(prediction),
        atol=1e-8,
        rtol=0,
    )
    with mock.patch.object(
        method.method,
        "predict",
        wraps=method.method.predict,
    ) as predict:
        method.predict(train.media)
    received = predict.call_args.args[0]
    assert isinstance(received, Tensor)
    assert torch.equal(received, masked)
    cfg.visible_steps = train.media.shape[1] + 1
    assert torch.equal(cfg.make().mask(train.media), train.media)
    cfg.visible_steps = 0
    with pytest.raises(ValueError, match="positive"):
        cfg.make()


def test_history_ablation_forwards_study_geometry_to_a_wrapped_network() -> None:
    cfg = ForecastStudy.Config()
    cfg.data.seq_len, cfg.data.pred_len = 5, 3
    cfg.method = HistoryAblation.Config(method=NeuralForecast.Config())
    wrapper = cfg.copy_tree().finalize().method
    assert isinstance(wrapper, HistoryAblation.Config)
    assert isinstance(wrapper.method, NeuralForecast.Config)
    assert (wrapper.method.seq_len, wrapper.method.pred_len) == (5, 3)
    # A wrapped method without a geometry is left untouched.
    kernel = HistoryAblation.Config(seq_len=5, pred_len=3).finalize()
    assert kernel.method == KernelCorrection.Config()


def test_network_batches_pass_through_the_step_preprocessor() -> None:
    method = _neural_config().make()
    train = _windows()
    with mock.patch.object(
        method.step,
        "preprocess_batch",
        wraps=method.step.preprocess_batch,
    ) as preprocess:
        method.fit(
            train,
            validation=train,
            synthetic=train,
            simulator=Oscillator.Config().make(),
        )
    # Three training batches, then one validation forecast after each update.
    assert preprocess.call_count == 6


def test_neural_config_is_fully_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp014", config=exp014())


def test_two_block_config_is_fully_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp029", config=exp029())


def test_bias_free_single_gate_has_only_a_quadratic_even_part() -> None:
    cfg = SwiGLU.Config(channels_in=4, channels_out=3, channels_hidden=5)
    cfg.init_weight_out = unit_fan_in_uniform
    block = cfg.make().double()
    bits = torch.arange(3, -1, -1)[None]
    signs = 1 - 2 * ((torch.arange(16)[:, None] >> bits) & 1)
    values = signs.double() * torch.tensor([0.2, 0.4, 0.7, 0.9])
    projected = block.up_proj(values)
    gate, up = projected.chunk(2, dim=-1)
    actual = block(values) + block(-values)
    expected = block.down_proj(gate * up)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    target = signs.prod(dim=-1).double()
    torch.testing.assert_close(
        block(values).T @ target,
        torch.zeros(3),
        atol=1e-12,
        rtol=0,
        check_dtype=False,
    )
    biased_cfg = cfg.copy_tree()
    biased_cfg.bias = True
    biased = biased_cfg.make().double()
    assert biased.up_proj.bias is not None
    with torch.no_grad():
        biased.up_proj.bias.fill_(0.3)
    assert (biased(values).T @ target).abs().max() > 1e-8


def test_composed_blocks_receive_gradients_and_inherit_geometry() -> None:
    cfg = _neural_config()
    cfg.step.model.block = Sequential.Config(
        elements=[
            SwiGLU.Config(
                channels_hidden=5,
                channels_out=7,
                init_weight_out=unit_fan_in_uniform,
            ),
            SwiGLU.Config(channels_hidden=4),
        ],
    )
    method = cfg.make()
    assert isinstance(method.step.model, HistoryNetwork)
    assert isinstance(method.step.model.block, Sequential)
    first = method.step.model.block[0]
    assert isinstance(first, SwiGLU)
    before = first.up_proj.weight.clone()
    train = _windows()
    report = method.fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert report["updates"] == 3
    assert not torch.equal(first.up_proj.weight, before)
    assert method.predict(train.media).shape == train.label.shape


def test_scripted_updates_restore_an_earlier_checkpoint_and_use_training_stats() -> (
    None
):
    cfg = _neural_config()
    cfg.validation_every = 2
    step = _ScriptedStep.Config()
    step.update(cfg.step)
    cfg.step = step
    method = cfg.make()
    train = _windows()
    report = method.fit(
        train,
        validation=Windows(media=train.media * 3, label=train.label * 3),
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert report["updates"] == 3
    assert report["best_update"] == method.step.global_step == 2
    assert all(
        torch.count_nonzero(value) == 0 for value in method.step.model.parameters()
    )
    assert isinstance(method.step, _ScriptedStep)
    mean, scale = train.channel_stats()
    generator = torch.Generator().manual_seed(cfg.sample_seed)
    for media, label in method.step.batches:
        indices = torch.randint(
            len(train.media),
            (cfg.batch_size,),
            generator=generator,
        )
        assert torch.equal(media, (train.media[indices] - mean) / scale)
        assert torch.equal(label, (train.label[indices] - mean) / scale)
    with pytest.raises(RuntimeError, match="fresh network"):
        method.fit(
            train,
            validation=train,
            synthetic=train,
            simulator=Oscillator.Config().make(),
        )


def test_the_final_update_is_validated_off_the_interval() -> None:
    cfg = _neural_config()
    cfg.validation_every = 2
    step = _ScriptedStep.Config()
    step.update(cfg.step)
    step.final_is_best = True
    cfg.step = step
    train = _windows()
    report = cfg.make().fit(
        train,
        validation=train,
        synthetic=train,
        simulator=Oscillator.Config().make(),
    )
    assert report["best_update"] == report["updates"] == 3


class _ScriptedStep(TrainStep):
    """Exercise checkpoint selection without an optimizer trajectory."""

    class Config(Makes["_ScriptedStep"], TrainStep.Config[HistoryNetwork.Config]):
        model: HistoryNetwork.Config = field(default_factory=HistoryNetwork.Config)
        """Small network whose weights are supplied by the test."""

        final_is_best: bool = False
        """Give the final update the zero weights instead of the large ones."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.batches: list[tuple[Tensor, Tensor]] = []
        self.final_is_best = config.final_is_best

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        media, label = preprocessed_batch["media"], preprocessed_batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        self.batches.append((media.clone(), label.clone()))
        self.timer_step.global_count += 1
        final = self.global_step == self.config.train_budget_steps
        value = 100.0 if final != self.final_is_best else 0.0
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.fill_(value)
        return self.train_loss(**preprocessed_batch)


def _windows() -> Windows:
    magnitudes = torch.tensor([[0.4, 0.3], [0.8, 0.5], [1.2, 0.7]])
    signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    factors = (magnitudes[:, None] * signs[None]).reshape(12, 2)
    media = torch.zeros(12, 3, 2)
    media[:, 0, 0], media[:, 1, 1] = factors.unbind(dim=-1)
    shape = torch.arange(1, 9).reshape(4, 2).float() / 10
    return Windows(media=media, label=factors.prod(dim=-1)[:, None, None] * shape)


def _neural_config() -> NeuralForecast.Config:
    cfg = NeuralForecast.Config()
    cfg.seq_len, cfg.pred_len = 3, 4
    cfg.updates = 3
    cfg.batch_size = 5
    cfg.validation_every = 1
    assert isinstance(cfg.step.model.block, SwiGLU.Config)
    cfg.step.model.block.channels_hidden = 4
    return cfg


def _finite_loss_nan_gradient(output: object, **kwargs: object) -> LossOutput:
    del kwargs
    assert isinstance(output, Tensor)
    return {
        "loss": torch.where(
            torch.ones_like(output, dtype=torch.bool),
            output.square(),
            output * float("nan"),
        ),
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
