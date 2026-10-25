"""Check history-only forecasting and the shared training controls."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import math

from configgle import Makeable, PartialConfig

import pytest
import torch

from priml.baselines.etth1.model import DLinear
from priml.baselines.oscillator.adaptation import (
    FixedPhysics,
    LastObservation,
    ObservedDLinear,
    ObservedForecaster,
    ObservedHistory,
    ObservedNeuralForecast,
    ObservedRidge,
    RecentPhysics,
)
from priml.baselines.oscillator.regimes import (
    ObservedChange,
    ObservedWindows,
    PositionObservation,
    RegimeData,
    RegimeDynamics,
)
from priml.cost import cost
from priml.model.sequential import Sequential
from priml.model.special import Identity
from priml.model.swiglu import SwiGLU
from priml.testing.cost import assert_cost_matches_torch
from priml.train.ema import EMA


if TYPE_CHECKING:
    from collections.abc import Callable


def test_dlinear_adapter_preserves_original_initialization_and_forward() -> None:
    cfg = ObservedDLinear.Config()
    cfg.seq_len, cfg.pred_len, cfg.channels = 7, 3, 2
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(17)
        original = cfg.model.make()
        torch.default_generator.manual_seed(17)
        adapted = cfg.make()
    media = torch.arange(5 * 7 * 2).reshape(5, 7, 2).float() / 13
    assert isinstance(original, DLinear)
    assert torch.equal(
        original(media),
        adapted(media=media, label=torch.zeros(5, 3, 2)),
    )
    assert all(
        torch.equal(v, adapted.model.state_dict()[k])
        for k, v in original.state_dict().items()
    )


def test_nonlinear_block_receives_all_observed_geometry() -> None:
    cfg = ObservedHistory.Config(seq_len=7, pred_len=3, channels=2)
    model = cfg.make()
    assert model(torch.ones(5, 7, 2), label=None).shape == (5, 3, 2)
    with pytest.raises(ValueError, match="positive"):
        ObservedHistory.Config(channels=0).make()


@pytest.mark.parametrize("channels", [1, 4])
def test_dlinear_cost_preserves_child_work_and_inactive_decoder(channels: int) -> None:
    cfg = ObservedDLinear.Config()
    cfg.seq_len, cfg.pred_len, cfg.channels = 7, 5, channels
    # PositionObservation forces a single visible coordinate in the masked task.
    counted = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(3, 7, channels, requires_grad=True),
        seq_len=7,
        batch_size=3,
        dtype=torch.float32,
    )
    assert counted == cost(
        cfg.model.finalized(),
        seq_len=7,
        batch_size=3,
        dtype=torch.float32,
    )
    assert counted.params > counted.params_active > 0


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("composed", [False, True])
def test_history_cost_counts_one_flattened_invocation(
    channels: int,
    composed: bool,
) -> None:
    cfg = ObservedHistory.Config(seq_len=7, pred_len=5, channels=channels)
    if composed:
        cfg.block = Sequential.Config(
            elements=[
                SwiGLU.Config(channels_hidden=11, channels_out=13, bias=True),
                SwiGLU.Config(channels_hidden=17, bias=True),
            ],
        )
    else:
        assert isinstance(cfg.block, SwiGLU.Config)
        cfg.block.channels_hidden = 11
    # PositionObservation forces one channel; the full physical state has two.
    counted = assert_cost_matches_torch(
        cfg,
        build_input=lambda: torch.randn(3, 7, channels, requires_grad=True),
        seq_len=7,
        batch_size=3,
        dtype=torch.float32,
    )
    finalized = cfg.finalized()
    assert counted == cost(
        finalized.block,
        seq_len=1,
        batch_size=3,
        dtype=torch.float32,
    )
    smaller = cost(finalized, seq_len=7, batch_size=2, dtype=torch.float32)
    assert counted.params == smaller.params > 0
    assert (
        2 * counted["primal", "flops", "matmul"].sum()
        == 3 * smaller["primal", "flops", "matmul"].sum()
    )


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_observed_costs_forward_dtype_and_reject_inconsistent_history(
    dtype: torch.dtype | None,
) -> None:
    dlinear = ObservedDLinear.Config()
    dlinear.seq_len, dlinear.pred_len, dlinear.channels = 7, 5, 2
    dlinear = dlinear.finalized()
    history = ObservedHistory.Config(seq_len=7, pred_len=5, channels=2).finalized()
    assert cost(dlinear, seq_len=7, batch_size=3, dtype=dtype) == cost(
        dlinear.model,
        seq_len=7,
        batch_size=3,
        dtype=dtype,
    )
    assert cost(history, seq_len=7, batch_size=3, dtype=dtype) == cost(
        history.block,
        seq_len=1,
        batch_size=3,
        dtype=dtype,
    )
    for cfg in (dlinear, history):
        with pytest.raises(ValueError, match="history"):
            cost(cfg, seq_len=11, batch_size=3, dtype=dtype)


def test_pooled_ridge_learns_affine_history_mapping_without_validation_targets() -> (
    None
):
    generator = torch.Generator().manual_seed(31)
    media = torch.randn(37, 5, 2, generator=generator, dtype=torch.float64)
    weights = torch.randn(10, 6, generator=generator, dtype=torch.float64)
    label = (media.flatten(1) @ weights + 0.7).reshape(37, 3, 2)
    windows = ObservedWindows(media=media, label=label)
    cfg = ObservedRidge.Config(penalty=1e-9)
    first, second = cfg.make(), cfg.make()
    dynamics = RegimeDynamics.Config().make()
    first.fit(windows, validation=windows, dynamics=dynamics)
    poisoned = ObservedWindows(media=media + 100, label=label - 100)
    second.fit(windows, validation=poisoned, dynamics=dynamics)
    unseen = torch.randn(7, 5, 2, generator=generator, dtype=torch.float64)
    expected = (unseen.flatten(1) @ weights + 0.7).reshape(7, 3, 2)
    torch.testing.assert_close(first.predict(unseen), expected, atol=1e-8, rtol=1e-8)
    assert torch.equal(first.predict(unseen), second.predict(unseen))


@pytest.mark.parametrize("position_only", [False, True])
def test_recent_physics_identifies_a_solvable_observed_change(
    position_only: bool,
) -> None:
    cfg = RegimeData.Config(seq_len=43, pred_len=13)
    cfg.schedule = ObservedChange.Config(recent_steps=(19, 23))
    if position_only:
        cfg.observation = PositionObservation.Config()
    data = cfg.make()
    train = data.sample(count=3, seed=37).windows
    evaluation = data.sample(count=5, seed=41)
    method = RecentPhysics.Config(visible_steps=17).make()
    method.fit(train, validation=train, dynamics=data.dynamics)
    prediction = method.predict(evaluation.windows.media)
    assert prediction.shape == evaluation.windows.label.shape
    assert (
        float((prediction[..., 0] - evaluation.windows.label[..., 0]).square().mean())
        < 1e-5
    )
    if not position_only:
        assert torch.equal(
            method.observed_state(evaluation.windows.media),
            evaluation.windows.media[:, -1],
        )


def test_fixed_physics_uses_no_parameter_identification_solve() -> None:
    data = RegimeData.Config(seq_len=11, pred_len=5).make()
    windows = data.sample(count=3, seed=43).windows
    method = FixedPhysics.Config().make()
    method.fit(windows, validation=windows, dynamics=data.dynamics)
    with mock.patch.object(
        torch.linalg,
        "solve",
        side_effect=AssertionError("hidden fit"),
    ):
        state, coefficients = method.identify(windows.media)
        prediction = method.predict(windows.media)
    assert torch.equal(state, windows.media[:, -1])
    expected = data.dynamics.forecast(state, coefficients=coefficients, steps=5)
    assert torch.equal(prediction, expected)


def test_persistence_uses_the_newest_observation_without_storage_aliasing() -> None:
    data = RegimeData.Config(seq_len=11, pred_len=5).make()
    windows = data.sample(count=3, seed=47).windows
    method = LastObservation.Config().make()
    method.fit(windows, validation=windows, dynamics=data.dynamics)
    prediction = method.predict(windows.media)
    assert torch.equal(prediction, windows.media[:, -1:].expand(-1, 5, -1))
    prediction.zero_()
    assert windows.media.any()


@pytest.mark.parametrize(
    "factory",
    [
        ObservedRidge.Config,
        RecentPhysics.Config,
        FixedPhysics.Config,
        LastObservation.Config,
        ObservedNeuralForecast.Config,
    ],
)
def test_prediction_requires_a_fit(
    factory: Callable[[], Makeable[ObservedForecaster]],
) -> None:
    with pytest.raises(RuntimeError, match="Fit"):
        factory().make().predict(torch.zeros(3, 7, 2))


def test_training_honors_accumulation_clipping_ema_and_best_checkpoint() -> None:
    data = RegimeData.Config(seq_len=7, pred_len=3).make()
    windows = data.sample(count=5, seed=53).windows
    cfg = ObservedNeuralForecast.Config()
    cfg.seq_len, cfg.pred_len, cfg.channels = 7, 3, 2
    cfg.step.train_budget_steps = 3
    cfg.validation_every = 2
    cfg.batch_size = 4
    cfg.step.accumulate_grad_batches = 2
    cfg.step.gradient_clip_norm = 1e-5
    cfg.step.ema = EMA.Config(decay=0.5)
    method = cfg.make()
    with (
        mock.patch.object(
            method.step,
            "train_step",
            wraps=method.step.train_step,
        ) as train,
        mock.patch.object(
            torch.nn.utils,
            "clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip,
    ):
        fit = method.fit(windows, validation=windows, dynamics=data.dynamics)
    assert train.call_count == 6
    assert clip.call_count == 3
    assert fit["attempted_updates"] == 3
    assert fit["skipped_updates"] == 0
    assert method.step.global_step == fit["best_update"]
    assert method.step.ema.global_step == fit["best_update"]
    assert method.step.accumulation_steps == 0
    assert torch.isfinite(method.predict(windows.media)).all()
    with pytest.raises(RuntimeError, match="fresh"):
        method.fit(windows, validation=windows, dynamics=data.dynamics)


def test_training_uses_configured_compiler() -> None:
    data = RegimeData.Config(seq_len=7, pred_len=3).make()
    windows = data.sample(count=3, seed=59).windows
    cfg = ObservedNeuralForecast.Config()
    cfg.seq_len, cfg.pred_len, cfg.channels = 7, 3, 2
    cfg.step.train_budget_steps = 2
    cfg.validation_every = 3

    def passthrough(model: torch.nn.Module) -> torch.nn.Module:
        return model

    with mock.patch.object(
        torch,
        "compile",
        side_effect=passthrough,
    ) as compile_model:
        cfg.step.compile = PartialConfig(torch.compile)
        method = cfg.make()
        method.fit(windows, validation=windows, dynamics=data.dynamics)
    assert compile_model.call_count == 1


def test_nonfinite_gradient_control_skips_updates_without_corrupting_weights() -> None:
    data = RegimeData.Config(seq_len=7, pred_len=3).make()
    windows = data.sample(count=3, seed=79).windows
    cfg = ObservedNeuralForecast.Config()
    cfg.seq_len, cfg.pred_len, cfg.channels = 7, 3, 2
    cfg.step.train_budget_steps = 2
    cfg.step.skip_step_on_nonfinite_grad = True
    method = cfg.make()
    before = {k: v.clone() for k, v in method.step.model.state_dict().items()}
    preprocess = method.step.preprocess_batch

    def corrupt_training_labels(batch: dict[str, object]) -> dict[str, object]:
        processed = preprocess(batch)
        label = processed.get("label")
        if isinstance(label, torch.Tensor):
            processed["label"] = torch.full_like(label, math.nan)
        return processed

    with mock.patch.object(
        method.step,
        "preprocess_batch",
        side_effect=corrupt_training_labels,
    ):
        fit = method.fit(windows, validation=windows, dynamics=data.dynamics)
    assert fit["attempted_updates"] == fit["skipped_updates"] == 2
    assert all(
        torch.equal(value, method.step.model.state_dict()[name])
        for name, value in before.items()
    )


def test_no_finite_validation_checkpoint_is_an_explicit_failure() -> None:
    data = RegimeData.Config(seq_len=7, pred_len=3).make()
    windows = data.sample(count=3, seed=61).windows
    cfg = ObservedNeuralForecast.Config()
    cfg.seq_len, cfg.pred_len, cfg.channels = 7, 3, 2
    cfg.step.train_budget_steps = 2
    method = cfg.make()
    with (
        mock.patch.object(
            method,
            "predict",
            return_value=torch.full_like(windows.label, math.nan),
        ),
        pytest.raises(ValueError, match="No finite"),
    ):
        method.fit(windows, validation=windows, dynamics=data.dynamics)
    with pytest.raises(RuntimeError, match="Fit"):
        method.predict(windows.media)


def test_training_rejects_unmatched_geometry() -> None:
    data = RegimeData.Config(seq_len=11, pred_len=5).make()
    windows = data.sample(count=3, seed=67).windows
    cfg = ObservedNeuralForecast.Config()
    cfg.step.train_budget_steps = 2
    with pytest.raises(ValueError, match="geometry"):
        cfg.make().fit(windows, validation=windows, dynamics=data.dynamics)


@pytest.mark.parametrize("budget", [0.0, 1.5, math.inf])
def test_neural_budget_must_be_finite_and_whole(budget: float) -> None:
    cfg = ObservedNeuralForecast.Config()
    cfg.step.train_budget_steps = budget
    with pytest.raises(ValueError, match="whole count"):
        cfg.make()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("validation_every", 0),
        ("step.train_budget_sec", 5.0),
        ("step.train_budget_epochs", 2.0),
        ("step.dtype_autocast", torch.bfloat16),
    ],
)
def test_neural_rejects_unsupported_controls(field: str, value: object) -> None:
    cfg = ObservedNeuralForecast.Config()
    if field.startswith("step."):
        setattr(cfg.step, field.split(".")[1], value)
    else:
        setattr(cfg, field, value)
    with pytest.raises(ValueError, match=r"positive|counts|updates|autocast"):
        cfg.make()


def test_neural_requires_model_owned_geometry() -> None:
    cfg = ObservedNeuralForecast.Config()
    cfg.step.model = Identity.Config()
    with pytest.raises(TypeError, match="geometry"):
        cfg.make()


@pytest.mark.parametrize("penalty", [0.0, -1.0, math.inf, math.nan])
def test_ridge_penalty_is_positive_and_finite(penalty: float) -> None:
    with pytest.raises(ValueError, match="penalty"):
        ObservedRidge.Config(penalty=penalty).make()
    with pytest.raises(ValueError, match="penalty"):
        RecentPhysics.Config(penalty=penalty).make()


def test_physics_rejects_invalid_history_and_parameter_bounds() -> None:
    data = RegimeData.Config(seq_len=11, pred_len=5).make()
    windows = data.sample(count=3, seed=71).windows
    method = RecentPhysics.Config().make()
    with pytest.raises(RuntimeError, match="Fit"):
        method.observed_state(windows.media)
    with pytest.raises(RuntimeError, match="Fit"):
        method.identify(windows.media)
    method.fit(windows, validation=windows, dynamics=data.dynamics)
    for media in (windows.media[:, :4], windows.media[..., 0], torch.zeros(3, 7, 4)):
        with pytest.raises(ValueError, match="observations"):
            method.identify(media)
        with pytest.raises(ValueError, match="observations"):
            method.observed_state(media)
    with pytest.raises(ValueError, match="seven"):
        RecentPhysics.Config(visible_steps=6).make()
    with pytest.raises(ValueError, match="bounds"):
        RecentPhysics.Config(lower=(0.1, -0.2, 0.0)).make()
    with pytest.raises(ValueError, match="physical"):
        FixedPhysics.Config(coefficients=(0.0, 0.1, 0.0)).make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
