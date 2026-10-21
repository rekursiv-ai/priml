"""Three-epoch step golden, update controls, and off-target loss arithmetic."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, override

import math

from configgle import Fig, InlineConfig, Makes2, PartialConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import pytest
import torch

from priml.baselines.motor_imagery.model import (
    AttentionEEGNet,
    MultiScaleEEGNet,
    ShallowConvNet,
)
from priml.baselines.motor_imagery.testing import (
    compare_record,
    model_record,
    optimizer_record,
    random_record,
    source_step,
)
from priml.baselines.motor_imagery.train_step import (
    MotorImageryStep,
    cross_entropy_off_target,
)
from priml.cost import cost
from priml.loss.simple_loss import SimpleLoss, cross_entropy
from priml.math.schedules import linear
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr import learning_rate
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.optimizers.newton import Newton
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import PinnedRandom, assert_tensor_golden_part
from priml.timer import CheckpointableStepTimer
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.loss.custom_types import LossOutput


_CWD: Final = Path(__file__).resolve().parent


class RecordingCompiler:
    """Observe the compiler boundary without compiling CPU fixture kernels."""

    class Config(Fig["RecordingCompiler"]):
        """Construct an isolated compiler recorder."""

    def __init__(self, config: Config) -> None:
        del config
        self.calls = 0

    def __call__(self, model: Callable[..., object]) -> Callable[..., object]:
        self.calls += 1
        return model


class RecordingStep(MotorImageryStep):
    """Retain the final clipped gradients the shared update clears."""

    class Config(Makes2["RecordingStep"], MotorImageryStep.Config):
        @override
        def make(self) -> RecordingStep:
            return RecordingStep(self.finalized())

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.updates = 0
        self.gradients: list[Tensor] = []

    @override
    def optimizer_step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> bool:
        self.updates += 1
        self.gradients = [
            parameter.grad.detach().clone()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        return super().optimizer_step(closure)


def linear_step_config() -> MotorImageryStep.Config:
    """Place an inexpensive genuine model on the CPU for control regressions."""
    return source_step(
        InlineConfig(torch.nn.Linear, in_features=3, out_features=2),
        epochs=3,
    )


def recording_linear_step() -> RecordingStep:
    """Build the linear control with its gradients recorded."""
    cfg = RecordingStep.Config()
    cfg.update(linear_step_config())
    return cfg.make()


def finish_epoch(step: MotorImageryStep) -> None:
    """Close a pass in the order the loop does: tick the epoch timer, then notify."""
    step.timer_epoch.global_count += 1
    step.on_epoch_end()


def test_step_honors_compiler_and_forward_evaluation_timers() -> None:
    cfg = linear_step_config()
    cfg.compile = RecordingCompiler.Config()
    step = cfg.make()
    step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert step.timer_forward.global_count == 1
    assert isinstance(step._compile_fn, RecordingCompiler)
    assert step._compile_fn.calls == 1
    step.eval_loss(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert step.timer_eval.global_count == 1


def test_step_honors_configured_clipping() -> None:
    cfg = RecordingStep.Config()
    cfg.update(linear_step_config())
    cfg.gradient_clip_norm = 1e-20
    step = cfg.make()
    step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert all(parameter.grad is None for parameter in step.model.parameters())
    assert (
        torch.linalg.vector_norm(torch.cat([g.flatten() for g in step.gradients]))
        < 1e-19
    )


def test_mean_loss_backward_uses_shared_update_and_clears_gradients() -> None:
    step = recording_linear_step()
    step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert step.updates == 1
    assert all(parameter.grad is None for parameter in step.model.parameters())
    assert len(step.gradients) == len(list(step.model.parameters()))


@pytest.mark.parametrize("horizon", [3, 16, 18])
def test_epoch_schedule_slot_indexes_each_rate_by_completed_passes(
    horizon: int,
) -> None:
    cfg = linear_step_config()
    cfg.optimizer = PartialConfig(torch.optim.SGD, lr=1e-3)
    schedule = cfg.lr_schedule = RecursiveCosine.Config(T_max=horizon, eta_min=1e-5)
    rates = schedule.make()
    step = cfg.make()
    for epoch in range(2 * horizon + 3):
        # Two updates per pass, so the update count never equals the pass count.
        for _ in range(2):
            step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
            assert learning_rate(step.optimizer) == rates.learning_rate(
                1e-3,
                step=epoch,
            )
        finish_epoch(step)


def test_schedule_clock_is_the_bound_epoch_timer() -> None:
    cfg = linear_step_config()
    cfg.optimizer = PartialConfig(torch.optim.SGD, lr=1e-3)
    cfg.lr_schedule = RecursiveCosine.Config(T_max=4, eta_min=1e-5)
    step = cfg.make()
    timer = CheckpointableStepTimer()
    step.bind_epoch_timer(timer)
    batch = {"media": torch.ones(2, 3), "label": torch.tensor([0, 1])}
    step.train_step(**batch)
    initial = learning_rate(step.optimizer)
    timer.global_count = 2
    step.train_step(**batch)
    assert learning_rate(step.optimizer) == pytest.approx(5.05e-4, abs=1e-12)
    assert initial == 1e-3


def test_shared_update_does_not_clip_when_disabled_and_the_norm_overflows() -> None:
    cfg = linear_step_config()
    cfg.gradient_clip_norm = float("inf")
    cfg.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    step = cfg.make()
    before = [parameter.detach().clone() for parameter in step.model.parameters()]
    for parameter in step.model.parameters():
        parameter.grad = torch.full_like(parameter, 1e20)

    TrainStep.step(step)

    for old, parameter in zip(before, step.model.parameters(), strict=True):
        assert torch.equal(parameter, old.add(torch.full_like(old, 1e20), alpha=-0.1))


def test_step_honors_autocast_in_training_and_evaluation() -> None:
    cfg = linear_step_config()
    cfg.dtype_autocast = torch.bfloat16
    step = cfg.make()
    result = step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert result["model"].dtype == torch.bfloat16
    evaluated = step.eval_loss(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert evaluated["model"].dtype == torch.bfloat16


def test_step_rejects_unsupported_gradient_accumulation() -> None:
    cfg = MotorImageryStep.Config()
    cfg.accumulate_grad_batches = 2
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        cfg.make()


def test_step_honors_an_injected_schedule_slot() -> None:
    cfg = linear_step_config()
    cfg.lr_schedule = PartialConfig(linear)
    cfg.train_budget_steps = 2
    step = cfg.make()
    for _ in range(2):
        step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    assert learning_rate(step.optimizer) == 5e-4


def test_step_uses_the_standard_priml_loss_slot() -> None:
    cfg = linear_step_config()
    objective = cfg.loss = SimpleLoss.Config()
    objective.loss_fn = cross_entropy
    step = cfg.make()
    labels = torch.tensor([0, 1])
    actual = step.eval_loss(media=torch.ones(2, 3), label=labels)
    assert torch.equal(
        actual["loss"],
        torch.nn.functional.cross_entropy(actual["model"], labels, reduction="none"),
    )


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_injected_loss_receives_the_complete_batch_and_precision_policy(
    training: bool,
    autocast: bool,
) -> None:
    cfg = linear_step_config()
    if autocast:
        cfg.dtype_autocast = torch.bfloat16
    objective = cfg.loss = SimpleLoss.Config()
    objective.loss_fn = cross_entropy
    objective.target_key = "trial"
    step = cfg.make()
    media = torch.arange(12).reshape(4, 3).double()
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    targets = 1 - labels.long()
    if training:
        actual = step.train_step(
            media=media,
            label=labels,
            trial=targets,
            valid_count=4,
        )
    else:
        actual = step.eval_loss(media=media, label=labels, trial=targets, valid_count=4)
    assert actual["loss"].dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = cross_entropy(actual["model"], targets, reduction="none")
    assert torch.equal(actual["loss"], expected)


@pytest.mark.parametrize("training", [False, True])
def test_loss_uses_autocast_even_with_the_default_target(training: bool) -> None:
    cfg = linear_step_config()
    cfg.dtype_autocast = torch.bfloat16
    objective = cfg.loss = SimpleLoss.Config()
    objective.loss_fn = cross_entropy
    step = cfg.make()
    batch = {
        "media": torch.arange(12).reshape(4, 3).float(),
        "label": torch.tensor([0, 1, 0, 1]),
    }
    actual = step.train_step(**batch) if training else step.eval_loss(**batch)
    assert actual["loss"].dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = cross_entropy(actual["model"], batch["label"], reduction="none")
    assert torch.equal(actual["loss"], expected)


def test_skip_nonfinite_gradient_preserves_weights_and_advances_step() -> None:
    cfg = linear_step_config()
    cfg.skip_step_on_nonfinite_grad = True
    step = cfg.make()
    before = model_record(step.model, prefix="weights")
    step.train_step(media=torch.full((2, 3), float("nan")), label=torch.tensor([0, 1]))
    compare_record(before, model_record(step.model, prefix="weights"))
    assert step.skipped_steps == step.global_step == 1
    assert all(param.grad is None for param in step.model.parameters())


@pytest.mark.parametrize(
    "field",
    ["train_budget_steps", "train_budget_sec", "train_budget_epochs"],
)
def test_step_honors_the_shared_schedule_budget_horizons(
    field: str,
) -> None:
    cfg = linear_step_config()
    setattr(cfg, field, 10)
    step = cfg.make()
    if field == "train_budget_steps":
        step.timer_step.global_count = 5
    elif field == "train_budget_sec":
        step.timer_step.global_sec = 5
    else:
        step.timer_epoch.global_count = 5
    assert step.progress_complete == 0.5


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_three_training_epochs_match_golden(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    cfg = model_type.Config(3, 2)
    cfg.num_samples = 100
    cfg.channels_hidden = 2
    if isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
        cfg.channels_projection = 2
    with (
        host_agnostic_numerics(),
        PinnedRandom() as pins,
    ):
        step_cfg = RecordingStep.Config()
        step_cfg.update(source_step(cfg, epochs=3))
        step = step_cfg.make()
        model = step.model
        # Every network's Conv2d requires a singleton input plane; targets are binary.
        media = torch.arange(4 * 3 * 100).reshape(4, 1, 3, 100).float() / 100
        label = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(
            TensorDataset(media, label),
            batch_size=4,
            shuffle=True,
            drop_last=True,
        )
        losses: list[Tensor] = []
        for _ in range(3):
            for batch_media, batch_label in loader:
                result = step.train_step(media=batch_media, label=batch_label)
                losses.append(result["loss"].mean())
            finish_epoch(step)
        actual = {"losses": torch.stack(losses)}
        actual.update(model_record(model, prefix="final"))
        actual.update(optimizer_record(cast(torch.optim.Optimizer, step.optimizer)))
        actual.update(
            {
                f"gradient/{name}": gradient.detach().clone()
                for (name, _), gradient in zip(
                    model.named_parameters(),
                    step.gradients,
                    strict=True,
                )
            },
        )
        evaluated = step.eval_loss(media=media, label=label)
        actual["eval_logits"] = evaluated["model"]
        actual["eval_loss"] = evaluated["loss"].mean()
        actual["probabilities"] = evaluated["model"].softmax(dim=1)[:, 1]
        actual.update(random_record(pins))
    assert_tensor_golden_part(
        _CWD / "testdata" / f"{model_type.__name__.lower()}.pt",
        prefix="train/",
        record={f"train/{key}": value for key, value in actual.items()},
    )
    assert step.global_step == 3


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
@pytest.mark.parametrize("geometry", ["full", "tiny"])
def test_three_updates_at_each_geometry(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
    geometry: Literal["full", "tiny"],
) -> None:
    cfg = model_type.Config(8, 2)
    cfg.num_samples = 500
    if geometry == "tiny":
        cfg.channels_in = 3
        cfg.num_samples = 100
        cfg.channels_hidden = 2
        if isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
            cfg.channels_projection = 2
    step = source_step(cfg, epochs=3).make()
    # All three networks require a singleton input plane and two target classes.
    media = (
        torch.arange(4 * cfg.channels_in * cfg.num_samples)
        .reshape(4, 1, cfg.channels_in, cfg.num_samples)
        .float()
        / 100
    )
    label = torch.tensor([0, 1, 0, 1])
    for _ in range(3):
        result = step.train_step(media=media, label=label)
        assert result["loss"].shape == (4,)
        assert torch.isfinite(result["loss"]).all()
        finish_epoch(step)
    evaluated = step.eval_loss(media=media, label=label)
    assert evaluated["model"].shape == (4, 2)
    assert torch.isfinite(evaluated["loss"]).all()
    assert step.global_step == 3


def off_target_loss(smoothing: float) -> SimpleLoss:
    """Build the standard loss slot around the off-target objective."""
    objective = SimpleLoss.Config()
    objective.loss_fn = partial(cross_entropy_off_target, smoothing=smoothing)
    return objective.make()


def test_off_target_smoothing_uses_reference_distribution() -> None:
    logits = torch.tensor([[2.0, -1.0], [-3.0, 4.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1, 0])
    expected = -(
        torch.tensor([[0.95, 0.05], [0.05, 0.95], [0.95, 0.05]])
        * logits.log_softmax(dim=-1)
    ).sum(dim=-1)
    assert torch.equal(off_target_loss(0.05)(logits, label=labels)["loss"], expected)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_off_target_smoothing_preserves_dense_values_and_gradients(
    reduction: Literal["none", "mean", "sum"],
) -> None:
    logits = torch.tensor(
        [[2.0, -1.0, 0.5, 3.0], [-3.0, 4.0, 1.0, -2.0], [0.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    target = torch.tensor([0, 1, 3])
    distribution = torch.tensor(
        [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.05, 0.85]],
    )
    expected = (-distribution * logits.log_softmax(-1)).sum(-1)
    if reduction == "mean":
        expected = expected.mean()
    if reduction == "sum":
        expected = expected.sum()
    actual = cross_entropy_off_target(
        logits,
        target,
        smoothing=0.15,
        reduction=reduction,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.sum(), logits)[0],
        torch.autograd.grad(expected.sum(), logits)[0],
    )


def test_off_target_default_has_no_smoothing() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]])
    target = torch.tensor([0, 1])
    actual = cross_entropy_off_target(logits, target)
    assert torch.equal(actual, cross_entropy_off_target(logits, target, smoothing=0.0))


def test_off_target_and_standard_smoothing_are_distinct() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]])
    target = torch.tensor([0, 1])
    actual = cross_entropy_off_target(logits, target, smoothing=0.05)
    assert not torch.equal(
        actual,
        nn.functional.cross_entropy(
            logits,
            target,
            label_smoothing=0.05,
            reduction="none",
        ),
    )


@pytest.mark.parametrize("smoothing", [0.0, 1.0])
def test_off_target_zero_mass_excludes_infinite_log_probabilities(
    smoothing: float,
) -> None:
    # Binary off-target smoothing at either endpoint selects exactly one class.
    logits = torch.tensor([[-math.inf, 0.0], [0.0, -math.inf]], requires_grad=True)
    target = torch.tensor([1, 0]) if smoothing == 0 else torch.tensor([0, 1])
    loss = cross_entropy_off_target(logits, target, smoothing=smoothing)
    assert torch.equal(loss, torch.zeros(2))
    assert torch.equal(
        torch.autograd.grad(loss.sum(), logits)[0],
        torch.zeros_like(logits),
    )


def test_eval_loss_has_no_dropout_or_gradients() -> None:
    model = MultiScaleEEGNet.Config(3, 2)
    model.num_samples = 100
    model.channels_hidden = model.channels_projection = 2
    step = source_step(model, epochs=3).make()
    first = step.eval_loss(
        # MultiScaleEEGNet.__init__ fixes the Conv2d input plane to one.
        media=torch.zeros(4, 1, 3, 100),
        label=torch.tensor([0, 1, 0, 1]),
    )
    second = step.eval_loss(
        # MultiScaleEEGNet.__init__ fixes the Conv2d input plane to one.
        media=torch.zeros(4, 1, 3, 100),
        label=torch.tensor([0, 1, 0, 1]),
    )
    assert torch.equal(first["loss"], second["loss"])
    assert not first["loss"].requires_grad


def test_smoothed_objective_cost_counts_dense_targets_and_log_softmax() -> None:
    cfg = SimpleLoss.Config()
    cfg.loss_fn = partial(cross_entropy_off_target, smoothing=0.05)
    cfg.channels_out = 4
    logits = torch.arange(24).reshape(6, 4).float().requires_grad_()
    actual = cfg.make()(logits, label=torch.tensor([0, 1, 2, 3, 0, 2]))["loss"]
    assert actual.shape == (6,)
    actual.sum().backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    counted = cost(cfg, seq_len=2, batch_size=3, dtype=torch.float32)
    assert counted["flops", "matmul"].sum() == 0
    assert counted.params == counted.params_active == 0
    assert counted["flops", "primal", "elementwise"].sum() == 126
    assert counted["flops", "primal", "reduction"].sum() == 54
    assert counted["flops", "adjoint", "elementwise"].sum() == 96
    assert counted["flops", "adjoint", "reduction"].sum() == 18
    assert counted["bytes", "primal", "elementwise"].sum() == 1_344
    assert counted["bytes", "adjoint", "elementwise"].sum() == 792
    assert counted["bytes", "primal", "selection", torch.int64] == 48
    assert counted == cost(cfg, seq_len=3, batch_size=2, dtype=torch.float32)


def test_smoothed_objective_cost_requires_the_class_count() -> None:
    cfg = SimpleLoss.Config()
    cfg.loss_fn = cross_entropy_off_target
    with pytest.raises(ValueError, match="channels_out"):
        cost(cfg, seq_len=2, batch_size=3, dtype=torch.float32)


@pytest.mark.parametrize("composed", [False, True])
def test_unsupported_closure_optimizer_is_rejected_before_training(
    composed: bool,
) -> None:
    config = linear_step_config()
    config.optimizer = (
        CompositeOptimizer.Config(optimizers=[Newton.Config()])
        if composed
        else Newton.Config()
    )
    with pytest.raises(ValueError, match="closure"):
        config.make()


def test_train_loss_uses_the_same_prepared_batch_as_training() -> None:
    step = linear_step_config().make()
    before = model_record(step.model, prefix="weights")
    result = step.train_loss(
        media=torch.arange(12).reshape(4, 3).double(),
        label=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )
    assert result["model"].shape == (4, 2)
    assert result["model"].dtype == torch.float32
    assert result["loss"].shape == (4,)
    assert step.global_step == 0
    compare_record(before, model_record(step.model, prefix="weights"))
    assert all(parameter.grad is None for parameter in step.model.parameters())


def _large_linear_loss(logits: Tensor, **batch: object) -> LossOutput:
    del batch
    return {"loss": logits.sum(dim=-1) * 1e20}


def test_disabled_clipping_keeps_large_finite_gradients_unchanged() -> None:
    config = RecordingStep.Config()
    config.update(linear_step_config())
    config.gradient_clip_norm = float("inf")
    config.loss = PartialConfig(_large_linear_loss)
    config.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    step = config.make()
    parameters = list(step.model.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    step.train_step(media=torch.ones(2, 3), label=torch.tensor([0, 1]))
    for old, parameter, gradient in zip(
        before,
        parameters,
        step.gradients,
        strict=True,
    ):
        expected = old.add(torch.full_like(old, 1e20), alpha=-0.1)
        assert torch.equal(parameter, expected)
        assert parameter.grad is None
        assert torch.equal(gradient, torch.full_like(parameter, 1e20))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
