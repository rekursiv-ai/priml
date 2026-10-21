"""Tests for the vision training step, its schedule and its default loss."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Final, cast, override

import copy
import itertools
import math

from configgle import Makes2, PartialConfig
from torch import Tensor
from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import (
    LossContext,
    VisionTrainStep,
    cosine_annealing,
    cross_entropy,
)
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from priml.loss.custom_types import LossOutput

CLASSES: Final = 7
"""Logits of the test network."""

EPOCHS: Final = 4
"""Schedule horizon of the test steps."""

CONTEXT: Final = LossContext(num_classes=4, epoch=0, total_epochs=2)
"""What the loss tests tell the loss."""


def tiny_config() -> VisionTrainStep.Config:
    """Return a CPU step small enough to run in a few milliseconds."""
    return sized(VisionTrainStep.Config())


def sized[C: VisionTrainStep.Config](config: C) -> C:
    """Shrink ``config`` to a CPU step that runs in a few milliseconds."""
    model = config.model = ResNet.Config()
    model.channels_in = 3
    model.channels_out = CLASSES
    model.channels_hidden = (4, 5, 6)
    model.blocks_per_stage = (1, 1, 1)
    sgd = config.optimizer = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.1
    sgd.momentum = 0.9
    config.loss = PartialConfig(cross_entropy)
    config.parallelism = NoParallel.Config(device="cpu")
    config.train_budget_epochs = EPOCHS
    return config


def batch() -> dict[str, Tensor]:
    """Return two images and their labels."""
    generator = torch.Generator().manual_seed(0)
    return {
        "media": torch.randn(2, 3, 4, 6, generator=generator),
        "label": torch.randint(0, CLASSES, (2,), generator=generator),
    }


def built(config: VisionTrainStep.Config | None = None) -> VisionTrainStep:
    """Make a step, narrowing what ``make`` returns."""
    step = (config or tiny_config()).make()
    assert isinstance(step, VisionTrainStep)
    return step


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("accumulate_grad_batches", 2),
        ("lr_schedule", PartialConfig(cosine_annealing)),
        ("ema", EMA.Config()),
        ("gradient_clip_norm", 1.0),
        ("skip_step_on_nonfinite_grad", True),
    ],
)
def test_a_setting_the_bare_update_would_ignore_is_refused(
    name: str,
    value: object,
) -> None:
    config = tiny_config()
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        built(config)


def logits() -> tuple[Tensor, Tensor]:
    """Return three rows of four scores and their labels."""
    scores = torch.tensor(
        [[2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 3.0, 0.0]],
    )
    return scores, torch.tensor([0, 3, 2])


class RecordingLoss:
    """Cross-entropy that notes what the step told it."""

    def __init__(self) -> None:
        self.seen: list[LossContext] = []

    def __call__(
        self,
        scores: Tensor,
        *,
        label: Tensor,
        context: LossContext,
    ) -> LossOutput:
        """Return cross-entropy and note the context."""
        self.seen.append(context)
        return cross_entropy(scores, label=label, context=context)


def constant_loss(scores: Tensor, *, label: Tensor, context: LossContext) -> LossOutput:
    """Return a loss that cannot be the cross-entropy."""
    del label, context
    return {"loss": scores.sum() * 0 + 123.0}


def halving(epoch: int, /, *, base_lr: float, total_epochs: int) -> float:
    """Return half the base rate, whatever the epoch."""
    del epoch, total_epochs
    return base_lr / 2


def test_train_step_returns_a_scalar_loss_and_the_logits() -> None:
    result = built().train_step(**batch())
    assert result["loss"].shape == ()
    assert result["model"].shape == (2, CLASSES)


def test_train_step_updates_the_weights_and_counts_one_step() -> None:
    step = built()
    model = step.model
    assert isinstance(model, ResNet)
    stem, head = model.stem.weight.clone(), model.head.weight.clone()
    _ = step.train_step(**batch())
    assert step.global_step == 1
    assert not torch.equal(stem, model.stem.weight)
    assert not torch.equal(head, model.head.weight)


def test_training_lowers_the_loss_on_a_batch_it_repeats() -> None:
    step = built()
    losses = [step.train_step(**batch())["loss"] for _ in range(6)]
    assert losses[-1] < losses[0]


def test_the_loss_receives_the_logits_the_label_and_the_epoch() -> None:
    recorder = RecordingLoss()
    config = tiny_config()
    config.loss = PartialConfig(recorder)
    step = built(config)
    step.timer_epoch.global_count = 3
    _ = step.train_step(**batch())
    assert recorder.seen == [
        LossContext(num_classes=CLASSES, epoch=3, total_epochs=EPOCHS),
    ]


def test_the_learning_rate_of_the_current_epoch_is_applied_before_the_update() -> None:
    step = built()
    _ = step.train_step(**batch())
    assert step.optimizer.param_groups[0]["lr"] == 0.1
    step.timer_epoch.global_count = 2
    _ = step.train_step(**batch())
    assert step.optimizer.param_groups[0]["lr"] == cosine_annealing(
        2,
        base_lr=0.1,
        total_epochs=EPOCHS,
    )


def test_a_schedule_slot_replaces_cosine_annealing() -> None:
    config = tiny_config()
    config.schedule = PartialConfig(halving)
    step = built(config)
    _ = step.train_step(**batch())
    assert step.optimizer.param_groups[0]["lr"] == 0.05


def test_the_schedule_starts_from_the_initial_rate_not_the_current_one() -> None:
    step = built()
    step.optimizer.param_groups[0]["initial_lr"] = 0.5
    step.timer_epoch.global_count = 1
    _ = step.train_step(**batch())
    assert step.optimizer.param_groups[0]["lr"] == cosine_annealing(
        1,
        base_lr=0.5,
        total_epochs=EPOCHS,
    )


def test_eval_loss_is_cross_entropy_whatever_the_training_loss() -> None:
    config = tiny_config()
    config.loss = PartialConfig(constant_loss)
    step = built(config)
    data = batch()
    result = step.eval_loss(**data)
    assert result["loss"] == functional.cross_entropy(result["model"], data["label"])
    assert result["loss"] != 123.0
    assert all(p.grad is None for p in step.model.parameters())
    assert step.global_step == 0


def test_eval_loss_runs_the_network_in_evaluation_mode() -> None:
    step = built()
    data = batch()
    _ = step.train_step(**data)
    result = step.eval_loss(**data)
    step.model.eval()
    with torch.no_grad():
        assert torch.equal(
            result["model"],
            cast(Tensor, step.model(data["media"])),
        )


def test_the_schedule_horizon_must_be_finite() -> None:
    config = tiny_config()
    config.train_budget_epochs = math.inf
    with pytest.raises(ValueError, match="finite"):
        built(config)


def test_the_baseline_runs_eager() -> None:
    assert VisionTrainStep.Config().compile is None


def test_a_resumed_step_continues_the_same_trajectory() -> None:
    first = built()
    _ = first.train_step(**batch())
    second = built()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    _ = first.train_step(**batch())
    _ = second.train_step(**batch())
    for want, got in zip(
        first.model.state_dict().values(),
        second.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(want, got)
    assert second.global_step == 2


def test_the_loss_context_cannot_be_modified() -> None:
    field = "epoch"
    with pytest.raises(FrozenInstanceError):
        setattr(CONTEXT, field, 1)


def test_cosine_annealing_starts_at_the_base_rate_and_ends_at_zero() -> None:
    assert cosine_annealing(0, base_lr=0.1, total_epochs=200) == 0.1
    assert cosine_annealing(200, base_lr=0.1, total_epochs=200) == 0.0


def test_cosine_annealing_decreases_and_halves_at_the_midpoint() -> None:
    rates = [
        cosine_annealing(epoch, base_lr=0.1, total_epochs=200) for epoch in range(201)
    ]
    assert all(a > b for a, b in itertools.pairwise(rates))
    assert rates[100] == pytest.approx(0.05, rel=1e-12)


def test_cosine_annealing_follows_the_recurrence_not_the_closed_form() -> None:
    closed = [0.1 * (1 + math.cos(math.pi * epoch / 200)) / 2 for epoch in range(200)]
    rates = [
        cosine_annealing(epoch, base_lr=0.1, total_epochs=200) for epoch in range(200)
    ]
    assert rates != closed
    assert all(
        a == pytest.approx(b, rel=1e-12) for a, b in zip(rates, closed, strict=True)
    )


def test_cosine_annealing_is_undefined_past_the_epoch_after_the_horizon() -> None:
    with pytest.raises(ZeroDivisionError):
        cosine_annealing(EPOCHS + 1, base_lr=0.1, total_epochs=EPOCHS)


def test_cross_entropy_is_the_mean_negative_log_likelihood() -> None:
    scores, label = logits()
    expected = -functional.log_softmax(scores, dim=-1)[torch.arange(3), label].mean()
    assert cross_entropy(scores, label=label, context=CONTEXT)["loss"] == pytest.approx(
        float(expected),
        rel=1e-6,
    )


class DoubledStep(VisionTrainStep):
    """A step whose objective is twice its loss."""

    class Config(Makes2["DoubledStep"], VisionTrainStep.Config, kw_only=True):
        """The vision step's fields."""

    @override
    def objective(
        self,
        logits: Tensor,
        *,
        media: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        return 2 * super().objective(
            logits,
            media=media,
            label=label,
            context=context,
        )


def test_the_objective_is_what_the_update_backpropagates() -> None:
    plain = built()
    doubled = sized(DoubledStep.Config()).make()
    assert isinstance(doubled, DoubledStep)
    doubled.load_state_dict(plain.state_dict())
    first = plain.train_step(**batch())["loss"]
    second = doubled.train_step(**batch())["loss"]
    assert torch.equal(second, 2 * first)


@pytest.mark.parametrize("epochs", [2.5, 0.0])
def test_a_budget_that_is_not_a_positive_whole_number_is_refused(epochs: float) -> None:
    config = tiny_config()
    config.train_budget_epochs = epochs
    with pytest.raises(ValueError, match="train_budget_epochs"):
        built(config)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
