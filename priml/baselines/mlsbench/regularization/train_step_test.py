"""Tests for the vision step that adds a regularization term."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import functools

from configgle import Makeable, PartialConfig

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision import reference
from priml.baselines.mlsbench.pytorch_vision.train_step import LossContext
from priml.baselines.mlsbench.pytorch_vision.train_step_test import (
    CLASSES,
    EPOCHS,
    batch,
    tiny_config,
)
from priml.baselines.mlsbench.regularization.experiments import exp001
from priml.baselines.mlsbench.regularization.regularizers import (
    spatial_coactivation,
    zero,
)
from priml.baselines.mlsbench.regularization.train_step import (
    RegularizedTrainStep,
    Regularizer,
)


if TYPE_CHECKING:
    from torch import Tensor, nn


def config() -> RegularizedTrainStep.Config:
    """Return the vision test step's config, on the regularized step."""
    return RegularizedTrainStep.Config().update(tiny_config())


class Recording:
    """A regularizer that returns a fixed value and notes what it was given."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.seen: list[tuple[nn.Module, Tensor, Tensor, Tensor, LossContext]] = []

    def __call__(
        self,
        model: nn.Module,
        /,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        """Return :attr:`value` as a tensor on the logits."""
        self.seen.append((model, media, logits, label, context))
        return logits.sum() * 0 + self.value


def test_the_default_regularizer_adds_zero_and_leaves_the_update_unchanged() -> None:
    plain = tiny_config().make()
    regularized = config().make()
    made = regularized.regularization
    assert isinstance(made, functools.partial)
    assert cast("functools.partial[Tensor]", made).func is zero
    regularized.model.load_state_dict(plain.model.state_dict())
    ours, theirs = regularized.train_step(**batch()), plain.train_step(**batch())
    assert torch.equal(ours["loss"], theirs["loss"])
    for a, b in zip(
        regularized.model.parameters(),
        plain.model.parameters(),
        strict=True,
    ):
        assert torch.equal(a, b)


def test_the_term_is_added_to_the_loss_and_sees_the_batch() -> None:
    step = config().make()
    recording = step.regularization = Recording(2.5)
    step.timer_epoch.global_count = 3
    data = batch()
    result = step.train_step(**data)
    ((model, media, logits, label, context),) = recording.seen
    assert model is step.model
    assert torch.equal(media, data["media"])
    assert torch.equal(label, data["label"])
    assert torch.equal(logits.detach(), result["model"])
    assert context == LossContext(num_classes=CLASSES, epoch=3, total_epochs=EPOCHS)
    plain = torch.nn.functional.cross_entropy(result["model"], data["label"])
    assert torch.equal(result["loss"], plain + 2.5)


def test_settings_the_bare_update_bypasses_are_refused() -> None:
    cfg = config()
    cfg.gradient_clip_norm = 1.0
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        cfg.make()


def test_an_active_dropblock_term_is_positive_and_reaches_the_gradient() -> None:
    def updated(regularization: Makeable[Regularizer]) -> tuple[float, list[Tensor]]:
        cfg = config()
        cfg.regularization = regularization
        with reference.seeded(reference.SEED):
            step = cfg.make()
        step.timer_epoch.global_count = EPOCHS - 1
        step.regularization = spy = _Spy(step.regularization)
        _ = step.train_step(**batch())
        (value,) = spy.values
        return value, [p.detach().clone() for p in step.model.parameters()]

    dropblock = exp001().step
    assert isinstance(dropblock, RegularizedTrainStep.Config)
    off = PartialConfig(spatial_coactivation)
    off.block_size = 3
    off.strength = 0.0
    off.delay = 0.2
    (value, with_term), (none, without) = (
        updated(dropblock.regularization),
        updated(off),
    )
    assert value > 0
    assert none == 0
    assert any(not torch.equal(a, b) for a, b in zip(with_term, without, strict=True))


class _Spy:
    """Forwards to a regularizer and keeps each value it returned."""

    def __init__(self, regularizer: Regularizer) -> None:
        self.regularizer = regularizer
        self.values: list[float] = []

    def __call__(
        self,
        model: nn.Module,
        /,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        """Return the wrapped term and note its value."""
        value = self.regularizer(
            model,
            media=media,
            logits=logits,
            label=label,
            context=context,
        )
        self.values.append(value.item())
        return value


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
