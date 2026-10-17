"""Training step for the MLS-Bench vision tasks.

One optimizer update per batch, reduced to the arithmetic of the reference
loop: zero the gradients, run the forward, take a scalar loss, backpropagate,
step. The loss and the learning-rate schedule are INJECTED. The editable region
of the reference is a function from ``(logits, targets, config)`` to a scalar,
so here it is a slot holding such a function; a new loss is a different value
in it, never a branch in this step.

The loss receives what the reference's ``config`` dict carries, as a
:class:`LossContext`, so an epoch-dependent loss is expressible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import math

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain

import torch

from priml.baselines.mlsbench.pytorch_vision.model import ResNet
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.loss.custom_types import LossOutput
    from priml.train.custom_types import TrainStepOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class LossContext:
    """What the loss may read besides the logits and the labels."""

    num_classes: int
    """Width of the logits."""

    epoch: int
    """Completed passes over the training data, so zero during the first."""

    total_epochs: int
    """Passes the run is scheduled for."""


class LearningRateSchedule(Protocol):
    """Maps an epoch to the learning rate every parameter group uses in it."""

    def __call__(self, epoch: int, /, *, base_lr: float, total_epochs: int) -> float:
        """Return the learning rate of ``epoch`` for a group starting at ``base_lr``."""
        raise NotImplementedError


UNSUPPORTED: Final = (
    "accumulate_grad_batches",
    "lr_schedule",
    "ema",
    "gradient_clip_norm",
    "skip_step_on_nonfinite_grad",
)
"""Inherited settings the base applies in ``step``, which this update bypasses."""


class VisionTrainStep(TrainStep):
    """Model plus optimization for one MLS-Bench vision experiment."""

    class Config(Makes["VisionTrainStep"], TrainStep.Config, kw_only=True):
        """Model, optimizer, loss and schedule for one run."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: Makeable[nn.Module] = field(default_factory=ResNet.Config)
        """Network to train."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Compilation fuses kernels and changes the last bits, so the baseline runs eager."""

        # ---- This recipe's own. ----

        schedule: Makeable[LearningRateSchedule] = field(
            default_factory=lambda: PartialConfig(cosine_annealing),
        )
        """Maps the epoch to a learning rate, applied before every update.

        Read instead of the base's ``lr_schedule``, whose multiplier cannot
        express a rate carried through a recurrence. Its horizon is
        ``train_budget_epochs``."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.train_budget_epochs):
            raise ValueError(
                "train_budget_epochs must be finite: the schedule and the loss "
                f"context both read it; got {config.train_budget_epochs}.",
            )
        if config.train_budget_epochs < 1 or config.train_budget_epochs % 1:
            raise ValueError(
                "train_budget_epochs must be a whole number of passes; got "
                f"{config.train_budget_epochs}.",
            )
        defaults = VisionTrainStep.Config()
        ignored = [
            name
            for name in UNSUPPORTED
            if getattr(config, name) != getattr(defaults, name)
        ]
        if ignored:
            raise ValueError(
                f"{type(self).__name__} runs the reference's bare update, which "
                f"has no {', '.join(ignored)}; leave them at their defaults.",
            )
        super().__init__(config)
        self.config: VisionTrainStep.Config = config
        self.schedule: LearningRateSchedule = config.schedule.make()

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one update: forward, loss, backward, optimizer step.

        Args:
          **batch: Preprocessed batch with ``media`` and ``label``.

        Returns:
          result: ``loss`` (a scalar) and ``model`` (logits).

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        self._apply_schedule()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self(media=media)
        assert isinstance(logits, Tensor)
        loss = self.objective(
            logits,
            media=media,
            label=label,
            context=self._context(logits),
        )
        loss.backward()
        with self.timer_step:
            self.optimizer.step()
        return {"loss": loss.detach(), "model": logits.detach()}

    def objective(
        self,
        logits: Tensor,
        *,
        media: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        """Return the scalar the update backpropagates: the loss slot's.

        A recipe that adds a term to the loss overrides this, and only this.

        Args:
          logits: ``(B, C)`` class scores of the batch.
          media: ``(B, C_in, H, W)`` the images; the loss slot does not read them.
          label: ``(B,)`` target classes.
          context: What the loss may read besides the logits and the labels.

        Returns:
          loss: The scalar the optimizer descends.

        """
        del media
        return self.loss(logits, label=label, context=context)["loss"]

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score a test batch with plain cross-entropy, whatever the training loss.

        Args:
          **batch: Preprocessed batch with ``media`` and ``label``.

        Returns:
          result: ``loss`` (a scalar) and ``model`` (logits).

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        logits = self.call_eval(media=media)
        assert isinstance(logits, Tensor)
        return {"loss": functional.cross_entropy(logits, label), "model": logits}

    def _apply_schedule(self) -> None:
        """Set every parameter group's rate for the current epoch."""
        epoch = self.timer_epoch.global_count
        for group in self.optimizer.param_groups:
            group["lr"] = self.schedule(
                epoch,
                base_lr=from_plain(cast(object, group["initial_lr"]), float),
                total_epochs=int(self.config.train_budget_epochs),
            )

    def _context(self, logits: Tensor) -> LossContext:
        """Describe the current step to the loss."""
        return LossContext(
            num_classes=logits.shape[-1],
            epoch=self.timer_epoch.global_count,
            total_epochs=int(self.config.train_budget_epochs),
        )


def cosine_annealing(epoch: int, /, *, base_lr: float, total_epochs: int) -> float:
    """Anneal the learning rate to zero along half a cosine, one step per epoch.

    Evaluates the recurrence ``lr_e = lr_{e-1} * (1 + cos(pi e / T)) / (1 +
    cos(pi (e - 1) / T))`` from ``lr_0 = base_lr``. It is not rewritten as the
    closed form ``base_lr * (1 + cos(pi e / T)) / 2``: the two round
    differently, and a run that must reproduce the reference's rates bit for
    bit follows the reference's arithmetic.

    Args:
      epoch: Completed epochs, from 0 to ``total_epochs``.
      base_lr: Rate of epoch 0.
      total_epochs: Epochs over which the rate reaches zero.

    Returns:
      lr: Learning rate to use during ``epoch``.

    References:
      https://arxiv.org/abs/1608.03983
        Loshchilov and Hutter 2017, "SGDR: Stochastic Gradient Descent with
        Warm Restarts."

    """
    lr = base_lr
    for step in range(1, epoch + 1):
        lr = (
            (1 + math.cos(math.pi * step / total_epochs))
            / (1 + math.cos(math.pi * (step - 1) / total_epochs))
            * lr
        )
    return lr


def cross_entropy(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
) -> LossOutput:
    """Return the mean cross-entropy over the batch.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.

    Returns:
      loss: Scalar under ``"loss"``.

    """
    del context
    return {"loss": functional.cross_entropy(logits, label)}


def label_smoothing(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    smoothing: float,
) -> LossOutput:
    """Return cross-entropy against ``(1 - smoothing) * one_hot + smoothing / C``.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      smoothing: Probability mass spread uniformly over the classes.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/1512.00567
        Szegedy et al. 2016, "Rethinking the Inception Architecture for
        Computer Vision."

    """
    del context
    return {
        "loss": functional.cross_entropy(logits, label, label_smoothing=smoothing),
    }


def focal(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    gamma: float,
) -> LossOutput:
    """Return cross-entropy scaled per example by ``(1 - p) ** gamma``.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      gamma: Focusing exponent; zero recovers cross-entropy.

    Returns:
      loss: Scalar under ``"loss"``, where ``p`` is the probability the model
        gives the target class.

    References:
      https://arxiv.org/abs/1708.02002
        Lin et al. 2017, "Focal Loss for Dense Object Detection."

    """
    del context
    ce = functional.cross_entropy(logits, label, reduction="none")
    return {"loss": ((1 - torch.exp(-ce)) ** gamma * ce).mean()}


def poly(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    epsilon: float,
) -> LossOutput:
    """Return cross-entropy plus ``epsilon * (1 - p)`` for the first polynomial term.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      epsilon: Weight of the correction; zero recovers cross-entropy.

    Returns:
      loss: Scalar under ``"loss"``, where ``p`` is the probability the model
        gives the target class.

    References:
      https://arxiv.org/abs/2204.12511
        Leng et al. 2022, "PolyLoss: A Polynomial Expansion Perspective of
        Classification Loss Functions."

    """
    del context
    ce = functional.cross_entropy(logits, label)
    p = functional.softmax(logits, dim=-1).gather(1, label.unsqueeze(1)).squeeze(1)
    return {"loss": ce + epsilon * (1 - p).mean()}
