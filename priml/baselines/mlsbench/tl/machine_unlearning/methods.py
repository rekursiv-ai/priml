"""Unlearning updates: the task's editable slot and its four published baselines.

The reference calls its method once per step with a retained batch and a
forget batch, both on the device, the network in training mode, and one Adam
optimizer the harness owns. A method is a stateful object built fresh for each
run: the teacher-based ones copy the network the first time they are called,
so their teachers are the pretrained network, frozen.

Every method issues the reference's operations in the reference's order. The
two forwards of :class:`NegativeGradient` each update batch-norm statistics,
and :class:`BadTeacher`'s re-initialization draws from the global generator
after the first step's batches were drawn, so reordering either is not a
refactor.
"""

from __future__ import annotations

from typing import Protocol

import copy

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.mlsbench.dl.vision.model import ModelInitFn


type Batch = tuple[Tensor, Tensor]
"""Images and their labels, on the device."""


class Classifier(Protocol):
    """A network as a method uses it: images in, logits out."""

    def __call__(self, media: Tensor, /) -> Tensor:
        """Return ``(B, classes)`` logits for ``(B, C, H, W)`` images."""
        raise NotImplementedError


class UnlearningMethod(Protocol):
    """One unlearning update, given a retained and a forget batch."""

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Update ``model`` in place and return the scalar loss it reports."""
        raise NotImplementedError


class RetainFinetune:
    """Keep training on the retained data alone; the forget batch is ignored."""

    class Config(Fig["RetainFinetune"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Descend the cross-entropy of the retained batch.

        Args:
          model: The network, in training mode.
          retain_batch: Retained images and labels.
          forget_batch: Unused.
          optimizer: Steps ``model``'s parameters.
          step: Unused.
          epoch: Unused.

        Returns:
          loss: The retained batch's cross-entropy.

        """
        del forget_batch, step, epoch
        retain_x, retain_y = retain_batch
        loss = functional.cross_entropy(model_logits(model, retain_x), retain_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach()


class NegativeGradient:
    """Descend the retained loss while ascending a weighted forget loss."""

    class Config(Fig["NegativeGradient"]):
        """Weight of the ascended forget loss."""

        forget_weight: float = 0.5
        """Multiplier of the forget batch's cross-entropy, subtracted from the loss."""

    def __init__(self, config: Config) -> None:
        self.forget_weight = config.forget_weight

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Descend ``retain - forget_weight * forget`` cross-entropy.

        Args:
          model: The network, in training mode.
          retain_batch: Retained images and labels.
          forget_batch: Forget images and labels.
          optimizer: Steps ``model``'s parameters.
          step: Unused.
          epoch: Unused.

        Returns:
          loss: The combined loss.

        References:
          https://arxiv.org/abs/1911.04933
            Golatkar, Achille and Soatto 2020, "Eternal Sunshine of the Spotless
            Net: Selective Forgetting in Deep Networks."

        """
        del step, epoch
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        retain_loss = functional.cross_entropy(model_logits(model, retain_x), retain_y)
        forget_loss = functional.cross_entropy(model_logits(model, forget_x), forget_y)
        loss = retain_loss - self.forget_weight * forget_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach()


def reinitialize(model: nn.Module) -> None:
    """Draw fresh He-normal weights for every layer, in ``modules()`` order.

    Convolution weights take ``fan_out``, linear weights ``fan_in``; linear
    biases are zeroed and batch norms reset to unit scale and zero shift. A
    convolution's bias and a batch norm's running statistics are left as they
    are, as the reference leaves them.

    Args:
      model: The network, changed in place.

    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class BadTeacher:
    """Distill retained images from the original network, forget images from a random one."""

    class Config(Fig["BadTeacher"]):
        """Distillation temperature and how the incompetent teacher is drawn."""

        temperature: float = 1.0
        """Divides every logit before the softmaxes of the distillation."""

        reinit: ModelInitFn = reinitialize
        """Re-draws a copy of the network into the incompetent teacher, in place."""

    def __init__(self, config: Config) -> None:
        self.temperature = config.temperature
        self.reinit = config.reinit
        self.competent: nn.Module | None = None
        """The pretrained network, frozen; copied on the first step."""
        self.incompetent: nn.Module | None = None
        """A re-initialized copy of the pretrained network, frozen."""

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Distill the joined batch, each image from the teacher its set assigns.

        Args:
          model: The network, in training mode.
          retain_batch: Retained images; labels unused.
          forget_batch: Forget images; labels unused.
          optimizer: Steps ``model``'s parameters.
          step: Unused.
          epoch: Unused.

        Returns:
          loss: KL divergence of the student from the per-image teacher.

        References:
          https://arxiv.org/abs/2205.08096
            Chundawat et al. 2023, "Can Bad Teaching Induce Forgetting?
            Unlearning in Deep Networks using an Incompetent Teacher."

        """
        del step, epoch
        if self.competent is None or self.incompetent is None:
            self.competent = _frozen(copy.deepcopy(model))
            incompetent = copy.deepcopy(model)
            self.reinit(incompetent)
            self.incompetent = _frozen(incompetent)
        retain_x, _ = retain_batch
        forget_x, _ = forget_batch
        x = torch.cat([retain_x, forget_x], dim=0)
        is_forget = torch.cat(
            [
                torch.zeros(retain_x.size(0), device=retain_x.device),
                torch.ones(forget_x.size(0), device=forget_x.device),
            ],
            dim=0,
        )
        student = model_logits(model, x)
        with torch.no_grad():
            full = model_logits(self.competent, x)
            unlearn = model_logits(self.incompetent, x)
        loss = self._unlearner_loss(student, full, unlearn, is_forget)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach()

    def _unlearner_loss(
        self,
        student: Tensor,
        full: Tensor,
        unlearn: Tensor,
        is_forget: Tensor,
    ) -> Tensor:
        """KL of the student from the label-mixed teacher distribution."""
        t = self.temperature
        f_t = functional.softmax(full / t, dim=1)
        u_t = functional.softmax(unlearn / t, dim=1)
        label = is_forget.view(-1, 1).float()
        target = label * u_t + (1.0 - label) * f_t
        log_s = functional.log_softmax(student / t, dim=1)
        return functional.kl_div(log_s, target, reduction="batchmean")


class Scrub:
    """Push the student from a frozen teacher on forget images, toward it on retained ones."""

    class Config(Fig["Scrub"]):
        """The min-max schedule and the weights of the min step."""

        max_epochs: int = 2
        """Epochs, from the first, that begin each step with a max step."""

        temperature: float = 4.0
        """Distillation temperature of both KL terms."""

        alpha: float = 0.01
        """Weight of the KL to the teacher in the min step."""

        gamma: float = 0.99
        """Weight of the cross-entropy in the min step."""

    def __init__(self, config: Config) -> None:
        self.max_epochs = config.max_epochs
        self.temperature = config.temperature
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.teacher: nn.Module | None = None
        """The pretrained network, frozen; copied on the first step."""

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Ascend the forget KL while ``epoch < max_epochs``, then descend the min loss.

        Args:
          model: The network, in training mode.
          retain_batch: Retained images and labels.
          forget_batch: Forget images; labels unused.
          optimizer: Steps ``model``'s parameters, once per phase.
          step: Unused.
          epoch: Zero-based unlearning epoch, which gates the max step.

        Returns:
          loss: The min step's ``gamma * CE + alpha * KL``.

        References:
          https://arxiv.org/abs/2302.09880
            Kurmanji et al. 2023, "Towards Unbounded Machine Unlearning."

        """
        del step
        if self.teacher is None:
            self.teacher = _frozen(copy.deepcopy(model))
        teacher = self.teacher
        retain_x, retain_y = retain_batch
        forget_x, _ = forget_batch
        if epoch < self.max_epochs:
            optimizer.zero_grad()
            student_forget = model_logits(model, forget_x)
            with torch.no_grad():
                teacher_forget = model_logits(teacher, forget_x)
            (-self._kd_kl(student_forget, teacher_forget)).backward()
            optimizer.step()
        optimizer.zero_grad()
        student_retain = model_logits(model, retain_x)
        with torch.no_grad():
            teacher_retain = model_logits(teacher, retain_x)
        retain_ce = functional.cross_entropy(student_retain, retain_y)
        retain_kl = self._kd_kl(student_retain, teacher_retain)
        loss = self.gamma * retain_ce + self.alpha * retain_kl
        loss.backward()
        optimizer.step()
        return loss.detach()

    def _kd_kl(self, student: Tensor, teacher: Tensor) -> Tensor:
        """Hinton's temperature-scaled KL of the student from the teacher."""
        t = self.temperature
        log_s = functional.log_softmax(student / t, dim=1)
        p_t = functional.softmax(teacher / t, dim=1)
        return functional.kl_div(log_s, p_t, reduction="batchmean") * (t * t)


def model_logits(model: Classifier, media: Tensor) -> Tensor:
    """Run a network, typed as the images-to-logits map it is."""
    return model(media)


def _frozen(model: nn.Module) -> nn.Module:
    """Stop ``model``'s gradients and put it in evaluation mode; return it."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model
