"""The vision training step with each gradient compressed before the update.

The task simulates data-parallel training on one process: after the backward
pass, every parameter's gradient goes through a compressor and comes back as
the gradient the optimizer steps with. The editable region of the reference is
that compressor, so here it is a slot; the forward, loss, schedule and
optimizer are the vision step's.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, cast, override

import math

from configgle import Makeable, Makes2, PartialConfig
from torch import Tensor

from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.opt.gradient_compression.compressors import (
    Compressor,
    identity,
)
from priml.custom_types import CheckpointableProtocol


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from priml.train.custom_types import TrainStepOutput


class CompressedTrainStep(VisionTrainStep):
    """The vision step, stepping with compressed gradients."""

    class Config(Makes2["CompressedTrainStep"], VisionTrainStep.Config, kw_only=True):
        """The vision step's slots and the compressor."""

        compressor: Makeable[Compressor] = field(
            default_factory=lambda: PartialConfig(identity),
        )
        """Maps each parameter's gradient to the one the optimizer uses."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.compressor: Compressor = config.compressor.make()

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one update: forward, loss, backward, compression, optimizer step.

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
        self.compress_gradients()
        with self.timer_step:
            self.optimizer.step()
        return {"loss": loss.detach(), "model": logits.detach()}

    def compress_gradients(self) -> None:
        """Replace every parameter's gradient by its compression, in registration order."""
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                parameter.grad = self.compressor(parameter.grad, name=name)

    class StateDict(VisionTrainStep.StateDict):
        """The vision step's state and, for a stateful compressor, its own."""

        compressor: Mapping[str, object]

    @override
    def state_dict(self) -> StateDict:
        """Return the step's state and the compressor's, so residuals survive a resume.

        Returns:
          state: The vision step's entries and ``compressor``, empty when the
            compressor keeps no state.

        """
        compressor = self.compressor
        return {  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.
            **super().state_dict(),
            "compressor": compressor.state_dict()
            if isinstance(compressor, CheckpointableProtocol)
            else {},
        }

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
        """Restore the step and the compressor.

        Args:
          state_dict: State as returned by :meth:`state_dict`.
          strict: As the vision step's.
          load_optimizer: As the vision step's; also gates the compressor,
            whose residuals belong to the optimizer's trajectory.
          remap: As the vision step's.

        """
        super().load_state_dict(
            state_dict,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        compressor = self.compressor
        if load_optimizer and isinstance(compressor, CheckpointableProtocol):
            state = cast(CompressedTrainStep.StateDict, state_dict)
            compressor.load_state_dict(state["compressor"])


def warmup_cosine(
    epoch: int,
    /,
    *,
    base_lr: float,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """Rise linearly for ``warmup_epochs``, then fall along half a cosine to a floor.

    Warmup epoch ``e`` runs at ``base_lr * (e + 1) / (warmup_epochs + 1)``, so
    the first epoch is not at zero and the last warmup epoch stops short of
    ``base_lr``. The cosine then starts at ``base_lr`` and would reach the floor
    ``base_lr * min_lr_ratio`` at epoch ``total_epochs``, one past the last.

    Args:
      epoch: Completed epochs, from 0.
      base_lr: Peak rate.
      total_epochs: Epochs of the run.
      warmup_epochs: Epochs of linear rise.
      min_lr_ratio: Floor of the cosine, as a fraction of ``base_lr``.

    Returns:
      lr: Learning rate to use during ``epoch``.

    References:
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet
        in 1 Hour."
      https://arxiv.org/abs/1608.03983
        Loshchilov and Hutter 2017, "SGDR: Stochastic Gradient Descent with
        Warm Restarts."

    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / (warmup_epochs + 1)
    min_lr = base_lr * min_lr_ratio
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
