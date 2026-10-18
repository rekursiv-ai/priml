"""Training step for two classification heads whose losses are combined.

The vision step's update with the reference's multi-task arithmetic: the
network scores a batch with both heads, each head's loss is the vision step's
``loss`` slot against its own labels, and ``mtl_loss`` combines the two into
the scalar that is backpropagated. ``mtl_loss`` is the task's editable
``MultiTaskLoss``: a module, so a combination may hold parameters. The
optimizer is built over the network's parameters and then the combination's,
the order of the reference's ``all_params``.

A combination that changes the backward, as gradient surgery does, does so
by what it returns: the step backpropagates whatever scalar it is given.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, NotRequired, cast, override

from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.mlsbench.dl.multitask.losses import SumLoss
from priml.baselines.mlsbench.dl.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from configgle import Makeable

    from priml.train.custom_types import OptimizerProtocol, TrainStepOutput


class MultiTaskStep(VisionTrainStep):
    """Model, optimizer and loss combination for one multi-task experiment."""

    class Config(VisionTrainStep.Config):
        """The vision step's slots plus the combination of the two losses."""

        # Not ``Makes["MultiTaskStep"]``: the base already derives from
        # ``Makes["VisionTrainStep"]``, and the two give no consistent MRO.
        if TYPE_CHECKING:

            @override
            def make(self) -> MultiTaskStep:
                """Build the multi-task step."""
                ...

        model: Makeable[nn.Module] = field(default_factory=TwoHeadResNet.Config)
        """Network returning the fine and the coarse logits, in that order."""

        mtl_loss: Makeable[nn.Module] = field(default_factory=SumLoss.Config)
        """Combines ``(loss_fine, loss_coarse, epoch, total_epochs)`` into one scalar.

        Its parameters join the optimizer after the network's."""

    def __init__(self, config: Config) -> None:
        # Built before the base builds the network: none of this package's
        # combinations draws from the generator, and the optimizer needs it.
        mtl_loss = config.mtl_loss.make()
        self.mtl_loss: nn.Module = mtl_loss
        self.optimized: Optimized
        """What the optimizer is built over; set by :meth:`build_optimizer`."""
        super().__init__(config)
        self.mtl_epochs = int(config.train_budget_epochs)

    @override
    def build_optimizer(self, model: nn.Module) -> OptimizerProtocol:
        """Build the optimizer over the network's parameters, then the combination's.

        Args:
          model: The placed network.

        Returns:
          optimizer: Over :attr:`optimized`, whose parameters are named
            ``model.*`` then ``mtl_loss.*``.

        """
        self.mtl_loss = self.mtl_loss.to(self.device)
        self.optimized = Optimized(model=model, mtl_loss=self.mtl_loss)
        return super().build_optimizer(self.optimized)

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one update: forward, both losses, their combination, backward, step.

        Args:
          **batch: Preprocessed batch with ``media``, ``label`` and
            ``label_coarse``.

        Returns:
          result: ``loss`` (the combined scalar) and ``model`` (both heads'
            logits, fine then coarse, joined on the last axis).

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        label_coarse = batch["label_coarse"]
        assert isinstance(label_coarse, Tensor)
        self._apply_schedule()
        self.optimizer.zero_grad(set_to_none=True)
        self.mtl_loss.train()
        logits, logits_coarse = cast(tuple[Tensor, Tensor], self(media=media))
        loss_fine = self.loss(logits, label=label, context=self._context(logits))[
            "loss"
        ]
        loss_coarse = self.loss(
            logits_coarse,
            label=label_coarse,
            context=self._context(logits_coarse),
        )["loss"]
        loss = cast(
            object,
            self.mtl_loss(
                loss_fine,
                loss_coarse,
                self.timer_epoch.global_count,
                self.mtl_epochs,
            ),
        )
        assert isinstance(loss, Tensor)
        loss.backward()
        with self.timer_step:
            self.optimizer.step()
        # Reported as the plain sum, whatever was backpropagated: PCGrad returns a
        # surrogate whose value is not a loss.
        return {
            "loss": (loss_fine + loss_coarse).detach(),
            "model": torch.cat([logits, logits_coarse], dim=-1).detach(),
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score a test batch with the fine head's plain cross-entropy.

        Args:
          **batch: Preprocessed batch with ``media`` and ``label``.

        Returns:
          result: ``loss`` (a scalar) and ``model`` (both heads' logits, fine
            then coarse, joined on the last axis).

        """
        media = batch["media"]
        assert isinstance(media, Tensor)
        label = batch["label"]
        assert isinstance(label, Tensor)
        logits, logits_coarse = cast(tuple[Tensor, Tensor], self.call_eval(media=media))
        return {
            "loss": functional.cross_entropy(logits, label),
            "model": torch.cat([logits, logits_coarse], dim=-1),
        }

    class StateDict(VisionTrainStep.StateDict):
        """The vision step's state and the combination's."""

        mtl_loss: NotRequired[dict[str, Tensor]]

    @override
    def state_dict(self) -> StateDict:
        """Return the vision step's state with the combination's parameters and buffers.

        Returns:
          state: The base entries plus ``"mtl_loss"``.

        """
        return {**super().state_dict(), "mtl_loss": self.mtl_loss.state_dict()}

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
        """Restore the vision step's state and the combination's.

        Args:
          state_dict: Checkpoint from :meth:`state_dict`.
          strict: Require exact key matches, the combination's entry included.
          load_optimizer: Restore the optimizer and EMA too.
          remap: Transforms the saved model dict before loading.

        Raises:
          KeyError: ``strict`` and the checkpoint holds no ``"mtl_loss"``.

        """
        state = cast(MultiTaskStep.StateDict, state_dict)
        if strict and "mtl_loss" not in state:
            raise KeyError("The checkpoint holds no 'mtl_loss' state.")
        super().load_state_dict(
            state_dict,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        if "mtl_loss" in state:
            self.mtl_loss.load_state_dict(state["mtl_loss"], strict=strict)


class Optimized(nn.Module):
    """The network and the combination, under the names their parameters carry."""

    def __init__(self, *, model: nn.Module, mtl_loss: nn.Module) -> None:
        super().__init__()
        # Registered in this order: the optimizer's parameter order is the
        # reference's ``list(model.parameters()) + list(mtl_loss.parameters())``.
        self.model = model
        self.mtl_loss = mtl_loss
