"""Port-side runs the ``cv-classification-loss`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and each baseline's
edit of it, and these runs, on the same inputs under the same pins, compares
them bit for bit, and stores the script's values under ``testdata/``;
``reference_test.py`` replays these runs against them. The network, data,
schedule and optimizer are the shared recipe's, proven by
:mod:`~priml.baselines.mlsbench.dl.vision.reference`; this task adds the
losses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.classification_loss.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.train_step import LossContext
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.loss.custom_types import LossOutput


LOSSES: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "cross_entropy": exp000,
    "label_smoothing": exp001,
    "focal": exp002,
    "poly": exp003,
}
"""The experiment that carries each loss, so its own settings are compared."""


def loss_record() -> dict[str, Tensor]:
    """Run every loss on fixed logits, forward and backward.

    Returns:
      record: Each loss's value and its gradient with respect to the logits.

    """
    logits, label = reference.fixed_logits()
    record: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        for name, factory in LOSSES.items():
            leaf = logits.clone().requires_grad_()
            context = LossContext(num_classes=4, epoch=0, total_epochs=1)
            loss = factory().step.loss.make()(leaf, label=label, context=context)
            loss["loss"].backward()
            record[f"loss/{name}/value"] = loss["loss"].detach()
            record[f"loss/{name}/grad"] = cast(Tensor, leaf.grad)
    return record


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`LOSSES`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from, the logits of a
        fixed probe in evaluation mode afterwards, and what the loss was given.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    contexts: list[LossContext] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            LOSSES[name],
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        )
        step.loss = _ContextSpy(step.loss, contexts)
        for index in range(reference.STEPS):
            step.timer_epoch.global_count = index
            result = step.train_step(
                media=reference.fixed_media(index),
                label=reference.fixed_label(index),
            )
            losses.append(result["loss"])
            logits.append(result["model"])
            probe = step.call_eval(media=reference.fixed_media(reference.STEPS))
            assert isinstance(probe, Tensor)
            probes.append(probe)
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return {
        f"lockstep/{name}/loss": torch.stack(losses),
        f"lockstep/{name}/logits": torch.stack(logits),
        f"lockstep/{name}/probe": torch.stack(probes),
        f"lockstep/{name}/context": torch.tensor(
            [[c.num_classes, c.epoch, c.total_epochs] for c in contexts],
        ),
    }


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = loss_record()
    for name in LOSSES:
        record |= lockstep_record(name)
    return record


class _ContextSpy:
    """Forwards to a loss and keeps what it was told about the run."""

    def __init__(
        self,
        loss: Callable[..., LossOutput],
        contexts: list[LossContext],
    ) -> None:
        self.loss = loss
        self.contexts = contexts

    def __call__(
        self,
        logits: Tensor,
        *,
        label: Tensor,
        context: LossContext,
    ) -> LossOutput:
        """Record ``context`` and return the wrapped loss."""
        self.contexts.append(context)
        return self.loss(logits, label=label, context=context)
