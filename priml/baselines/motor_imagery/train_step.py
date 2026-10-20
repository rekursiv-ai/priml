"""Reference mean-loss backward, norm clipping, and epoch-indexed schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, override

from configgle import Makes
from torch import Tensor

import torch

from priml.cost import Cost, set_cost, traffic
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepOutput


def _off_target_cost(
    *,
    dtype: torch.dtype | None,
    channels_out: int,
    weighted: bool,
    rescale: int,
) -> Cost:
    """Cost log-softmax and the dense detached target distribution of one row."""
    del weighted
    if channels_out == -1:
        raise ValueError(
            "cross_entropy_off_target is costed per row of channels_out classes; "
            "set channels_out on the loss config.",
        )
    c = channels_out
    return (
        traffic(
            "primal",
            "elementwise",
            elements=13 * c + 4,
            flops=5 * c + 1,
            dtype=dtype,
        )
        + traffic(
            "primal",
            "reduction",
            elements=3 * (c + 1),
            flops=3 * (c - 1),
            dtype=dtype,
        )
        + traffic("primal", "selection", elements=1, dtype=torch.int64)
        + traffic("primal", "selection", elements=1, dtype=dtype)
        + traffic(
            "adjoint",
            "elementwise",
            elements=8 * c + 1 + 2 * rescale,
            flops=4 * c + rescale,
            dtype=dtype,
        )
        + traffic(
            "adjoint",
            "reduction",
            elements=c + 1,
            flops=c - 1,
            dtype=dtype,
        )
    )


@set_cost(_off_target_cost)
def cross_entropy_off_target(
    input: Tensor,
    target: Tensor,
    *,
    smoothing: float = 0.0,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> Tensor:
    """Cross entropy allocating smoothing mass only to incorrect classes.

    Unlike torch's ``label_smoothing``, the correct class keeps ``1 - smoothing``
    rather than also receiving a share of the uniform mass.

    Args:
      input: Class logits, shaped ``[batch, classes]``.
      target: Integer class labels, shaped ``[batch]``.
      smoothing: Total probability assigned to the incorrect classes.
      reduction: Batch reduction; the default preserves each example.

    Returns:
      loss: One dense cross entropy per example, without a batch reduction.

    """
    log_probability = input.log_softmax(dim=-1)
    with torch.no_grad():
        probability = torch.zeros_like(log_probability)
        probability.fill_(smoothing / (input.shape[-1] - 1))
        probability.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    loss = (-probability * torch.where(probability != 0, log_probability, 0.0)).sum(
        dim=-1,
    )
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class MotorImageryStep(TrainStep):
    """Train one public-reference network through Priml's TrainStep contract."""

    class Config(Makes["MotorImageryStep"], TrainStep.Config, kw_only=True):
        """The reference step; its recipe is stated by the experiment."""

    def __init__(self, config: Config, *, model: torch.nn.Module | None = None) -> None:
        if config.accumulate_grad_batches != 1:
            raise ValueError("The reference requires accumulate_grad_batches=1.")
        super().__init__(config, model=model)
        if getattr(self.optimizer, "requires_closure", False):
            raise ValueError("The reference does not support a closure optimizer.")

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Perform the source's mean backward, norm clip, and optimizer step."""
        self.optimizer.zero_grad(set_to_none=True)
        result = self.train_loss(**batch)
        loss, logits = result["loss"], result["model"]
        loss.mean().backward()
        self.step()
        return {"loss": loss.detach(), "model": logits.detach()}

    @override
    def apply_learning_rate(self, *, schedule_step: int | None = None) -> None:
        """Index the absolute schedule by completed data passes."""
        super().apply_learning_rate(
            schedule_step=self.timer_epoch.global_count
            if schedule_step is None
            else schedule_step,
        )

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Compute the source's prepared training objective without updating."""
        media, label = _batch_tensors(batch)
        logits = cast(Tensor, self(media.to(dtype=torch.float32)))
        with self._autocast_scope():
            loss = self._loss_on_output(
                logits,
                **{**batch, "label": label.to(dtype=torch.long)},
            )["loss"]
        return {"loss": loss, "model": logits}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Evaluate the source's objective and logits without stochastic dropout."""
        media, label = _batch_tensors(batch)
        with torch.no_grad():
            logits = cast(Tensor, self.call_eval(media.to(dtype=torch.float32)))
            with self._autocast_scope():
                loss = self._loss_on_output(
                    logits,
                    **{**batch, "label": label.to(dtype=torch.long)},
                )["loss"]
        return {"loss": loss, "model": logits}


def _batch_tensors(batch: dict[str, object]) -> tuple[Tensor, Tensor]:
    media, label = batch["media"], batch["label"]
    assert isinstance(media, Tensor)
    assert isinstance(label, Tensor)
    return media, label
