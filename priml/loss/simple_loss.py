"""Simple wrapper for PyTorch functional losses."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Literal, cast

import functools

from configgle import Fig
from torch import Tensor
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    reduction_cost,
    set_cost,
    traffic,
)
from priml.loss.custom_types import SimpleLossFn


if TYPE_CHECKING:
    from priml.loss.custom_types import LossOutput


# Every loss here answers ``cost`` with what :class:`SimpleLoss.Config.cost`
# knows -- the dtype, the class width, whether BCE carries a ``weight``, and
# whether a ``mean`` reduction scales the adjoint -- per token, before reduction.
#
# Torch's stable ``(1 - y) x - logsigmoid(x)`` with ``logsigmoid(x) = min(x, 0) -
# log1p(exp(-|x|))``: eight ops forward; the adjoint ``(1 - y) - sigmoid(x)`` times
# the upstream gradient is five. A ``weight`` is one more multiply each way.
def _bce_with_logits_cost(
    *,
    dtype: torch.dtype | None,
    channels_out: int,
    weighted: bool,
    rescale: int,
) -> Cost:
    del channels_out
    w = int(weighted)
    return traffic(
        "primal",
        "elementwise",
        elements=21 + 3 * w,
        flops=8 + w,
        dtype=dtype,
    ) + traffic(
        "adjoint",
        "elementwise",
        elements=10 + 2 * rescale + 3 * w,
        flops=5 + rescale + w,
        dtype=dtype,
    )


# A log-softmax over the row (subtract the max, exp, log the sum, subtract; the
# max and the sum are reductions) plus a negated gather of the label's entry. The
# adjoint exponentiates the saved log-softmax, scatter-adds ``-1`` at the label,
# and scales by the upstream gradient. The label is read forward and back, once
# as an index each way; the gathered logit and the scattered -1 are payload.
def _cross_entropy_cost(
    *,
    dtype: torch.dtype | None,
    channels_out: int,
    weighted: bool,
    rescale: int,
) -> Cost:
    del weighted
    if channels_out == -1:
        raise ValueError(
            "cross_entropy is costed per row of channels_out classes; "
            "set channels_out on the loss config.",
        )
    c = channels_out
    index = torch.int64
    return (
        traffic(
            "primal",
            "elementwise",
            elements=6 * c + 6,
            flops=3 * c + 2,
            dtype=dtype,
        )
        + traffic(
            "primal",
            "reduction",
            elements=2 * (c + 1),
            flops=2 * (c - 1),
            dtype=dtype,
        )
        + traffic("primal", "selection", elements=1, dtype=index)
        + traffic("primal", "selection", elements=2, dtype=dtype)
        + traffic(
            "adjoint",
            "elementwise",
            elements=4 * c + 1 + 2 * rescale,
            flops=2 * c + rescale,
            dtype=dtype,
        )
        + traffic("adjoint", "selection", elements=1, dtype=index)
        + traffic("adjoint", "selection", elements=3, flops=1, dtype=dtype)
    )


# Subtract, then square or take the magnitude; the adjoint scales the saved
# difference by the upstream gradient.
def _regression_cost(
    *,
    dtype: torch.dtype | None,
    channels_out: int,
    weighted: bool,
    rescale: int,
) -> Cost:
    del channels_out, weighted
    return traffic("primal", "elementwise", elements=5, flops=2, dtype=dtype) + traffic(
        "adjoint",
        "elementwise",
        elements=5 + 2 * rescale,
        flops=2 + rescale,
        dtype=dtype,
    )


@set_cost(_bce_with_logits_cost)
def bce_with_logits(
    input: Tensor,
    target: Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
) -> Tensor:
    """Apply ``binary_cross_entropy_with_logits``; costed per logit.

    Args:
      input: Logits.
      target: Targets of the same shape.
      reduction: Torch's reduction over every element.
      weight: Per-element rescaling, broadcast to ``input``.
      pos_weight: Per-class weight on the positive examples.

    Returns:
      loss: As torch returns it.

    """
    return functional.binary_cross_entropy_with_logits(
        input,
        target,
        weight=weight,
        reduction=reduction,
        pos_weight=pos_weight,
    )


@set_cost(_cross_entropy_cost)
def cross_entropy(
    input: Tensor,
    target: Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
    weight: Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Apply ``cross_entropy``; costed per row of ``channels_out`` logits.

    Args:
      input: Logits, classes on the channel axis.
      target: Class indices.
      reduction: Torch's reduction over every row.
      weight: Per-class rescaling.
      ignore_index: Target value that contributes nothing.
      label_smoothing: Mass moved from the label to the other classes.

    Returns:
      loss: As torch returns it.

    """
    return functional.cross_entropy(
        input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


@set_cost(_regression_cost)
def mse(
    input: Tensor,
    target: Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Apply ``mse_loss``; costed per element.

    Args:
      input: Predictions.
      target: Targets of the same shape.
      reduction: Torch's reduction over every element.

    Returns:
      loss: As torch returns it.

    """
    return functional.mse_loss(input, target, reduction=reduction)


@set_cost(_regression_cost)
def l1(
    input: Tensor,
    target: Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Apply ``l1_loss``; costed per element.

    Args:
      input: Predictions.
      target: Targets of the same shape.
      reduction: Torch's reduction over every element.

    Returns:
      loss: As torch returns it.

    """
    return functional.l1_loss(input, target, reduction=reduction)


class SimpleLoss:
    """Simple loss wrapper for PyTorch functional losses.

    Extracts target from batch kwargs and calls PyTorch loss function.
    Returns dict format for composability with other losses.

    Example:
      # Binary classification (default)
      cfg = SimpleLoss.Config()
      loss = cfg.make()

      # Multi-class classification
      cfg = SimpleLoss.Config(
          loss_fn=functional.cross_entropy,
          kwargs={"label_smoothing": 0.1},
      )
      loss = cfg.make()

      # In training
      prediction = model(**batch)
      result = loss(prediction, **batch)  # Returns {"loss": tensor}

    """

    class Config(Fig["SimpleLoss"]):
        """SimpleLoss configuration."""

        loss_fn: SimpleLossFn = bce_with_logits
        """Loss function, costed; the four in this module wrap torch's."""

        target_key: str = "label"
        """Batch key containing the target tensor."""

        kwargs: dict[str, object] = field(
            default_factory=lambda: {"reduction": "none"},
        )
        """Extra keyword arguments passed to loss_fn."""

        channels_out: int = -1
        """Classes per prediction row; read only by ``cross_entropy``, which
        is costed per row of this many logits. -1 leaves it uncosted."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the reduction this wrapper owns, then ask ``loss_fn`` for its own.

            A token is one element of the prediction -- one logit for
            :func:`bce_with_logits`, one value for :func:`mse` and :func:`l1`
            -- except :func:`cross_entropy`, where it is one ROW of
            ``channels_out`` class logits. The loss function costs that token
            itself (see each one's docstring).

            ``reduction="none"`` stops there. ``"mean"`` and ``"sum"`` add one
            reduction over the batch's tokens, ``(n - 1) / n`` per token;
            ``"mean"`` also scales the adjoint by ``1 / n``, one more op. The
            loss owns no parameters and issues no matmul. Labels are
            ``int64``; predictions and losses are at the batch's dtype.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this loss.

            Raises:
              TypeError: ``loss_fn`` carries no ``cost``.
              NotImplementedError: A non-default loss option lacks an analytical cost.
                BCE ``weight`` is costed as one extra tensor multiply each way.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = dtype
            channels_out = self.channels_out
            weighted = False
            # ``is`` against a protocol-typed slot: the checker sees no overlap
            # between the protocol and the concrete wrapper, so widen first.
            loss_fn = cast(object, self.loss_fn)
            for option, value in self.kwargs.items():
                if option == "reduction":
                    continue
                if (
                    option in ("weight", "pos_weight", "size_average", "reduce")
                    and value is None
                ):
                    continue
                if (
                    option == "label_smoothing"
                    and isinstance(value, (int, float))
                    and value == 0
                ):
                    continue
                if (
                    option == "ignore_index"
                    and isinstance(value, int)
                    and value == -100
                ):
                    continue
                if option == "weight" and loss_fn is bce_with_logits:
                    weighted = True
                    continue
                raise NotImplementedError(f"SimpleLoss.cost has no cost for {option}.")
            reduction = self.kwargs.get("reduction", "mean")
            reduced = Cost()
            if reduction != "none":
                reduced = reduction_cost(
                    input_elements=rows,
                    rows=rows,
                    dtype=dt,
                ) + traffic(
                    "adjoint",
                    "reduction",
                    elements=(rows + 1) / rows,
                    dtype=dt,
                )
            rescale = int(reduction == "mean")
            if rescale:
                reduced += traffic("primal", "elementwise", elements=2 / rows, dtype=dt)
            return reduced + cost(
                self.loss_fn,
                dtype=dt,
                channels_out=channels_out,
                weighted=weighted,
                rescale=rescale,
            )

    def __init__(self, config: Config) -> None:
        self.target_key = config.target_key
        self._loss_fn = functools.partial(config.loss_fn, **config.kwargs)

    def __call__(
        self,
        prediction: Tensor,
        **batch: object,
    ) -> LossOutput:
        """Compute loss from model prediction and batch.

        Args:
          prediction: Model output (logits, predictions, etc).
          **batch: Batch data containing target.

        Returns:
          result: Dict with 'loss' key.

        """
        if self.target_key not in batch:
            raise KeyError(
                f"Target key '{self.target_key}' not found in batch. "
                f"Available keys: {list(batch.keys())}",
            )

        target = batch[self.target_key]
        loss = self._loss_fn(prediction, target)
        return {"loss": loss}
