"""Simple wrapper for PyTorch functional losses."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import functools

from configgle import Fig
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    reduction_cost,
    traffic,
)
from priml.loss.custom_types import SimpleLossFn


if TYPE_CHECKING:
    from torch import Tensor

    from priml.loss.custom_types import LossOutput


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

        loss_fn: SimpleLossFn = functional.binary_cross_entropy_with_logits
        """PyTorch loss function to use."""

        target_key: str = "label"
        """Batch key containing the target tensor."""

        kwargs: dict[str, object] = field(
            default_factory=lambda: {"reduction": "none"},
        )
        """Extra keyword arguments passed to loss_fn."""

        channels_out: int = -1
        """Classes per prediction row; read only by ``cross_entropy``, which
        is priced per row of this many logits. -1 leaves it unpriced."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price ``loss_fn`` by identity on one token of the prediction.

            A token is one element of the prediction -- one logit for
            ``binary_cross_entropy_with_logits``, one value for ``mse_loss`` and
            ``l1_loss`` -- except ``cross_entropy``, where it is one ROW of
            ``channels_out`` class logits. Per token:

            - BCE with logits, torch's stable ``(1 - y) x - logsigmoid(x)`` with
              ``logsigmoid(x) = min(x, 0) - log1p(exp(-|x|))``: eight ops
              forward; the adjoint ``(1 - y) - sigmoid(x)`` times the upstream
              gradient is five.
            - Cross-entropy: a log-softmax over the row (subtract the max, exp,
              log the sum, subtract; the max and the sum are reductions) plus a
              negated gather of the label's entry. The adjoint exponentiates
              the saved log-softmax, scatter-adds ``-1`` at the label, and
              scales by the upstream gradient.
            - MSE and L1: subtract, then square or take the magnitude; the
              adjoint scales the saved difference by the upstream gradient.

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
              TypeError: ``loss_fn`` is not one of the four priced above.
              ValueError: ``cross_entropy`` without ``channels_out``.
              NotImplementedError: A non-default loss option lacks an analytical price.
                BCE ``weight`` is priced as one extra tensor multiply each way.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = dtype
            index = torch.int64
            channels_out = self.channels_out
            weighted = False
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
                if (
                    option == "weight"
                    and self.loss_fn is functional.binary_cross_entropy_with_logits
                ):
                    weighted = True
                    continue
                raise NotImplementedError(f"SimpleLoss.cost has no price for {option}.")
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
            if self.loss_fn is functional.binary_cross_entropy_with_logits:
                w = int(weighted)
                return (
                    traffic(
                        "primal",
                        "elementwise",
                        elements=21 + 3 * w,
                        flops=8 + w,
                        dtype=dt,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=10 + 2 * rescale + 3 * w,
                        flops=5 + rescale + w,
                        dtype=dt,
                    )
                    + reduced
                )
            if self.loss_fn is functional.cross_entropy:
                if channels_out == -1:
                    raise ValueError(
                        "cross_entropy is priced per row of channels_out classes; "
                        "set channels_out on the loss config.",
                    )
                c = channels_out
                # The label is read forward and back, once as an index each
                # way; the gathered logit and the scattered -1 are payload.
                return (
                    traffic(
                        "primal",
                        "elementwise",
                        elements=6 * c + 6,
                        flops=3 * c + 2,
                        dtype=dt,
                    )
                    + traffic(
                        "primal",
                        "reduction",
                        elements=2 * (c + 1),
                        flops=2 * (c - 1),
                        dtype=dt,
                    )
                    + traffic("primal", "selection", elements=1, dtype=index)
                    + traffic("primal", "selection", elements=2, dtype=dt)
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=4 * c + 1 + 2 * rescale,
                        flops=2 * c + rescale,
                        dtype=dt,
                    )
                    + traffic("adjoint", "selection", elements=1, dtype=index)
                    + traffic("adjoint", "selection", elements=3, flops=1, dtype=dt)
                    + reduced
                )
            if (
                self.loss_fn is functional.mse_loss
                or self.loss_fn is functional.l1_loss
            ):
                return (
                    traffic("primal", "elementwise", elements=5, flops=2, dtype=dt)
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=5 + 2 * rescale,
                        flops=2 + rescale,
                        dtype=dt,
                    )
                    + reduced
                )
            raise TypeError(
                f"{getattr(self.loss_fn, '__qualname__', self.loss_fn)} has no "
                "price; SimpleLoss.cost knows binary_cross_entropy_with_logits, "
                "cross_entropy, mse_loss, and l1_loss.",
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
