"""Simple wrapper for PyTorch functional losses."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import functools

from configgle import Fig
from torch.nn import functional

from priml.loss.custom_types import LossOutput, SimpleLossFn
from priml.model.cost import Bytes, Compute, Cost, Flops, elementwise_cost


if TYPE_CHECKING:
    from torch import Tensor


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

        def cost(
            self,
            *,
            num_tokens: int,
            channels_out: int = -1,
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
            reduction over the ``num_tokens`` tokens, ``(n - 1) / n`` per token;
            ``"mean"`` also scales the adjoint by ``1 / n``, one more op. The
            loss owns no parameters and issues no matmul.

            Args:
              num_tokens: Tokens the reduction spans.
              channels_out: Classes per row; read only by ``cross_entropy``.
              **kwargs: The rest of the bus, unread.

            Returns:
              cost: Per-token cost of this loss.

            Raises:
              TypeError: ``loss_fn`` is not one of the four priced above.
              ValueError: ``cross_entropy`` without ``channels_out``.

            """
            del kwargs
            reduction = self.kwargs.get("reduction", "mean")
            reduced = Cost(
                primal=Compute(
                    flops=Flops(reduction=(num_tokens - 1) / num_tokens),
                ),
            )
            if reduction == "none":
                reduced = Cost()
            rescale = int(reduction == "mean")
            if self.loss_fn is functional.binary_cross_entropy_with_logits:
                return elementwise_cost(primal=8, adjoint=5 + rescale) + reduced
            if self.loss_fn is functional.cross_entropy:
                if channels_out == -1:
                    raise ValueError(
                        "cross_entropy is priced per row of channels_out classes; "
                        "put channels_out on the bus.",
                    )
                return (
                    Cost(
                        primal=Compute(
                            flops=Flops(
                                elementwise=3 * channels_out + 2,
                                reduction=2 * (channels_out - 1),
                            ),
                            bytes=Bytes(selection=1),
                        ),
                        adjoint=Compute(
                            flops=Flops(
                                elementwise=2 * channels_out + rescale,
                                selection=1,
                            ),
                        ),
                    )
                    + reduced
                )
            if (
                self.loss_fn is functional.mse_loss
                or self.loss_fn is functional.l1_loss
            ):
                return elementwise_cost(primal=2, adjoint=2 + rescale) + reduced
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
