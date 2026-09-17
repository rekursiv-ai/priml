"""Simple wrapper for PyTorch functional losses."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import functools

from configgle import Fig
from torch.nn import functional

from priml.loss.custom_types import LossOutput, SimpleLossFn
from priml.model.cost import Bytes, Compute, Cost, Flops, reduction_cost


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
            rows: int,
            channels_out: int = -1,
            itemsize: int = 4,
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
            reduction over the ``rows`` tokens, ``(n - 1) / n`` per token;
            ``"mean"`` also scales the adjoint by ``1 / n``, one more op. The
            loss owns no parameters and issues no matmul.

            Args:
              rows: Tokens the reduction spans.
              channels_out: Classes per row; read only by ``cross_entropy``.
              itemsize: Bytes per logical tensor element, including labels.
              **kwargs: The rest of the bus, unread.

            Returns:
              cost: Per-token cost of this loss.

            Raises:
              TypeError: ``loss_fn`` is not one of the four priced above.
              ValueError: ``cross_entropy`` without ``channels_out``.
              NotImplementedError: A non-default loss option lacks an analytical price.
                BCE ``weight`` is priced as one extra tensor multiply each way.

            """
            del kwargs
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
                reduced = Cost(
                    primal=reduction_cost(
                        input_elements=rows,
                        rows=rows,
                        itemsize=itemsize,
                    ),
                    adjoint=Compute(
                        bytes=Bytes(reduction=itemsize * (rows + 1) / rows),
                    ),
                )
            rescale = int(reduction == "mean")
            if rescale:
                reduced += Cost(
                    primal=Compute(bytes=Bytes(elementwise=2 * itemsize / rows)),
                )
            if self.loss_fn is functional.binary_cross_entropy_with_logits:
                return (
                    Cost(
                        primal=Compute(
                            flops=Flops(elementwise=8 + int(weighted)),
                            bytes=Bytes(
                                elementwise=(21 + 3 * int(weighted)) * itemsize,
                            ),
                        ),
                        adjoint=Compute(
                            flops=Flops(elementwise=5 + rescale + int(weighted)),
                            bytes=Bytes(
                                elementwise=(10 + 2 * rescale + 3 * int(weighted))
                                * itemsize,
                            ),
                        ),
                    )
                    + reduced
                )
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
                            bytes=Bytes(
                                elementwise=(6 * channels_out + 6) * itemsize,
                                reduction=2 * (channels_out + 1) * itemsize,
                                selection=3 * itemsize,
                            ),
                        ),
                        adjoint=Compute(
                            flops=Flops(
                                elementwise=2 * channels_out + rescale,
                                selection=1,
                            ),
                            bytes=Bytes(
                                elementwise=(4 * channels_out + 1 + 2 * rescale)
                                * itemsize,
                                selection=4 * itemsize,
                            ),
                        ),
                    )
                    + reduced
                )
            if (
                self.loss_fn is functional.mse_loss
                or self.loss_fn is functional.l1_loss
            ):
                return (
                    Cost(
                        primal=Compute(
                            flops=Flops(elementwise=2),
                            bytes=Bytes(elementwise=5 * itemsize),
                        ),
                        adjoint=Compute(
                            flops=Flops(elementwise=2 + rescale),
                            bytes=Bytes(elementwise=(5 + 2 * rescale) * itemsize),
                        ),
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
