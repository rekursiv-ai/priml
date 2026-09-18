"""GAN loss functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Fig
from torch import nn

import torch

from priml.cost import (
    Cost,
    reduction_cost,
    traffic,
)


if TYPE_CHECKING:
    from torch import Tensor

    from priml.loss.custom_types import LossOutput


class AdversarialLoss:
    """Adversarial loss for GAN generator.

    Combines adversarial loss (fool discriminator) with content loss
    (L1 reconstruction).
    """

    class Config(Fig["AdversarialLoss"]):
        """Adversarial loss configuration."""

        adversarial_weight: float = 1.0
        """Weight for adversarial (fool discriminator) loss."""

        content_weight: float = 100.0
        """Weight for L1 content reconstruction loss."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one media element; per-sample scalar work is spread ``1 / n``.

            A token is one element of ``fake_media``; the geometry's rows are
            the elements one sample holds. Per element, the L1 term is a subtract
            and a magnitude, then the mean over the sample is one reduction of
            ``(n - 1) / n``; its adjoint scales the saved sign by the upstream
            gradient and by ``1 / n``, three ops. Per SAMPLE, and so divided by
            ``rows``: BCE with logits on the single ``[B, 1]`` logit
            (eight forward, five back, as :class:`SimpleLoss` costs it; the
            mean over a width of one reduces nothing) and the two weights
            (two multiplies and an add forward, two multiplies back).
            ``model_output`` is unread, and the discriminator that produced
            ``fake_logits`` is costed by its own config, not here.

            Traffic includes the generated label, both mean scales, and all
            logical intermediate reads/writes, even for the width-one mean.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-element cost of this loss.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = dtype
            return (
                traffic(
                    "primal",
                    "elementwise",
                    elements=5 + 33 / rows,
                    flops=2 + (8 + 3) / rows,
                    dtype=dt,
                )
                + reduction_cost(
                    input_elements=rows + 1,
                    output_groups=2,
                    rows=rows,
                    dtype=dt,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=7 + 16 / rows,
                    flops=3 + (5 + 2) / rows,
                    dtype=dt,
                )
                + traffic("adjoint", "reduction", elements=(rows + 3) / rows, dtype=dt)
            )

    def __init__(self, config: Config) -> None:
        """Initialize loss.

        Args:
          config: Loss configuration.

        """
        self.adversarial_weight = config.adversarial_weight
        self.content_weight = config.content_weight

    def __call__(
        self,
        model_output: Tensor,
        *,
        fake_logits: Tensor,
        fake_media: Tensor,
        real_media: Tensor,
        **batch: object,
    ) -> LossOutput:
        """Compute adversarial + content loss (pointwise).

        Args:
          model_output: Generator output (same as fake_media, ignored).
          fake_logits: Discriminator output on fake media [B, 1].
          fake_media: Generated media [B, ...].
          real_media: Real target media [B, ...].
          **batch: Additional batch keys (ignored).

        Returns:
          loss: Dict with pointwise loss [B].

        """
        del model_output, batch

        # Adversarial loss: fool discriminator (pointwise)
        adv_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits),
            reduction="none",
        ).mean(dim=1)  # [B, 1] -> [B].

        # Content loss: L1 reconstruction (pointwise, flatten all dims except batch)
        content_loss = (
            nn.functional.l1_loss(fake_media, real_media, reduction="none")
            .flatten(1)
            .mean(dim=1)
        )  # [B, ...] -> [B].

        loss = self.adversarial_weight * adv_loss + self.content_weight * content_loss
        return {"loss": loss}
