"""GAN loss functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Fig
from torch import nn

import torch

from priml.loss.custom_types import LossOutput


if TYPE_CHECKING:
    from torch import Tensor


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
        ).mean(dim=1)  # [B, 1] -> [B]

        # Content loss: L1 reconstruction (pointwise, flatten all dims except batch)
        content_loss = (
            nn.functional.l1_loss(fake_media, real_media, reduction="none")
            .flatten(1)
            .mean(dim=1)
        )  # [B, ...] -> [B]

        loss = self.adversarial_weight * adv_loss + self.content_weight * content_loss
        return {"loss": loss}
