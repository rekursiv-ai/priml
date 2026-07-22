"""LPIPS perceptual loss for videos."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import Fig
from torch import Tensor, nn
from wrapt import lazy_import

import torch

from priml.loss.custom_types import LossOutput


lpips = lazy_import("lpips")


if TYPE_CHECKING:
    from typing import Any


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss for videos."""

    class Config(Fig["LPIPSLoss"]):
        max_num_random_frames: int = 2
        """Maximum number of random frames to subsample per video."""
        net: str = "vgg"
        """Backbone network for LPIPS ("vgg", "alex", "squeeze")."""

    def __init__(self, config: Config):
        super().__init__()
        self.max_num_random_frames = config.max_num_random_frames
        self.lpips_criterion = lpips.LPIPS(net=config.net)

    @override
    def forward(
        self,
        model_output: Tensor,
        *,
        x: Tensor,
        xhat: Tensor,
        **batch: Any,
    ) -> LossOutput:
        """Compute LPIPS on subsampled frames (pointwise).

        Args:
          model_output: Model output (ignored, use xhat instead).
          x: [B, C, T, H, W] video tensor.
          xhat: [B, C, T, H, W] reconstructed video tensor.
          **batch: Additional batch keys (ignored).

        Returns:
          loss: Dict with pointwise loss [B], one entry per input sample.

        """
        del model_output, batch
        assert x.ndim == 5, "Expected 5D tensor [B, C, T, H, W]"

        # Subsample unique random frames (at most max_num_random_frames or total frames)
        num_frames = x.shape[-3]
        num_sample_frames = min(self.max_num_random_frames, num_frames)
        rand_indices = torch.randperm(num_frames, device=x.device)[:num_sample_frames]

        x_sub = x[:, :, rand_indices, :, :]
        xhat_sub = xhat[:, :, rand_indices, :, :]

        # Reshape to [B*T, C, H, W] for LPIPS
        b, c, t, h, w = x_sub.shape
        x_sub = x_sub.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        xhat_sub = xhat_sub.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # LPIPS returns [B*T, 1, 1, 1]
        loss = self.lpips_criterion(x_sub, xhat_sub)

        # Reshape back to [B, T] and mean over frames → [B]
        loss = loss.reshape(b, t).mean(dim=1)

        return {"loss": loss}
