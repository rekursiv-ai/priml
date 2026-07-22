"""PCA whitening conv layer."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

import torch

from priml.math.stats import pca


class PCAWhiteningConv2d(nn.Conv2d):
    """Conv2d initialized with PCA whitening eigenvectors (frozen weights).

    Extracts image patches, computes PCA decomposition, and sets the
    conv weights to the whitened eigenvectors with rank doubling via
    [V, -V]. Weights are frozen after initialization.

    Typical usage::

        layer = PCAWhiteningConv2d(3, 48, kernel_size=3, padding=1, bias=False)
        layer.init_whiten(train_images, eps=5e-4)  # train_images: (N, C, H, W)

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight.requires_grad = False

    def init_whiten(
        self,
        train_images: Tensor,
        eps: float = 5e-4,
        algorithm: str = "eigh",
    ) -> None:
        """Initialize weights with PCA eigenvectors from image patches.

        Args:
          train_images: Training images of shape (N, C, H, W).
          eps: Regularization for whitening eigenvalues.
          algorithm: PCA algorithm ("eigh", "svd", or "power").

        """
        out_channels, C, kH, kW = self.weight.shape
        expected = 2 * C * kH * kW
        if out_channels != expected:
            raise ValueError(
                f"out_channels={out_channels} must equal 2 * in_channels * "
                f"kH * kW = {expected} for [V, -V] rank doubling.",
            )
        patches = (
            train_images.unfold(2, kH, 1)
            .unfold(3, kW, 1)
            .transpose(1, 3)
            .reshape(-1, C * kH * kW)
        )
        _, eigenvectors = pca(patches, whiten=True, eps=eps, algorithm=algorithm)
        kernel = eigenvectors.T.reshape(-1, C, kH, kW).to(self.weight.dtype)
        self.weight.data[:] = torch.cat([kernel, -kernel])
