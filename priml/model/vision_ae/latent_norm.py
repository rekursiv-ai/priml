"""Latent normalizers: raw autoencoder latents to the space a diffusion model sees.

One class per published convention, each in its reference's exact operation
order. They are NOT folded into one precomputed scale and shift: multiplying by
a reciprocal rounds differently from the references' divides, so a merged form
would move the last bits every golden pins.

A normalizer is separate from the autoencoder (which never applies it) and from
a storage codec (which never sees it): a corpus stores raw latents, so changing
normalization never requires re-encoding.

Normalizers are plain objects, not modules: they hold no parameters and run on
data rather than in a model slot. Statistics stay on the CPU and move to the
latent's device per call, as the RAE reference moves them.
"""

from __future__ import annotations

from typing import cast

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.model.vision_ae.custom_types import CheckpointFile


class ScaleLatents:
    """Multiply by one scalar: ``z * scale`` and back by ``z / scale``."""

    class Config(Fig["ScaleLatents"]):
        """The scalar."""

        scale: float = 1.0
        """Multiplier into diffusion space; INVAE publishes 0.3099."""

    def __init__(self, config: Config) -> None:
        if config.scale == 0:
            raise ValueError("ScaleLatents needs a nonzero scale.")
        self.scale = config.scale

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * scale``.

        Args:
          latent: Raw latent.

        Returns:
          normalized: Diffusion-space latent, in ``latent``'s dtype.

        """
        return latent * self.scale

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent / scale``.

        Args:
          latent: Diffusion-space latent.

        Returns:
          raw: Raw latent, in ``latent``'s dtype.

        """
        return latent / self.scale


class ElementwiseLatentStats:
    """Standardize each ``[C, H, W]`` element: ``(z - mean) / sqrt(var + eps)``.

    RAE's convention, with statistics estimated over the training images and
    stored as ``{"mean": Tensor | None, "var": Tensor}``. A missing mean is
    skipped, which is bit-identical to the reference's subtracting 0. A missing
    var is refused: the reference's ``else 1`` reaches ``torch.sqrt`` as a
    float, which raises.

    References:
      https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/rae.py

    """

    class Config(Fig["ElementwiseLatentStats"]):
        """Where the statistics live."""

        stats: Makeable[CheckpointFile] | None = None
        """File holding ``var`` and, optionally, ``mean``."""

        eps: float = 1e-5
        """Added to the variance before the square root."""

    def __init__(self, config: Config) -> None:
        if config.stats is None:
            raise ValueError("ElementwiseLatentStats needs a stats file.")
        payload = cast(
            "dict[str, Tensor | None]",
            torch.load(
                config.stats.make().path(),
                map_location="cpu",
                weights_only=True,
            ),
        )
        var = payload.get("var")
        if var is None:
            raise ValueError("ElementwiseLatentStats needs a var in its stats file.")
        self.mean = payload.get("mean")
        self.var = var
        self.eps = config.eps

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / sqrt(var + eps)``.

        Args:
          latent: ``[B, C, H, W]`` raw latent.

        Returns:
          normalized: Standardized latent.

        """
        if self.mean is not None:
            latent = latent - self.mean.to(latent.device)
        return latent / torch.sqrt(self.var.to(latent.device) + self.eps)

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * sqrt(var + eps) + mean``.

        Args:
          latent: ``[B, C, H, W]`` standardized latent.

        Returns:
          raw: Raw latent.

        """
        latent = latent * torch.sqrt(self.var.to(latent.device) + self.eps)
        if self.mean is not None:
            latent = latent + self.mean.to(latent.device)
        return latent


class ChannelLatentStats:
    """Standardize per channel, then scale: ``(z - mean) / std * multiplier``.

    VTP's convention, applied by its LightningDiT training data and inverted as
    ``z * std / multiplier + mean`` before decoding. Statistics are stored as
    ``{"mean": [1, C, 1, 1], "std": [1, C, 1, 1]}``.

    References:
      https://github.com/JingfengYao/LightningDiT/blob/986098540d7d902cfee84e5344804de0eeeb2aaa/datasets/img_latent_dataset.py

    """

    class Config(Fig["ChannelLatentStats"]):
        """Where the statistics live, and the multiplier after them."""

        stats: Makeable[CheckpointFile] | None = None
        """File holding ``mean`` and ``std``."""

        multiplier: float = 1.0
        """Applied after standardizing; VTP's configs set 1.0."""

    def __init__(self, config: Config) -> None:
        if config.stats is None:
            raise ValueError("ChannelLatentStats needs a stats file.")
        payload = cast(
            "dict[str, Tensor]",
            torch.load(
                config.stats.make().path(),
                map_location="cpu",
                weights_only=True,
            ),
        )
        self.mean = payload["mean"]
        self.std = payload["std"]
        self.multiplier = config.multiplier

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / std * multiplier``.

        Args:
          latent: ``[B, C, H, W]`` raw latent.

        Returns:
          normalized: Standardized, scaled latent.

        """
        mean, std = self.mean.to(latent.device), self.std.to(latent.device)
        return (latent - mean) / std * self.multiplier

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * std / multiplier + mean``.

        Args:
          latent: ``[B, C, H, W]`` diffusion-space latent.

        Returns:
          raw: Raw latent.

        """
        mean, std = self.mean.to(latent.device), self.std.to(latent.device)
        return (latent * std) / self.multiplier + mean
