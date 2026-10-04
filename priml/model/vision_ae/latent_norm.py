"""Latent normalizers: raw autoencoder latents to the space a diffusion model sees.

One class per published convention, each in its reference's exact operation
order. They are NOT folded into one precomputed scale and shift: multiplying by
a reciprocal rounds differently from the references' divides, so a merged form
would move the last bits every golden pins.

A normalizer is separate from the autoencoder (which never applies it) and from
a storage codec (which never sees it): a corpus stores raw latents, so changing
normalization never requires re-encoding.

Normalizers are plain objects, not modules: they hold no parameters and run on
data rather than in a model slot. Statistics load on the CPU and are copied to
each device a latent arrives on once, then reused there.
"""

from __future__ import annotations

from typing import cast

import math

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
        scale = config.scale
        if math.isnan(scale) or math.isinf(scale) or scale == 0:
            raise ValueError("ScaleLatents needs a finite nonzero scale.")
        self.scale = scale

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
        eps = config.eps
        if math.isnan(eps) or math.isinf(eps) or eps < 0:
            raise ValueError("ElementwiseLatentStats needs a finite nonnegative eps.")
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
        self.eps = eps
        self._on_device: dict[torch.device, tuple[Tensor | None, Tensor]] = {}

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / sqrt(var + eps)``.

        Args:
          latent: ``[B, C, H, W]`` raw latent.

        Returns:
          normalized: Standardized latent.

        """
        mean, var = self._statistics(latent.device)
        if mean is not None:
            latent = latent - mean
        return latent / torch.sqrt(var + self.eps)

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * sqrt(var + eps) + mean``.

        Args:
          latent: ``[B, C, H, W]`` standardized latent.

        Returns:
          raw: Raw latent.

        """
        mean, var = self._statistics(latent.device)
        latent = latent * torch.sqrt(var + self.eps)
        if mean is not None:
            latent = latent + mean
        return latent

    def _statistics(self, device: torch.device) -> tuple[Tensor | None, Tensor]:
        """Return ``mean`` and ``var`` on ``device``, copying them there once."""
        # A host-to-device ``to`` from pageable memory blocks until the copy lands,
        # so a copy per call would stall the stream on every batch.
        if device not in self._on_device:
            mean = None if self.mean is None else self.mean.to(device)
            self._on_device[device] = mean, self.var.to(device)
        return self._on_device[device]


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
        multiplier = config.multiplier
        if math.isnan(multiplier) or math.isinf(multiplier) or multiplier == 0:
            raise ValueError("ChannelLatentStats needs a finite nonzero multiplier.")
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
        mean, std = payload["mean"], payload["std"]
        # Broadcasting would accept a bare ``[C]`` and standardize the width axis.
        if mean.shape != std.shape or mean.shape[:1] + mean.shape[2:] != (1, 1, 1):
            raise ValueError(
                "ChannelLatentStats needs mean and std of shape [1, C, 1, 1]; got "
                f"{list(mean.shape)} and {list(std.shape)}.",
            )
        self.mean = mean
        self.std = std
        self.multiplier = multiplier
        self._on_device: dict[torch.device, tuple[Tensor, Tensor]] = {}

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / std * multiplier``.

        Args:
          latent: ``[B, C, H, W]`` raw latent.

        Returns:
          normalized: Standardized, scaled latent.

        """
        mean, std = self._statistics(latent.device)
        return (latent - mean) / std * self.multiplier

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * std / multiplier + mean``.

        Args:
          latent: ``[B, C, H, W]`` diffusion-space latent.

        Returns:
          raw: Raw latent.

        """
        mean, std = self._statistics(latent.device)
        return (latent * std) / self.multiplier + mean

    def _statistics(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return ``mean`` and ``std`` on ``device``, copying them there once."""
        # As in ``ElementwiseLatentStats``: a copy per call would block every batch.
        if device not in self._on_device:
            self._on_device[device] = self.mean.to(device), self.std.to(device)
        return self._on_device[device]
