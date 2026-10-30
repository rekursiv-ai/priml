"""Map raw autoencoder latents to and from the space a diffusion model trains in."""

from __future__ import annotations

from dataclasses import field
from typing import Protocol, cast, runtime_checkable

import math

from configgle import Fig, Makeable
from torch import Tensor
from torch.nn import functional

import torch

from priml.model.ae.checkpoint import CheckpointFile, read_state


@runtime_checkable
class LatentNormalizer(Protocol):
    """Moves latents between an autoencoder's raw space and a diffusion space."""

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Map a raw latent to the space a diffusion model trains in."""
        ...

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Map a diffusion-space latent back to the raw space."""
        ...


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


class ShiftScaleLatents:
    """Shift, then scale: ``(z - shift) * scale`` and back by ``z / scale + shift``.

    FLUX.1's convention.

    References:
      https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/modules/autoencoder.py
        License: ``LICENSES/flux``.

    """

    class Config(Fig["ShiftScaleLatents"]):
        """The two scalars."""

        shift: float = 0.0
        """Subtracted before scaling; FLUX.1 publishes 0.1159."""

        scale: float = 1.0
        """Multiplier into diffusion space; FLUX.1 publishes 0.3611."""

    def __init__(self, config: Config) -> None:
        shift, scale = config.shift, config.scale
        if not math.isfinite(shift):
            raise ValueError("ShiftScaleLatents needs a finite shift.")
        if not math.isfinite(scale) or scale == 0:
            raise ValueError("ShiftScaleLatents needs a finite nonzero scale.")
        self.shift = shift
        self.scale = scale

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - shift) * scale``.

        Args:
          latent: Raw latent.

        Returns:
          normalized: Diffusion-space latent, in ``latent``'s dtype.

        """
        return (latent - self.shift) * self.scale

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent / scale + shift``.

        Args:
          latent: Diffusion-space latent.

        Returns:
          raw: Raw latent, in ``latent``'s dtype.

        """
        return latent / self.scale + self.shift


class BatchNormLatents:
    """Standardize per channel by a frozen batch norm's running statistics.

    FLUX.2's convention: ``functional.batch_norm`` in evaluation mode, and back
    by ``z * sqrt(var + eps) + mean``, each in the reference's order. The
    statistics are the checkpoint's ``bn.running_mean`` and ``bn.running_var``.

    References:
      https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/autoencoder.py
        License: ``LICENSES/flux2``.

    """

    class Config(Fig["BatchNormLatents"]):
        """Where the statistics live, and the epsilon added to the variance."""

        stats: Makeable[CheckpointFile] | None = None
        """File holding ``bn.running_mean`` and ``bn.running_var``."""

        eps: float = 1e-5
        """Added to the variance; torch's default, FLUX.2 publishes ``1e-4``."""

    def __init__(self, config: Config) -> None:
        eps = config.eps
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("BatchNormLatents needs a finite nonnegative eps.")
        if config.stats is None:
            raise ValueError("BatchNormLatents needs a stats file.")
        state = read_state(config.stats.make().path(), prefixes=("bn.",))
        mean, var = state.get("bn.running_mean"), state.get("bn.running_var")
        if mean is None or var is None:
            raise ValueError(
                "BatchNormLatents needs bn.running_mean and bn.running_var in its "
                "stats file.",
            )
        if mean.ndim != 1 or mean.shape != var.shape:
            raise ValueError(
                "BatchNormLatents needs a running mean and variance of shape [C]; "
                f"got {list(mean.shape)} and {list(var.shape)}.",
            )
        self.mean = mean
        self.var = var
        self.eps = eps
        self._on_device: dict[torch.device, tuple[Tensor, Tensor]] = {}

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``batch_norm(latent)`` with the running statistics.

        Args:
          latent: ``[..., C, H, W]`` raw latent.

        Returns:
          normalized: Standardized latent.

        """
        mean, var = self._statistics(latent.device)
        flat = latent.reshape(-1, *latent.shape[-3:])
        normalized = functional.batch_norm(
            flat,
            mean,
            var,
            training=False,
            eps=self.eps,
        )
        return normalized.reshape(latent.shape)

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * sqrt(var + eps) + mean``.

        Args:
          latent: ``[..., C, H, W]`` standardized latent.

        Returns:
          raw: Raw latent.

        """
        mean, var = self._statistics(latent.device)
        std = torch.sqrt(var.view(-1, 1, 1) + self.eps)
        return latent * std + mean.view(-1, 1, 1)

    def _statistics(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return ``mean`` and ``var`` on ``device``, copying them there once."""
        # As in ``ElementwiseLatentStats``: a copy per call would block every batch.
        if device not in self._on_device:
            self._on_device[device] = self.mean.to(device), self.var.to(device)
        return self._on_device[device]


class InverseStdLatents:
    """Standardize video latents per channel through a reciprocal standard deviation.

    Wan's convention: ``(z - mean) * (1 / std)``, and back by
    ``z / (1 / std) + mean``, the reciprocal rounded once in float32, as the
    reference holds it. Latents are ``[..., C, T, H, W]``.

    References:
      https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py
        License: ``LICENSES/Wan2.1``.

    """

    class Config(Fig["InverseStdLatents"]):
        """The published statistics, one per latent channel."""

        mean: list[float] = field(default_factory=list[float])
        """Subtracted from each channel."""

        std: list[float] = field(default_factory=list[float])
        """Each channel's standard deviation; its reciprocal scales."""

    def __init__(self, config: Config) -> None:
        self.mean, std = _video_channel_stats(
            "InverseStdLatents",
            config.mean,
            config.std,
        )
        self.inv_std = 1.0 / std
        self._on_device: dict[torch.device, tuple[Tensor, Tensor]] = {}

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) * (1 / std)``.

        Args:
          latent: ``[..., C, T, H, W]`` raw latent.

        Returns:
          normalized: Standardized latent.

        """
        mean, inv_std = self._statistics(latent.device)
        return (latent - mean) * inv_std

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent / (1 / std) + mean``.

        Args:
          latent: ``[..., C, T, H, W]`` standardized latent.

        Returns:
          raw: Raw latent.

        """
        mean, inv_std = self._statistics(latent.device)
        return latent / inv_std + mean

    def _statistics(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return ``mean`` and ``1 / std`` on ``device``, copying them there once."""
        # As in ``ElementwiseLatentStats``: a copy per call would block every batch.
        if device not in self._on_device:
            self._on_device[device] = self.mean.to(device), self.inv_std.to(device)
        return self._on_device[device]


class StdLatents:
    """Standardize video latents per channel: ``(z - mean) / std``.

    Mochi's convention, back by ``z * std + mean``. Latents are
    ``[..., C, T, H, W]``.

    References:
      https://github.com/genmoai/mochi/blob/15c9bad662df1fa3e7b53136423695b286484c08/src/genmo/mochi_preview/vae/vae_stats.py
        License: ``LICENSES/mochi``.

    """

    class Config(Fig["StdLatents"]):
        """The published statistics, one per latent channel."""

        mean: list[float] = field(default_factory=list[float])
        """Subtracted from each channel."""

        std: list[float] = field(default_factory=list[float])
        """Each channel's standard deviation, divided out."""

    def __init__(self, config: Config) -> None:
        self.mean, self.std = _video_channel_stats(
            "StdLatents",
            config.mean,
            config.std,
        )
        self._on_device: dict[torch.device, tuple[Tensor, Tensor]] = {}

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / std``.

        Args:
          latent: ``[..., C, T, H, W]`` raw latent.

        Returns:
          normalized: Standardized latent.

        """
        mean, std = self._statistics(latent.device)
        return (latent - mean) / std

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * std + mean``.

        Args:
          latent: ``[..., C, T, H, W]`` standardized latent.

        Returns:
          raw: Raw latent.

        """
        mean, std = self._statistics(latent.device)
        return latent * std + mean

    def _statistics(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return ``mean`` and ``std`` on ``device``, copying them there once."""
        # As in ``ElementwiseLatentStats``: a copy per call would block every batch.
        if device not in self._on_device:
            self._on_device[device] = self.mean.to(device), self.std.to(device)
        return self._on_device[device]


def _video_channel_stats(
    name: str,
    mean: list[float],
    std: list[float],
) -> tuple[Tensor, Tensor]:
    """Return per-channel ``mean`` and ``std`` as ``[C, 1, 1, 1]`` float32, validated."""
    if not mean or len(mean) != len(std):
        raise ValueError(
            f"{name} needs one mean and one std per channel; got {len(mean)} and "
            f"{len(std)}.",
        )
    deviation = torch.tensor(std, dtype=torch.float32)
    if not torch.all(torch.isfinite(deviation) & (deviation != 0)):
        raise ValueError(f"{name} needs finite nonzero std.")
    shape = (-1, 1, 1, 1)
    return torch.tensor(mean, dtype=torch.float32).view(shape), deviation.view(shape)


class ElementwiseLatentStats:
    """Standardize each ``[C, H, W]`` element: ``(z - mean) / sqrt(var + eps)``.

    RAE's convention. The statistics file holds ``{"mean": Tensor | None,
    "var": Tensor}``; a ``None`` mean is not subtracted.

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
          latent: ``[..., C, H, W]`` raw latent.

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
          latent: ``[..., C, H, W]`` standardized latent.

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

    VTP's convention. The statistics file holds ``{"mean": [1, C, 1, 1],
    "std": [1, C, 1, 1]}``.

    References:
      https://github.com/JingfengYao/LightningDiT/blob/986098540d7d902cfee84e5344804de0eeeb2aaa/datasets/img_latent_dataset.py
        License: ``LICENSES/LightningDiT``.

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
          latent: ``[..., C, H, W]`` raw latent.

        Returns:
          normalized: Standardized, scaled latent.

        """
        mean, std = self._statistics(latent.device)
        return (latent - mean) / std * self.multiplier

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * std / multiplier + mean``.

        Args:
          latent: ``[..., C, H, W]`` diffusion-space latent.

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
