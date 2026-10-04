"""Protocols a vision autoencoder, its posterior, and its latent statistics satisfy.

Deterministic and variational encoders share ONE ``encode``: it returns the
latent an experiment trains on, a plain tensor. What a variational encoder adds
is a capability, :meth:`VariationalAutoencoder.posterior`, which only code that
needs the distribution declares in its slot type -- so generic code never
branches on ``isinstance(code, Tensor)``. Whether ``encode`` samples the
posterior or takes its mode is a :data:`LatentFn` held by the config, where it
prints and diffs, rather than a call-site keyword that neither does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from configgle import Makeable
from torch import Tensor

import torch


if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class Autoencoder(Protocol):
    """Maps uint8 RGB images to raw latents, and raw latents back to pixels."""

    def encode(self, image: Tensor, /) -> Tensor:
        """Return the latent an experiment trains on.

        Args:
          image: ``[B, 3, H, W]`` uint8 RGB in ``[0, 255]``.

        Returns:
          latent: ``[B, C, h, w]`` raw (unnormalized) floating latent.

        """
        ...

    def decode(self, latent: Tensor, /) -> Tensor:
        """Return pixels for a raw latent.

        Args:
          latent: ``[B, C, h, w]`` raw latent, as :meth:`encode` returns it.

        Returns:
          image: ``[B, 3, H, W]`` float RGB, clamped to ``[0, 1]``.

        """
        ...


def require_uint8(image: Tensor) -> None:
    """Raise unless ``image`` meets ``Autoencoder.encode``'s uint8 contract.

    Every encoder scales by 255 itself, so a float image already in ``[0, 1]``
    would encode as near-black rather than fail.

    Args:
      image: The batch an encoder was handed.

    Raises:
      TypeError: ``image`` is not uint8.

    """
    if image.dtype != torch.uint8:
        raise TypeError(
            f"Autoencoders encode uint8 RGB in [0, 255]; got {image.dtype}.",
        )


@runtime_checkable
class Posterior(Protocol):
    """A distribution over latents, as much of one as a consumer needs."""

    def mode(self) -> Tensor:
        """Return the most likely latent."""
        ...

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Return a draw; ``None`` reads the global generator."""
        ...


@runtime_checkable
class VariationalAutoencoder(Autoencoder, Protocol):
    """An autoencoder whose encoder states a distribution, not a point."""

    def posterior(self, image: Tensor, /) -> Posterior:
        """Return the encoder's distribution for ``image`` (uint8, as ``encode``)."""
        ...


type LatentFn = Callable[[Posterior], Tensor]
"""Chooses the latent a variational encoder's ``encode`` returns."""


def posterior_sample(posterior: Posterior) -> Tensor:
    """Draw from the posterior with the global generator.

    Args:
      posterior: The encoder's distribution.

    Returns:
      latent: One draw.

    """
    return posterior.sample()


def posterior_mode(posterior: Posterior) -> Tensor:
    """Take the posterior's most likely latent.

    Args:
      posterior: The encoder's distribution.

    Returns:
      latent: The mode.

    """
    return posterior.mode()


@runtime_checkable
class LatentNormalizer(Protocol):
    """Moves latents between an autoencoder's raw space and a diffusion space."""

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Map a raw latent to the space a diffusion model trains in."""
        ...

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Map a diffusion-space latent back to the raw space."""
        ...


class VisionAutoencoderConfig(Makeable[Autoencoder], Protocol):
    """What a consumer reads from an autoencoder's config without building it.

    Building one downloads gigabytes, so the geometry and the statistics that
    belong to the checkpoint are stated on the config, where a dataset, a
    preparer, and a training loop can read them.
    """

    image_size: int
    """Side of the square image the checkpoint was trained at."""

    latent_norm: Makeable[LatentNormalizer]
    """The normalizer published with this checkpoint; the autoencoder never applies it."""

    def latent_shape(self) -> tuple[int, int, int]:
        """Return ``(channels, height, width)`` of a latent at ``image_size``."""
        ...


@runtime_checkable
class CheckpointFile(Protocol):
    """A weights file that resolves to a local path, downloading if it must."""

    def path(self) -> Path:
        """Return the local path, verified when a digest is configured."""
        ...

    def identity(self) -> dict[str, str]:
        """Return what names this file across machines, for a corpus receipt."""
        ...
