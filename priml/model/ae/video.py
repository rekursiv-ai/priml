"""Causal video KL autoencoders: the model around a family's halves, and their shared layers.

A family -- ``wan``, ``mochi`` -- supplies an encoder and a decoder;
:class:`CausalVideoVAE` maps uint8 clips to the posterior and latents back to
RGB around them. Every convolution is causal in time, so a clip is ``1 + k``
times the time factor in frames, and its first frame encodes as the image it is.

Apache-2.0 license and attribution: ``LICENSES/Wan2.1``, ``LICENSES/Wan2.2``,
``LICENSES/mochi``.

References:
  https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py
  https://github.com/genmoai/mochi/blob/15c9bad662df1fa3e7b53136423695b286484c08/src/genmo/mochi_preview/vae/models.py

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Protocol, Self, cast, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, traffic
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.frozen_cost import kl_round_trip_cost
from priml.model.ae.latent_norm import LatentNormalizer, ScaleLatents
from priml.model.conv import Conv3d
from priml.model.custom_types import ChannelsIn, ChannelsOut


class LatentFactors(Protocol):
    """A half's latent width and its resampling between pixels and latent."""

    channels_latent: int

    def factors(self) -> tuple[int, int]:
        """Return the ``(time, space)`` downsampling from pixels to latent."""
        ...


class VideoEncoderConfig(Makeable[nn.Module], ChannelsIn, LatentFactors, Protocol):
    """Builds ``[N, C_in, T, H, W]`` pixels in ``[-1, 1]`` to ``[N, 2 C, t, h, w]`` moments."""


class VideoDecoderConfig(Makeable[nn.Module], ChannelsOut, LatentFactors, Protocol):
    """Builds ``[N, C, t, h, w]`` latents to ``[N, C_out, T, H, W]`` pixels in ``[-1, 1]``."""


class CausalVideoVAE(nn.Module):
    """A causal video KL autoencoder as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a diagonal Gaussian; call ``sample()`` or ``mode()`` on
    it. The model is frozen and always in evaluation mode.
    """

    class Config(Fig["CausalVideoVAE"]):
        """The halves, the clip geometry, and the published latent normalizer."""

        encoder: VideoEncoderConfig | None = None
        """Pixels to posterior; ``None`` builds and loads no encoder."""

        decoder: VideoDecoderConfig | None = None
        """Latents to pixels; ``None`` builds and loads no decoder."""

        num_frames: int = 1
        """Frames in the clip the latent shape and the cost are stated at."""

        image_size: tuple[int, int] = (256, 256)
        """``(height, width)`` of those frames."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=ScaleLatents.Config,
        )
        """The statistics the family's diffusion models train with; the autoencoder never applies them."""

        def factors(self) -> tuple[int, int]:
            """Return the ``(time, space)`` downsampling from pixels to latent."""
            half = self.encoder or self.decoder
            if half is None:
                raise ValueError("CausalVideoVAE.Config needs an encoder or a decoder.")
            return half.factors()

        def latent_shape(self) -> tuple[int, int, int, int]:
            """Return ``(channels_latent, frames, height, width)`` of the clip's latent."""
            time, space = self.factors()
            half = self.encoder or self.decoder
            assert half is not None
            height, width = self.image_size
            frames = 1 + (self.num_frames - 1) // time
            return half.channels_latent, frames, height // space, width // space

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one ``decode(encode(clip).sample())`` round trip of the stated clip.

            Forward only; the parameters are still owned.

            Args:
              batch_size: Clips in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to both halves.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            if self.encoder is None or self.decoder is None:
                raise ValueError(
                    "CausalVideoVAE.Config.cost prices both halves; one is None.",
                )
            channels, frames, height, width = self.latent_shape()
            clip = (self.num_frames, *self.image_size)
            pixels_in = batch_size * self.encoder.channels_in * math.prod(clip)
            pixels_out = batch_size * self.decoder.channels_out * math.prod(clip)
            latents = batch_size * channels * frames * height * width
            full = (
                cost(
                    self.encoder,
                    **{
                        **kwargs,
                        "input_grid": clip,
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                + cost(
                    self.decoder,
                    **{
                        **kwargs,
                        "input_grid": (frames, height, width),
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                + kl_round_trip_cost(
                    pixels_in=pixels_in,
                    latents=latents,
                    pixels_out=pixels_out,
                    dtype=dtype,
                )
            )
            return Cost(
                cells=full.only("primal").cells,
                params=full.params,
                params_active=full.params_active,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        halves = [half for half in (config.encoder, config.decoder) if half is not None]
        if not halves:
            raise ValueError("CausalVideoVAE needs an encoder or a decoder.")
        if len({half.channels_latent for half in halves}) != 1:
            raise ValueError("The encoder and decoder disagree on channels_latent.")
        if len({half.factors() for half in halves}) != 1:
            raise ValueError(
                "The encoder and decoder disagree on their (time, space) factors, so "
                "the decoder would not return the clip.",
            )
        self.time, self.space = config.factors()
        height, width = config.image_size
        if (
            (config.num_frames - 1) % self.time
            or height % self.space
            or width % self.space
        ):
            raise ValueError(
                f"num_frames {config.num_frames} must be 1 + {self.time}k and "
                f"image_size {config.image_size} divisible by {self.space}.",
            )
        self.encoder = None if config.encoder is None else config.encoder.make()
        self.decoder = None if config.decoder is None else config.decoder.make()
        self.eval()
        self.requires_grad_(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, x: Tensor, /) -> DiagonalGaussian:
        """Return the posterior over latents.

        Args:
          x: ``[..., 3, T, H, W]`` uint8 RGB, ``T = 1 + time k`` frames.

        Returns:
          posterior: Diagonal Gaussian over ``[..., channels_latent, t, h, w]``.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder, or the clip's
            frames or sides do not divide.

        """
        if self.encoder is None:
            raise ValueError("This CausalVideoVAE was built without an encoder.")
        if x.dtype != torch.uint8:
            raise TypeError(
                f"CausalVideoVAE encodes uint8 RGB in [0, 255]; got {x.dtype}.",
            )
        frames, height, width = x.shape[-3:]
        if (frames - 1) % self.time or height % self.space or width % self.space:
            raise ValueError(
                f"CausalVideoVAE encodes 1 + {self.time}k frames of sides divisible by "
                f"{self.space}; got {frames} frames of {height}x{width}.",
            )
        # ``x / 127.5 - 1``, the references' own map, as ``kl`` maps uint8 pixels.
        pixels = x.reshape(-1, *x.shape[-4:]).float() / 127.5 - 1
        moments = cast(Tensor, self.encoder(pixels))
        moments = moments.reshape(*x.shape[:-4], *moments.shape[1:])
        mean, logvar = moments.chunk(2, dim=-4)
        return DiagonalGaussian(mean, logvar)

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels_latent, t, h, w]`` raw latents.

        Returns:
          video: ``[..., 3, 1 + time (t - 1), H, W]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder.

        """
        if self.decoder is None:
            raise ValueError("This CausalVideoVAE was built without a decoder.")
        decoded = cast(Tensor, self.decoder(z.reshape(-1, *z.shape[-4:])))
        decoded = decoded.reshape(*z.shape[:-4], *decoded.shape[1:])
        return ((decoded + 1) / 2).clamp(0, 1)


class CausalConv3d(Conv3d):
    """A ``Conv3d`` padded in time on its past side only.

    Time is padded by ``past`` frames before the first; each side by
    ``(kernel - 1) // 2``, both ways, so a stride-1 convolution keeps the grid.
    """

    class Config(Makes["CausalConv3d"], Conv3d.Config, kw_only=False):
        """``Conv3d`` settings, its own padding replaced by the causal pad."""

        _: KW_ONLY

        kernel_size: int | tuple[int, int, int] = (3, 3, 3)
        """``(time, height, width)`` extents."""

        padding: int | tuple[int, int, int] | str = 0
        """``Conv3d``'s own; the causal pad replaces it, so only 0 builds."""

        bias: bool = True
        """Include bias in the convolution."""

        past: int = 2
        """Frames padded before the first."""

        padding_mode: str = "constant"
        """``functional.pad``'s mode: ``"constant"`` pads zeros, ``"replicate"`` the edge."""

        def causal_padding(self) -> tuple[int, int, int, int, int, int]:
            """Return the pad ``functional.pad`` takes: width, height, then time."""
            kernel = self.kernel_size
            _, height, width = (kernel,) * 3 if isinstance(kernel, int) else kernel
            height, width = (height - 1) // 2, (width - 1) // 2
            return (width, width, height, height, self.past, 0)

        @override
        def cost(
            self,
            *,
            input_grid: int | tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the pad's copy, then the convolution on the padded grid.

            Args:
              input_grid: ``(frames, height, width)`` in; a scalar broadcasts.
              batch_size: Clips in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to ``Conv3d``'s cost.

            Returns:
              cost: Forward and adjoint.

            """
            t, h, w = (input_grid,) * 3 if isinstance(input_grid, int) else input_grid
            width, _, height, _, past, _ = self.causal_padding()
            padded = (t + past, h + 2 * height, w + 2 * width)
            moved = batch_size * self.channels_in * (t * h * w + math.prod(padded))
            return traffic(
                "primal",
                kernel="elementwise",
                elements=moved,
                dtype=dtype,
            ) + super().cost(
                input_grid=padded,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        if config.padding != 0:
            raise ValueError(
                f"CausalConv3d pads causally itself; padding must be 0, got "
                f"{config.padding!r}.",
            )
        super().__init__(config)
        self.causal = config.causal_padding()
        # Not ``padding_mode``: that is ``Conv3d``'s own, and anything but "zeros"
        # makes it pad again.
        self.pad_mode = config.padding_mode

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(
            functional.pad(input, self.causal, mode=self.pad_mode),
        )
