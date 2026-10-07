"""INVAE: the convolutional KL autoencoder SpeedrunDiT trains in the latent of.

The architecture is LDM's ``AutoencoderKL`` as the SpeedrunDiT reference ships
it; module and attribute names follow that checkpoint, so its ``state_dict``
loads unchanged. MIT license and attribution: ``priml/model/IN-VAE-LICENSE``.

References:
  https://github.com/SwayStar123/REG/blob/invae-sprint-rms-rope-valres-cfm-muon-layerwisescaling/models/invae.py
    SwayStar123. REG, ``models/invae.py``.

"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override

import math

from torch import Tensor, nn
from torch.nn import functional

import torch


if TYPE_CHECKING:
    from collections.abc import Sequence


class Upsample(nn.Module):
    """Double the spatial size by nearest-neighbor, then a 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(functional.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    """Halve the spatial size with a stride-2 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Padded on the bottom and right only: the checkpoint was trained with
        # this asymmetric padding, which ``Conv2d(padding=1)`` cannot express.
        return self.conv(functional.pad(x, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    """Two pre-activation GroupNorm-SiLU-conv layers around a residual.

    A 1x1 ``nin_shortcut`` projects the residual when the width changes.
    """

    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.norm2 = _group_norm(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(channels_in, channels_out, kernel_size=1)
            if channels_in != channels_out
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(functional.silu(self.norm1(x)))
        h = self.conv2(functional.silu(self.norm2(h)))
        return self.nin_shortcut(x) + h


class SpatialSelfAttention(nn.Module):
    """Single-head self-attention over every spatial position, plus a residual.

    The projections are 1x1 convolutions, as the checkpoint stores them.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        # [..., C, H, W] -> [..., H*W, C]: positions are the attended sequence.
        q, k, v = (
            projection(h).flatten(-2).transpose(-1, -2)
            for projection in (self.q, self.k, self.v)
        )
        attended = functional.scaled_dot_product_attention(q, k, v)
        return x + self.proj_out(attended.transpose(-1, -2).unflatten(-1, x.shape[-2:]))


class Encoder(nn.Module):
    """Downsample an image to the mean and log-variance of a latent."""

    def __init__(
        self,
        *,
        channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 1, 2, 2, 4),
        blocks_per_stage: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        channels_in: int = 3,
        resolution: int = 256,
        channels_latent: int = 16,
    ) -> None:
        """Build the encoder.

        Args:
          channels: Width of the first stage; each stage multiplies it.
          channel_multipliers: Per-stage width multipliers; one stage each.
          blocks_per_stage: Residual blocks in every stage.
          attention_resolutions: Spatial sizes whose blocks self-attend.
          channels_in: Image channels.
          resolution: Input spatial size the attention sizes refer to.
          channels_latent: Latent channels; the output holds twice as many,
            the mean and the log-variance.

        """
        super().__init__()
        self.conv_in = nn.Conv2d(channels_in, channels, kernel_size=3, padding=1)
        self.down: nn.ModuleList[_Stage] = nn.ModuleList()
        width_in = channels
        size = resolution
        for stage, multiplier in enumerate(channel_multipliers):
            width_out = channels * multiplier
            level = _Stage()
            for _ in range(blocks_per_stage):
                level.block.append(ResnetBlock(width_in, width_out))
                width_in = width_out
                if size in attention_resolutions:
                    level.attn.append(SpatialSelfAttention(width_in))
            if stage != len(channel_multipliers) - 1:
                level.downsample = Downsample(width_in)
                size //= 2
            self.down.append(level)
        self.mid = _Middle(width_in)
        self.norm_out = _group_norm(width_in)
        self.conv_out = nn.Conv2d(
            width_in,
            2 * channels_latent,
            kernel_size=3,
            padding=1,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for level in self.down:
            h = level(h)
        h = self.mid(h)
        return self.conv_out(functional.silu(self.norm_out(h)))


class Decoder(nn.Module):
    """Upsample a latent back to an image."""

    def __init__(
        self,
        *,
        channels: int = 128,
        channels_out: int = 3,
        channel_multipliers: Sequence[int] = (1, 1, 2, 2, 4),
        blocks_per_stage: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        resolution: int = 256,
        channels_latent: int = 16,
    ) -> None:
        """Build the decoder, the encoder's mirror with one extra block per stage.

        Args:
          channels: Width of the last stage; each stage multiplies it.
          channels_out: Image channels.
          channel_multipliers: Per-stage width multipliers, finest first.
          blocks_per_stage: The encoder's count; each stage holds one more.
          attention_resolutions: Spatial sizes whose blocks self-attend.
          resolution: Output spatial size the attention sizes refer to.
          channels_latent: Latent channels.

        """
        super().__init__()
        width_in = channels * channel_multipliers[-1]
        size = resolution // 2 ** (len(channel_multipliers) - 1)
        self.conv_in = nn.Conv2d(channels_latent, width_in, kernel_size=3, padding=1)
        self.mid = _Middle(width_in)
        stages: list[_Stage] = []
        for stage in reversed(range(len(channel_multipliers))):
            width_out = channels * channel_multipliers[stage]
            level = _Stage()
            for _ in range(blocks_per_stage + 1):
                level.block.append(ResnetBlock(width_in, width_out))
                width_in = width_out
                if size in attention_resolutions:
                    level.attn.append(SpatialSelfAttention(width_in))
            if stage != 0:
                level.upsample = Upsample(width_in)
                size *= 2
            stages.append(level)
        # Stored finest first, as the checkpoint indexes ``up``, and run coarsest first.
        self.up: nn.ModuleList[_Stage] = nn.ModuleList(reversed(stages))
        self.norm_out = _group_norm(width_in)
        self.conv_out = nn.Conv2d(width_in, channels_out, kernel_size=3, padding=1)

    @override
    def forward(self, z: Tensor) -> Tensor:
        h = self.mid(self.conv_in(z))
        for level in reversed(self.up):
            h = level(h)
        return self.conv_out(functional.silu(self.norm_out(h)))


class DiagonalGaussianDistribution:
    """A diagonal Gaussian over latents, from the encoder's moments."""

    def __init__(self, moments: Tensor) -> None:
        """Split ``moments`` into a mean and a clamped log-variance.

        Args:
          moments: Mean then log-variance, concatenated on the channel axis
            ``[..., 2 * C, H, W]``.

        """
        self.mean, logvar = moments.chunk(2, dim=-3)
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> Tensor:
        """Draw one latent by reparameterization."""
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other: DiagonalGaussianDistribution | None = None) -> Tensor:
        """Return KL(self || other) per sample; ``None`` is the unit Gaussian.

        Args:
          other: The distribution to diverge from.

        Returns:
          kl: Summed over channels and space, ``[...]``.

        """
        if other is None:
            divergence = self.mean.square() + self.var - 1.0 - self.logvar
        else:
            divergence = (
                (self.mean - other.mean).square() / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )
        return 0.5 * divergence.sum(dim=(-3, -2, -1))

    def nll(self, sample: Tensor, dim: Sequence[int] = (-3, -2, -1)) -> Tensor:
        """Return the negative log-likelihood of ``sample``, summed over ``dim``.

        Args:
          sample: Latents shaped like the mean.
          dim: Axes the per-element terms are summed over.

        Returns:
          nll: ``sample``'s shape with ``dim`` reduced.

        """
        return 0.5 * torch.sum(
            math.log(2.0 * math.pi)
            + self.logvar
            + (sample - self.mean).square() / self.var,
            dim=tuple(dim),
        )


class AutoencoderKL(nn.Module):
    """The KL autoencoder: encoder, posterior, decoder."""

    def __init__(
        self,
        *,
        channels_latent: int,
        channel_multipliers: Sequence[int],
    ) -> None:
        """Build the autoencoder.

        Args:
          channels_latent: Latent channels.
          channel_multipliers: Per-stage widths; each stage but the last halves
            the spatial size.

        """
        super().__init__()
        self.encoder = Encoder(
            channel_multipliers=channel_multipliers,
            channels_latent=channels_latent,
        )
        self.decoder = Decoder(
            channel_multipliers=channel_multipliers,
            channels_latent=channels_latent,
        )
        self.quant_conv = nn.Conv2d(2 * channels_latent, 2 * channels_latent, 1)
        self.post_quant_conv = nn.Conv2d(channels_latent, channels_latent, 1)

    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        """Return the posterior over latents for images ``x`` in [-1, 1]."""
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z: Tensor) -> Tensor:
        """Return images in [-1, 1] for latents ``z``."""
        return self.decoder(self.post_quant_conv(z))

    @override
    def forward(self, x: Tensor) -> tuple[DiagonalGaussianDistribution, Tensor]:
        """Encode, sample, and decode.

        Args:
          x: Images in [-1, 1].

        Returns:
          posterior: The encoder's distribution over latents.
          reconstruction: The decoded sample.

        """
        posterior = self.encode(x)
        return posterior, self.decode(posterior.sample())


def vae_f8d4() -> AutoencoderKL:
    """Return the 4-channel autoencoder downsampling 8x: ``[B, 4, 32, 32]`` at 256px."""
    return AutoencoderKL(channels_latent=4, channel_multipliers=(1, 2, 4, 4))


def vae_f16d32() -> AutoencoderKL:
    """Return the 32-channel autoencoder downsampling 16x: ``[B, 32, 16, 16]``."""
    return AutoencoderKL(channels_latent=32, channel_multipliers=(1, 1, 2, 2, 4))


@torch.no_grad()
def encode_image(vae: AutoencoderKL, image: Tensor) -> Tensor:
    """Sample unscaled INVAE latents from uint8 NCHW images."""
    return vae.encode(image.float() / 127.5 - 1).sample()


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    """Decode scaled model latents to float RGB in [0, 1]."""
    return ((vae.decode(latents / 0.3099) + 1) / 2).clamp(0, 1)


def load_invae(
    checkpoint: Path | str | None = None,
    *,
    device: torch.device | str = "cpu",
) -> AutoencoderKL:
    """Load the 32-channel INVAE from a local or REPA-E checkpoint.

    Args:
      checkpoint: Local checkpoint, or ``None`` to download the REPA-E weights.
      device: Device receiving the frozen model.

    Returns:
      vae: Frozen INVAE in evaluation mode.

    Raises:
      ImportError: If downloading is requested without Hugging Face Hub.
      TypeError: If Hugging Face Hub returns an invalid checkpoint path.

    """
    if checkpoint is None:
        try:
            hub = cast(_Hub, import_module("huggingface_hub"))
        except ImportError as error:
            raise ImportError(
                "Install priml[hub] or pass a local INVAE checkpoint path",
            ) from error
        downloaded = hub.hf_hub_download(
            repo_id="REPA-E/e2e-invae",
            filename="e2e-invae-400k.pt",
        )
        if not isinstance(downloaded, (str, Path)):
            raise TypeError("Hugging Face Hub returned an invalid checkpoint path.")
        checkpoint = downloaded
    vae = vae_f16d32()
    # torch.load is annotated `-> Any`; weights_only=True guarantees tensors.
    vae.load_state_dict(
        cast(
            dict[str, Tensor],
            torch.load(Path(checkpoint), map_location="cpu", weights_only=True),
        ),
    )
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def _group_norm(channels: int) -> nn.GroupNorm:
    """Return the checkpoint's 32-group normalization."""
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)


class _Stage(nn.Module):
    """One resolution's residual blocks, optional attention, and resampling.

    The attribute names are the checkpoint's: ``block``, ``attn``, and
    ``downsample`` or ``upsample``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block: nn.ModuleList[ResnetBlock] = nn.ModuleList()
        self.attn: nn.ModuleList[SpatialSelfAttention] = nn.ModuleList()
        self.downsample: Downsample | None = None
        self.upsample: Upsample | None = None

    @override
    def forward(self, h: Tensor) -> Tensor:
        for index, block in enumerate(self.block):
            h = block(h)
            if self.attn:
                h = self.attn[index](h)
        if self.downsample is not None:
            h = self.downsample(h)
        if self.upsample is not None:
            h = self.upsample(h)
        return h


class _Middle(nn.Module):
    """Residual block, self-attention, residual block, at the coarsest size."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = SpatialSelfAttention(channels)
        self.block_2 = ResnetBlock(channels, channels)

    @override
    def forward(self, h: Tensor) -> Tensor:
        return self.block_2(self.attn_1(self.block_1(h)))


class _Hub(Protocol):
    """The one ``huggingface_hub`` call the loader makes, imported only on demand."""

    def hf_hub_download(self, *, repo_id: str, filename: str) -> object: ...
