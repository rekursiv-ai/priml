"""INVAE: the 32-channel variational autoencoder SpeedrunDiT trains on.

The convolutional architecture is the reference implementation's, kept
verbatim so the published checkpoint's parameter names load unchanged; only the
width and normalization-group knobs were threaded through, at their published
defaults, so a test can build it small. :class:`INVAE` is the configured
wrapper every consumer uses.

MIT license and attribution: ``priml/model/vision_ae/IN-VAE-LICENSE``.

``scripts/reference_parity.py`` proves it bit-identical to the reference below.

References:
  https://github.com/SwayStar123/REG/blob/3c51606c801dd9e87ee9ef778782766ab7c379ca/models/invae.py
  https://huggingface.co/REPA-E/e2e-invae
    Leng et al. 2025. REPA-E: Unlocking VAE for end-to-end tuning with latent
    diffusion transformers.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol, Self, TypedDict, cast, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, elementwise_cost, traffic
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import conv_cost
from priml.model.norm import GroupNorm2d
from priml.model.swiglu import sigmoid
from priml.model.vision_ae.checkpoint import HubFile
from priml.model.vision_ae.custom_types import (
    CheckpointFile,
    LatentFn,
    LatentNormalizer,
    posterior_sample,
    require_uint8,
)
from priml.model.vision_ae.latent_norm import ScaleLatents


def nonlinearity(x: Tensor) -> Tensor:
    """Swish: ``x * sigmoid(x)``."""
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:  # noqa: N802 -- The reference's name, which reference_parity.py pairs by.
    """Build the reference's affine GroupNorm, epsilon 1e-6.

    Args:
      in_channels: Channels normalized.
      num_groups: Groups the channels are split into; must divide ``in_channels``.

    Returns:
      norm: The GroupNorm module.

    """
    return torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class Upsample(nn.Module):
    """Nearest-neighbour doubling, then optionally a 3x3 convolution."""

    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Halve the grid: a one-sided pad and stride-2 convolution, or average pooling."""

    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # No asymmetric padding in torch conv, must do it ourselves.
            self.conv = torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """Two GroupNorm, swish, convolution layers added back to a projected input."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, num_groups=num_groups)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    @override
    def forward(self, x: Tensor, temb: Tensor | None) -> Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention over the grid, added back to its input."""

    def __init__(self, in_channels: int, num_groups: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, num_groups=num_groups)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c.
        k = k.reshape(b, c, h * w)  # b,c,hw.
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j].
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # `b`, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j].
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    """Pixels to posterior moments: residual stages halving the grid, then a middle."""

    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 16,
        double_z: bool = True,
        num_groups: int = 32,
        **ignore_kwargs: object,
    ) -> None:
        # The reference's signature, which takes its whole ``ddconfig``.
        del out_ch, ignore_kwargs
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = resolution
        in_ch_mult = (1, *ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        num_groups=num_groups,
                        dropout=dropout,
                    ),
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, num_groups=num_groups))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in, num_groups=num_groups)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        temb = None

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = cast(
                    "ResnetBlock",
                    cast("_Stage", self.down[i_level]).block[i_block],
                )(hs[-1], temb=temb)
                if len(cast("_Stage", self.down[i_level]).attn) > 0:
                    h = cast(
                        "AttnBlock",
                        cast("_Stage", self.down[i_level]).attn[i_block],
                    )(h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(cast("_DownStage", self.down[i_level]).downsample(hs[-1]))

        h = hs[-1]
        h = cast("_Middle", self.mid).block_1(h, temb=temb)
        h = cast("_Middle", self.mid).attn_1(h)
        h = cast("_Middle", self.mid).block_2(h, temb=temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    """Latents to pixels: a middle, then residual stages doubling the grid."""

    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 16,
        give_pre_end: bool = False,
        num_groups: int = 32,
        **ignore_kwargs: object,
    ) -> None:
        # The reference's signature, which takes its whole ``ddconfig``.
        del ignore_kwargs
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = torch.nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.last_z_shape: torch.Size | None = None
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        num_groups=num_groups,
                        dropout=dropout,
                    ),
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, num_groups=num_groups))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # Prepend to get consistent order.

        self.norm_out = Normalize(block_in, num_groups=num_groups)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @override
    def forward(self, z: Tensor) -> Tensor:
        self.last_z_shape = z.shape

        temb = None

        h = self.conv_in(z)

        h = cast("_Middle", self.mid).block_1(h, temb=temb)
        h = cast("_Middle", self.mid).attn_1(h)
        h = cast("_Middle", self.mid).block_2(h, temb=temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = cast(
                    "ResnetBlock",
                    cast("_Stage", self.up[i_level]).block[i_block],
                )(h, temb=temb)
                if len(cast("_Stage", self.up[i_level]).attn) > 0:
                    h = cast(
                        "AttnBlock",
                        cast("_Stage", self.up[i_level]).attn[i_block],
                    )(h)
            if i_level != 0:
                h = cast("_UpStage", self.up[i_level]).upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class DiagonalGaussianDistribution:
    """The encoder's axis-aligned Gaussian over latents; satisfies ``Posterior``."""

    def __init__(self, parameters: Tensor) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Draw ``mean + std * noise``; ``None`` reads the global generator."""
        return self.mean + self.std * torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
        )

    def mode(self) -> Tensor:
        """Return the mean."""
        return self.mean


class INVAE(nn.Module):
    """INVAE behind the ``VariationalAutoencoder`` contract.

    ``encode`` returns whatever latent the config's ``latent_fn`` takes from
    the posterior -- a draw by default, which is what the REG corpora
    store -- and ``posterior`` exposes the distribution itself. Frozen: the
    wrapper stays in evaluation mode whatever its parent does.
    """

    class Config(Fig["INVAE"]):
        """Architecture size, latent choice, and checkpoint."""

        channels_hidden: int = 128
        """Stem width; stage ``i`` is ``channels_hidden * channel_multipliers[i]`` wide."""

        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4)
        """Width multiplier per stage; each stage after the first halves the grid."""

        channels_latent: int = 32
        """Latent channels."""

        blocks_per_stage: int = 2
        """Residual blocks per encoder stage; the decoder runs one more."""

        num_groups: int = 32
        """GroupNorm groups; must divide every level's width."""

        image_size: int = 256
        """Side of the square training image; attention runs where the grid is 16."""

        latent_fn: LatentFn = posterior_sample
        """Takes ``encode``'s latent from the posterior: a draw, or its mode."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: HubFile.Config(
                repo_id="REPA-E/e2e-invae",
                filename="e2e-invae-400k.pt",
                revision="79e3ea77928e1e34bc70e4e71de68fd5b4d4b3e0",
                sha256="4d45ef5452bd6325eaa4b5c74975328cfed5ecf15a877656bed0f3f6de23d718",
            ),
        )
        """Published weights; ``None`` keeps the random initialization."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: ScaleLatents.Config(scale=0.3099),
        )
        """The published scale; the autoencoder itself never applies it."""

        @override
        def finalize(self) -> Self:
            downsample = 1 << (len(self.channel_multipliers) - 1)
            if self.image_size % downsample:
                raise ValueError(
                    f"image_size {self.image_size} must be divisible by {downsample}.",
                )
            return super().finalize()

        def latent_shape(self) -> tuple[int, int, int]:
            """Return ``(channels_latent, side, side)`` at ``image_size``."""
            side = self.image_size // (1 << (len(self.channel_multipliers) - 1))
            return self.channels_latent, side, side

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one ``decode(encode(image))`` round trip at ``image_size``.

            Counts the uint8 cast and rescale, the encoder, ``quant_conv``, the
            posterior's clamp and exponentials, ``latent_fn`` by its own cost (a
            draw for :func:`posterior_sample`, nothing for the mode),
            ``post_quant_conv``, the decoder, and the output shift and clamp.
            The cast and rescale run in float32 whatever ``dtype``, as
            ``posterior`` casts. The weights are frozen, so only the forward is
            charged; the parameters are still owned.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            Raises:
              TypeError: ``latent_fn`` carries no cost.

            """
            del kwargs
            # A walk mirroring ``Encoder`` and ``Decoder.__init__`` rather than a
            # hooked meta-device forward: torch runs meta kernels in Python, ~50 ms
            # a call at the published size. The torch comparison in the tests
            # pins the walk to the modules.
            layers = _Layers(
                batch_size=batch_size,
                num_groups=self.num_groups,
                dtype=dtype,
            )
            ch, mult, blocks = (
                self.channels_hidden,
                self.channel_multipliers,
                self.blocks_per_stage,
            )
            z = self.channels_latent
            side = self.image_size
            pixels = batch_size * 3 * side * side
            latents = batch_size * math.prod(self.latent_shape())

            encoder = layers.conv(3, channels_out=ch, side=side, kernel_size=3)
            block_in = ch
            for level, m_out in enumerate(mult):
                block_in = ch * (mult[level - 1] if level else 1)
                for _ in range(blocks):
                    encoder += layers.resnet(
                        block_in,
                        channels_out=ch * m_out,
                        side=side,
                    )
                    block_in = ch * m_out
                    if side == _ATTENTION_SIDE:
                        encoder += layers.attention(block_in, side=side)
                if level != len(mult) - 1:
                    encoder += layers.downsample(block_in, side=side)
                    side //= 2
            encoder += layers.middle(block_in, side=side) + layers.head(
                block_in,
                channels_out=2 * z,
                side=side,
            )

            block_in = ch * mult[-1]
            decoder = layers.conv(z, channels_out=block_in, side=side, kernel_size=3)
            decoder += layers.middle(block_in, side=side)
            for level in reversed(range(len(mult))):
                for _ in range(blocks + 1):
                    decoder += layers.resnet(
                        block_in,
                        channels_out=ch * mult[level],
                        side=side,
                    )
                    block_in = ch * mult[level]
                    if side == _ATTENTION_SIDE:
                        decoder += layers.attention(block_in, side=side)
                if level != 0:
                    decoder += layers.upsample(block_in, side=side)
                    side *= 2
            decoder += layers.head(block_in, channels_out=3, side=side)

            latent_side = self.latent_shape()[1]
            f32 = torch.float32
            wrapper = (
                # ``image.float() / 127.5 - 1`` in.
                traffic(
                    "primal",
                    kernel="elementwise",
                    elements=pixels,
                    dtype=torch.uint8,
                )
                + traffic("primal", kernel="elementwise", elements=pixels, dtype=f32)
                + _pointwise(pixels, flops=1, dtype=f32).tile(2)
                # ``(x + 1) / 2`` and a two-sided clamp out.
                + _pointwise(pixels, flops=1, dtype=dtype).tile(2)
                + _pointwise(pixels, flops=2, dtype=dtype)
                + layers.conv(
                    2 * z,
                    channels_out=2 * z,
                    side=latent_side,
                    kernel_size=1,
                )
                + layers.conv(z, channels_out=z, side=latent_side, kernel_size=1)
                # Clamp logvar; ``exp(0.5 * logvar)`` and ``exp(logvar)``.
                + _pointwise(latents, flops=2, dtype=dtype)
                + _pointwise(latents, flops=1, dtype=dtype).tile(3)
                + cost(self.latent_fn, channels=latents, dtype=dtype)
            )
            full = encoder + decoder + wrapper
            return Cost(
                cells=full.only("primal").cells,
                params=full.params,
                params_active=full.params_active,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        architecture: _Architecture = {
            "ch": config.channels_hidden,
            "ch_mult": config.channel_multipliers,
            "num_res_blocks": config.blocks_per_stage,
            "resolution": config.image_size,
            "z_channels": config.channels_latent,
            "num_groups": config.num_groups,
        }
        # Registration order is the reference's -- encoder, decoder, then the two
        # 1x1 convolutions -- so initialization draws and checkpoint keys match it.
        self.encoder = Encoder(**architecture)
        self.decoder = Decoder(**architecture)
        self.quant_conv = torch.nn.Conv2d(
            2 * config.channels_latent,
            2 * config.channels_latent,
            1,
        )
        self.post_quant_conv = torch.nn.Conv2d(
            config.channels_latent,
            config.channels_latent,
            1,
        )
        self.latent_fn = config.latent_fn
        if config.checkpoint is not None:
            state = cast(
                dict[str, Tensor],
                torch.load(
                    config.checkpoint.make().path(),
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            self.load_state_dict(state)
        self.eval()
        self.requires_grad_(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def posterior(self, image: Tensor, /) -> DiagonalGaussianDistribution:
        """Return the encoder's distribution for a uint8 image batch.

        Args:
          image: ``[B, 3, H, W]`` uint8 RGB.

        Returns:
          posterior: Gaussian over ``[B, channels_latent, h, w]`` latents.

        Raises:
          TypeError: ``image`` is not uint8.

        """
        require_uint8(image)
        # Not ``rgb2float``: its ``(x - 127.5) / 127.5`` rounds 128 of the 256
        # levels differently from the reference's ``x / 127.5 - 1``, which the
        # published weights and existing corpora saw.
        moments = self.quant_conv(self.encoder(image.float() / 127.5 - 1))
        return DiagonalGaussianDistribution(moments)

    def encode(self, image: Tensor, /) -> Tensor:
        """Return the configured latent of the posterior.

        Args:
          image: ``[B, 3, H, W]`` uint8 RGB.

        Returns:
          latent: ``[B, channels_latent, h, w]`` raw latent.

        Raises:
          TypeError: ``image`` is not uint8.

        """
        return self.latent_fn(self.posterior(image))

    def decode(self, latent: Tensor, /) -> Tensor:
        """Decode a raw latent to RGB in ``[0, 1]``.

        Args:
          latent: ``[B, channels_latent, h, w]`` raw latent.

        Returns:
          image: ``[B, 3, H, W]`` float RGB, clamped.

        """
        decoded = self.decoder(self.post_quant_conv(latent))
        return ((decoded + 1) / 2).clamp(0, 1)


# ``Encoder`` and ``Decoder`` default ``attn_resolutions=(16,)``; INVAE never
# overrides it, so attention runs at every block whose grid side is 16.
_ATTENTION_SIDE: Final = 16


@dataclass(frozen=True, slots=True, kw_only=True)
class _Layers:
    """Price the reference's layers at one batch size, group count, and dtype."""

    batch_size: int
    num_groups: int
    dtype: torch.dtype | None

    def conv(
        self,
        channels_in: int,
        channels_out: int,
        side: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
    ) -> Cost:
        """Cost a biased convolution on a square grid.

        Args:
          channels_in: Input channels.
          channels_out: Output channels.
          side: Input grid side.
          kernel_size: Kernel side.
          stride: Step between applications.
          padding: Zero padding per side; ``None`` pads same.

        Returns:
          cost: The convolution's ledger.

        """
        return conv_cost(
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=kernel_size,
            ndim=2,
            groups=1,
            bias=True,
            input_grid=(side, side),
            batch_size=self.batch_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            dtype=self.dtype,
        )

    def norm(self, channels: int, side: int) -> Cost:
        """Cost ``Normalize``, an affine GroupNorm.

        Args:
          channels: Channels normalized.
          side: Grid side.

        Returns:
          cost: The norm's ledger.

        """
        return cost(
            GroupNorm2d.Config(
                channels,
                num_groups=self.num_groups,
                elementwise_affine=True,
            ),
            seq_len=side * side,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def resnet(self, channels_in: int, channels_out: int, side: int) -> Cost:
        """Cost ``ResnetBlock`` without a time embedding.

        Args:
          channels_in: Input channels.
          channels_out: Output channels.
          side: Grid side.

        Returns:
          cost: The block's ledger, its shortcut projection included.

        """
        elements_in = self.batch_size * channels_in * side * side
        elements_out = self.batch_size * channels_out * side * side
        block = (
            self.norm(channels_in, side=side)
            + _swish(elements_in, dtype=self.dtype)
            + self.conv(
                channels_in,
                channels_out=channels_out,
                side=side,
                kernel_size=3,
            )
            + self.norm(channels_out, side=side)
            + _swish(elements_out, dtype=self.dtype)
            + self.conv(
                channels_out,
                channels_out=channels_out,
                side=side,
                kernel_size=3,
            )
            + _pointwise(elements_out, flops=1, inputs=2, dtype=self.dtype)
        )
        if channels_in != channels_out:
            block += self.conv(
                channels_in,
                channels_out=channels_out,
                side=side,
                kernel_size=1,
            )
        return block

    def attention(self, channels: int, side: int) -> Cost:
        """Cost ``AttnBlock``: single-head attention over the grid.

        Args:
          channels: Channels attended over.
          side: Grid side.

        Returns:
          cost: The block's ledger.

        """
        return (
            self.norm(channels, side=side)
            + self.conv(channels, channels_out=channels, side=side, kernel_size=1).tile(
                4,
                copies=4,
            )
            + attention_kernel_cost(
                seq_len=side * side,
                batch_size=self.batch_size,
                dtype=self.dtype,
                num_heads=1,
                channels_head=channels,
            )
            + _pointwise(
                self.batch_size * channels * side * side,
                flops=1,
                inputs=2,
                dtype=self.dtype,
            )
        )

    def middle(self, channels: int, side: int) -> Cost:
        """Cost the ``mid`` stage: block, attention, block."""
        return self.resnet(channels, channels_out=channels, side=side).tile(
            2,
            copies=2,
        ) + self.attention(channels, side=side)

    def head(self, channels_in: int, channels_out: int, side: int) -> Cost:
        """Cost ``norm_out``, swish, and ``conv_out``.

        Args:
          channels_in: Input channels.
          channels_out: Output channels.
          side: Grid side.

        Returns:
          cost: The head's ledger.

        """
        return (
            self.norm(channels_in, side=side)
            + _swish(self.batch_size * channels_in * side * side, dtype=self.dtype)
            + self.conv(
                channels_in,
                channels_out=channels_out,
                side=side,
                kernel_size=3,
            )
        )

    def downsample(self, channels: int, side: int) -> Cost:
        """Cost ``Downsample``: a one-sided zero pad, then a stride-2 conv.

        Args:
          channels: Channels resampled.
          side: Input grid side.

        Returns:
          cost: The pad's traffic and the convolution's ledger.

        """
        elements = self.batch_size * channels * side * side
        padded = self.batch_size * channels * (side + 1) ** 2
        return traffic(
            "primal",
            kernel="elementwise",
            elements=elements + padded,
            dtype=self.dtype,
        ) + self.conv(
            channels,
            channels_out=channels,
            side=side + 1,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def upsample(self, channels: int, side: int) -> Cost:
        """Cost ``Upsample``: nearest-neighbour doubling, then a conv.

        Args:
          channels: Channels resampled.
          side: Input grid side.

        Returns:
          cost: The interpolation's traffic and the convolution's ledger.

        """
        elements = self.batch_size * channels * side * side
        return traffic(
            "primal",
            kernel="elementwise",
            elements=5 * elements,
            dtype=self.dtype,
        ) + self.conv(channels, channels_out=channels, side=2 * side, kernel_size=3)


def _swish(elements: int, *, dtype: torch.dtype | None) -> Cost:
    """Cost ``nonlinearity``: a sigmoid, then its product with the input."""
    return cost(sigmoid, channels=elements, dtype=dtype) + _pointwise(
        elements,
        flops=1,
        inputs=2,
        dtype=dtype,
    )


def _pointwise(
    elements: int,
    *,
    flops: int,
    inputs: int = 1,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one forward tensor op doing ``flops`` per element of its output."""
    return elementwise_cost(
        primal=flops * elements,
        adjoint=0,
        channels=elements,
        inputs=inputs,
        adjoint_inputs=0,
        adjoint_outputs=0,
        dtype=dtype,
    )


class _Stage(Protocol):
    """A level of the reference's checkpoint hierarchy: its blocks and attention."""

    block: nn.ModuleList
    attn: nn.ModuleList


class _DownStage(_Stage, Protocol):
    """An encoder level; every one but the last also downsamples."""

    downsample: Downsample


class _UpStage(_Stage, Protocol):
    """A decoder level; every one but the first also upsamples."""

    upsample: Upsample


class _Middle(Protocol):
    """The reference's two residual blocks surrounding one attention block."""

    block_1: ResnetBlock
    attn_1: AttnBlock
    block_2: ResnetBlock


class _Architecture(TypedDict):
    """Shared architecture arguments accepted by both reference modules."""

    ch: int
    ch_mult: tuple[int, ...]
    num_res_blocks: int
    resolution: int
    z_channels: int
    num_groups: int
