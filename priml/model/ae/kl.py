"""AutoencoderKL: the latent-diffusion convolutional KL autoencoder.

Stages of residual blocks halve the grid down to a middle of block, attention,
block; a diagonal Gaussian over the latent sits between the halves. Published
checkpoints differ in widths, depth, attention placement, and whether the halves
project the latent, all config values; in the arithmetic of their attention,
an injected kernel; and in their tensor names, which a half's ``remap`` reads.

Build a published model from its factory -- :func:`invae`, :func:`sd_ft_mse`,
:func:`sdxl`, :func:`sdxl_fp16_fix`, :func:`flux1`, :func:`flux2` -- or parse a
diffusers ``config.json`` with :meth:`AutoencoderKL.Config.from_hf`.

Licenses and attribution: ``LICENSES/REG``, ``LICENSES/flux``,
``LICENSES/flux2``. CompVis's code, under CreativeML Open RAIL-M, is not
ported: its outputs are replayed in the tests.

References:
  https://github.com/SwayStar123/REG/blob/3c51606c801dd9e87ee9ef778782766ab7c379ca/models/invae.py
  https://huggingface.co/REPA-E/e2e-invae-hf
    Leng et al. 2025. REPA-E: Unlocking VAE for end-to-end tuning with latent
    diffusion transformers.
  https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
    Rombach et al. 2022. High-resolution image synthesis with latent
    diffusion models.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Self, cast, override

import re

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain

import torch

from priml.cost import Cost, cost, traffic
from priml.math.pixel import patchify, unpatchify
from priml.model.ae.checkpoint import (
    CheckpointFile,
    HubFile,
    hub_or_local,
    read_json,
    read_state,
)
from priml.model.ae.custom_types import StateDictRemap
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.frozen_cost import SUM, kl_round_trip_cost
from priml.model.ae.latent_norm import (
    BatchNormLatents,
    LatentNormalizer,
    ScaleLatents,
    ShiftScaleLatents,
)
from priml.model.attention.kernel import SdpaFused, attention_kernel_cost
from priml.model.conv import Conv2d, conv_cost
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.norm import GroupNorm2d
from priml.model.swiglu import sigmoid


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


INVAE_REPO: Final = "REPA-E/e2e-invae-hf"
"""INVAE's published diffusers checkpoint."""

INVAE_REVISION: Final = "523ddf74b061c01ed0aebb7904c6bdb2f33e06d0"
"""The commit of :data:`INVAE_REPO` the defaults pin."""


def e2e_invae_hf() -> HubFile.Config:
    """Return the pinned diffusers safetensors both halves read by default.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id=INVAE_REPO,
        filename="diffusion_pytorch_model.safetensors",
        revision=INVAE_REVISION,
        sha256="6b1f754e414a841893fd242ca233c5aeec4bbdb7d2eeab2f7956adf48742e500",
    )


def remap_hf_state_dict(state: Mapping[str, Tensor], *, half: str) -> dict[str, Tensor]:
    """Rename one half's diffusers tensors to this module's names.

    Args:
      state: Diffusers tensors; keys outside ``half`` are ignored.
      half: ``"encoder"`` or ``"decoder"``.

    Returns:
      state: This half's ``state_dict``.

    """
    quant = {"encoder": "quant_conv.", "decoder": "post_quant_conv."}[half]
    remapped: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key.startswith(quant):
            remapped[key] = tensor
        elif key.startswith(f"{half}."):
            name = key.removeprefix(f"{half}.").replace(".to_out.0.", ".to_out.")
            projection = _ATTENTION_PROJECTION.search(name) is not None
            remapped[name] = tensor[..., None, None] if projection else tensor
    return remapped


def remap_ldm_state_dict(
    state: Mapping[str, Tensor],
    *,
    half: str,
) -> dict[str, Tensor]:
    """Rename one half's latent-diffusion tensors to this module's names.

    Reads the CompVis layout SD-1.x and SDXL ship (``down.0.block.1``,
    ``mid.attn_1.q``, ``nin_shortcut``), with its projections at the top level,
    and BFL's FLUX files, which nest them under their half
    (``encoder.quant_conv``). The decoder's ``up`` levels count from the finest
    grid, its stages from the coarsest.

    Args:
      state: Latent-diffusion tensors; keys outside ``half`` are ignored.
      half: ``"encoder"`` or ``"decoder"``.

    Returns:
      state: This half's ``state_dict``.

    """
    quant = {"encoder": "quant_conv.", "decoder": "post_quant_conv."}[half]
    prefix = f"{half}."
    levels = 1 + max(
        (
            int(match.group(1))
            for key in state
            if (match := _LDM_UP.match(key.removeprefix(prefix))) is not None
        ),
        default=-1,
    )
    remapped: dict[str, Tensor] = {}
    for key, tensor in state.items():
        nested = key.removeprefix(prefix)
        if key.startswith(quant) or (
            key.startswith(prefix) and nested.startswith(quant)
        ):
            remapped[key.removeprefix(prefix)] = tensor
        elif key.startswith(prefix):
            name = _LDM_UP.sub(
                lambda match: f"up_blocks.{levels - 1 - int(match.group(1))}.",
                nested,
            )
            for pattern, replacement in _LDM_RENAMES:
                name = pattern.sub(replacement, name)
            remapped[name] = tensor
    return remapped


class LdmAttention(nn.Module):
    """Single-head attention in the latent-diffusion reference's own arithmetic.

    Scores are one ``bmm`` multiplied by ``channels ** -0.5``, and the values are
    weighted by a second ``bmm``, as CompVis's and REG's ``AttnBlock`` compute
    them; :class:`~priml.model.attention.kernel.SdpaFused` divides and rounds
    differently. Takes and returns ``[N, S, 1, C]``.

    References:
      https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py

    """

    class Config(Fig["LdmAttention"]):
        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            **kwargs: object,
        ) -> Cost:
            """Cost the two products and the softmax.

            Args:
              seq_len: Grid positions.
              batch_size: Grids in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Heads; the reference has one.
              channels_head: Channels.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            """
            del kwargs
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_head,
            )

    def __init__(self, config: Config | None = None) -> None:
        del config
        super().__init__()

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        attn_mask: Tensor | None = None,
        window: int = -1,
        scale: float | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Attend ``[N, S, 1, C]`` queries to keys and values of the same shape.

        Raises:
          ValueError: More than one head, or a dropout, causal mask, mask,
            window, or scale, none of which the reference computes.

        """
        del kwargs
        if q.shape[-2] != 1:
            raise ValueError(f"LdmAttention has one head; got {q.shape[-2]}.")
        if (
            dropout_p
            or is_causal
            or attn_mask is not None
            or window >= 0
            or (scale is not None)
        ):
            raise ValueError(
                "LdmAttention attends every key at the reference's 1/sqrt(C) scale; "
                "it takes no dropout, causal mask, mask, window, or scale.",
            )
        # The reference's operands, strides included: the queries ``[N, S, C]``,
        # the keys and values ``[N, C, S]``.
        scores = torch.bmm(q.squeeze(-2), k.squeeze(-2).transpose(1, 2))
        weights = functional.softmax(
            scores * cast(float, q.shape[-1] ** (-0.5)),
            dim=2,
        )
        attended = torch.bmm(v.squeeze(-2).transpose(1, 2), weights.permute(0, 2, 1))
        return attended.transpose(1, 2).unsqueeze(-2)


class Encoder(nn.Module):
    """Pixels in ``[-1, 1]`` to posterior moments: stages halving the grid, then a middle."""

    class Config(Fig["Encoder"]):
        """Widths, depth, attention placement, and weights."""

        channels_in: int = 3
        """Image channels."""

        channels_hidden: tuple[int, ...] = (128, 128, 256, 256, 512)
        """Width per stage, finest first; each stage after the first halves the grid."""

        blocks_per_stage: int = 2
        """Residual blocks per stage."""

        attention: tuple[bool, ...] = (False, False, False, False, True)
        """Whether each stage's blocks self-attend, finest first."""

        channels_latent: int = 32
        """Latent channels; the encoder emits twice as many, mean then log-variance."""

        num_groups: int = 32
        """GroupNorm groups; must divide every stage's width."""

        eps: float = 1e-6
        """GroupNorm epsilon."""

        attn_kernel: Makeable[AttentionKernel] = field(
            default_factory=LdmAttention.Config,
        )
        """Every self-attention's arithmetic; ``SdpaFused`` where the reference calls SDPA."""

        proj_out: Makeable[TensorModule] | None = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """Projection of the moments, diffusers' ``quant_conv``; ``None`` omits it.

        Its widths are set to the moments'.
        """

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=e2e_invae_hf,
        )
        """Weights; only ``encoder.*`` and the moments' projection are read.

        ``None`` keeps the random initialization.
        """

        remap: StateDictRemap = remap_hf_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        @override
        def finalize(self) -> Self:
            if self.proj_out is not None:
                moments = 2 * self.channels_latent
                propagate_attr(
                    self.proj_out,
                    "channels_in",
                    moments,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    self.proj_out,
                    "channels_out",
                    moments,
                    protocol=ChannelsOut,
                )
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a square ``input_grid`` image, ``proj_out`` included.

            Args:
              input_grid: Image side.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            del kwargs
            layers = _Layers(
                batch_size=batch_size,
                num_groups=self.num_groups,
                attn_kernel=self.attn_kernel,
                dtype=dtype,
            )
            widths, side = self.channels_hidden, input_grid
            total = layers.conv(self.channels_in, widths[0], side=side, kernel_size=3)
            width_in = widths[0]
            for stage, (width, attends) in enumerate(
                zip(widths, self.attention, strict=True),
            ):
                for _ in range(self.blocks_per_stage):
                    total += layers.resnet(width_in, width, side=side)
                    width_in = width
                    if attends:
                        total += layers.attention(width, side=side)
                if stage != len(widths) - 1:
                    total += layers.downsample(width, side=side)
                    side //= 2
            total += layers.middle(width_in, side=side) + layers.head(
                width_in,
                channels_out=2 * self.channels_latent,
                side=side,
            )
            # A finalized copy's: an unfinalized slot still holds -1 widths.
            proj_out = self.finalized().proj_out
            if proj_out is not None:
                total += cost(
                    proj_out,
                    input_grid=side,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        widths, groups, eps = config.channels_hidden, config.num_groups, config.eps
        kernel = config.attn_kernel
        self.conv_in = _conv(config.channels_in, widths[0], kernel_size=3)
        self.down_blocks: nn.ModuleList[_Stage] = nn.ModuleList()
        width_in = widths[0]
        for stage, (width, attends) in enumerate(
            zip(widths, config.attention, strict=True),
        ):
            level = _Stage()
            for _ in range(config.blocks_per_stage):
                level.resnets.append(
                    _Resnet(width_in, width, num_groups=groups, eps=eps),
                )
                width_in = width
                if attends:
                    level.attentions.append(
                        _Attention(width, num_groups=groups, eps=eps, kernel=kernel),
                    )
            if stage != len(widths) - 1:
                level.downsamplers.append(_Downsample(width))
            self.down_blocks.append(level)
        self.mid_block = _Middle(width_in, num_groups=groups, eps=eps, kernel=kernel)
        self.conv_norm_out = _norm(width_in, num_groups=groups, eps=eps)
        self.conv_out = _conv(width_in, 2 * config.channels_latent, kernel_size=3)
        # Named for the checkpoints' own tensors, so a remap only strips prefixes.
        self.quant_conv = None if config.proj_out is None else config.proj_out.make()
        if config.checkpoint is not None:
            path = config.checkpoint.make().path()
            state = read_state(path, prefixes=("encoder.", "quant_conv."))
            self.load_state_dict(config.remap(state, half="encoder"))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[N, C_in, H, W]`` pixels in ``[-1, 1]`` to ``[N, 2 C, h, w]`` moments."""
        h = self.conv_in(x)
        for level in self.down_blocks:
            h = level(h)
        h = self.mid_block(h)
        h = self.conv_out(_swish(self.conv_norm_out(h)))
        return h if self.quant_conv is None else self.quant_conv(h)


class Decoder(nn.Module):
    """Latents to pixels in ``[-1, 1]``: a middle, then stages doubling the grid."""

    class Config(Fig["Decoder"]):
        """Widths, depth, attention placement, and weights; stages listed finest first."""

        channels_out: int = 3
        """Image channels."""

        channels_hidden: tuple[int, ...] = (128, 128, 256, 256, 512)
        """Width per stage, finest first; the decoder runs them coarsest first."""

        blocks_per_stage: int = 3
        """Residual blocks per stage; the published decoder runs one more than its encoder."""

        attention: tuple[bool, ...] = (False, False, False, False, True)
        """Whether each stage's blocks self-attend, finest first."""

        channels_latent: int = 32
        """Latent channels."""

        num_groups: int = 32
        """GroupNorm groups; must divide every stage's width."""

        eps: float = 1e-6
        """GroupNorm epsilon."""

        attn_kernel: Makeable[AttentionKernel] = field(
            default_factory=LdmAttention.Config,
        )
        """Every self-attention's arithmetic; ``SdpaFused`` where the reference calls SDPA."""

        proj_in: Makeable[TensorModule] | None = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """Projection of the latent, diffusers' ``post_quant_conv``; ``None`` omits it.

        Its widths are set to the latent's.
        """

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=e2e_invae_hf,
        )
        """Weights; only ``decoder.*`` and the latent's projection are read.

        ``None`` keeps the random initialization.
        """

        remap: StateDictRemap = remap_hf_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        @override
        def finalize(self) -> Self:
            if self.proj_in is not None:
                z = self.channels_latent
                propagate_attr(self.proj_in, "channels_in", z, protocol=ChannelsIn)
                propagate_attr(self.proj_in, "channels_out", z, protocol=ChannelsOut)
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a square ``input_grid`` latent, ``proj_in`` included.

            Args:
              input_grid: Latent side.
              batch_size: Latents in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            del kwargs
            layers = _Layers(
                batch_size=batch_size,
                num_groups=self.num_groups,
                attn_kernel=self.attn_kernel,
                dtype=dtype,
            )
            z, side = self.channels_latent, input_grid
            width_in = self.channels_hidden[-1]
            total = layers.conv(z, width_in, side=side, kernel_size=3) + layers.middle(
                width_in,
                side=side,
            )
            # A finalized copy's, as in ``Encoder.Config.cost``.
            proj_in = self.finalized().proj_in
            if proj_in is not None:
                total += cost(
                    proj_in,
                    input_grid=side,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            stages = list(zip(self.channels_hidden, self.attention, strict=True))
            for stage, (width, attends) in reversed(list(enumerate(stages))):
                for _ in range(self.blocks_per_stage):
                    total += layers.resnet(width_in, width, side=side)
                    width_in = width
                    if attends:
                        total += layers.attention(width, side=side)
                if stage != 0:
                    total += layers.upsample(width, side=side)
                    side *= 2
            return total + layers.head(
                width_in,
                channels_out=self.channels_out,
                side=side,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        widths, groups, eps = config.channels_hidden, config.num_groups, config.eps
        kernel = config.attn_kernel
        z = config.channels_latent
        # Named for the checkpoints' own tensors, as the encoder's ``quant_conv``.
        self.post_quant_conv = None if config.proj_in is None else config.proj_in.make()
        width_in = widths[-1]
        self.conv_in = _conv(z, width_in, kernel_size=3)
        self.mid_block = _Middle(width_in, num_groups=groups, eps=eps, kernel=kernel)
        self.up_blocks: nn.ModuleList[_Stage] = nn.ModuleList()
        stages = list(zip(widths, config.attention, strict=True))
        for stage, (width, attends) in reversed(list(enumerate(stages))):
            level = _Stage()
            for _ in range(config.blocks_per_stage):
                level.resnets.append(
                    _Resnet(width_in, width, num_groups=groups, eps=eps),
                )
                width_in = width
                if attends:
                    level.attentions.append(
                        _Attention(width, num_groups=groups, eps=eps, kernel=kernel),
                    )
            if stage != 0:
                level.upsamplers.append(_Upsample(width))
            self.up_blocks.append(level)
        self.conv_norm_out = _norm(width_in, num_groups=groups, eps=eps)
        self.conv_out = _conv(width_in, config.channels_out, kernel_size=3)
        if config.checkpoint is not None:
            path = config.checkpoint.make().path()
            state = read_state(path, prefixes=("decoder.", "post_quant_conv."))
            self.load_state_dict(config.remap(state, half="decoder"))

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Map ``[N, C, h, w]`` latents to ``[N, C_out, H, W]`` pixels in ``[-1, 1]``."""
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        h = self.mid_block(self.conv_in(z))
        for level in self.up_blocks:
            h = level(h)
        return self.conv_out(_swish(self.conv_norm_out(h)))


class AutoencoderKL(nn.Module):
    """A KL autoencoder as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a diagonal Gaussian; call ``sample()`` or ``mode()`` on
    it. The model is frozen and always in evaluation mode.
    """

    class Config(Fig["AutoencoderKL"]):
        """The halves, the image side, and the published latent normalizer.

        The defaults are INVAE's published model, which :func:`invae` returns.
        """

        encoder: Encoder.Config | None = field(default_factory=Encoder.Config)
        """Pixels to posterior; ``None`` builds and loads no encoder."""

        decoder: Decoder.Config | None = field(default_factory=Decoder.Config)
        """Latents to pixels; ``None`` builds and loads no decoder."""

        image_size: int = 256
        """Side of the square image the checkpoint was trained at."""

        patch_size: int = 1
        """Side of the patch folded into channels between the halves.

        The posterior and the latent ``decode`` takes are over the folded grid,
        ``channels_latent * patch_size**2`` channels; FLUX.2 folds 2.
        """

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: ScaleLatents.Config(scale=0.3099),
        )
        """The normalizer the paired diffusion models train with; INVAE's, REG's scale, by default.

        The autoencoder never applies it.
        """

        @classmethod
        def from_hf(
            cls,
            config: Mapping[str, object],
            *,
            checkpoint: Makeable[CheckpointFile] | None = None,
        ) -> Self:
            """Parse a diffusers ``AutoencoderKL`` ``config.json``.

            The architecture only: ``sample_size``, ``scaling_factor``,
            ``shift_factor``, ``latents_mean``, and ``latents_std`` are not read,
            so ``image_size`` and ``latent_norm`` keep their defaults.

            Args:
              config: The ``config.json`` object.
              checkpoint: Weights both halves read; ``None`` keeps a random init.

            Returns:
              config: Architecture with the published latent normalizer.

            Raises:
              ValueError: A field names an architecture this port does not build.

            """
            _reject_unsupported(config)
            widths = tuple(from_plain(config["block_out_channels"], list[int]))
            latent = from_plain(config["latent_channels"], int)
            groups = from_plain(config["norm_num_groups"], int)
            layers = from_plain(config["layers_per_block"], int)
            down = from_plain(config["down_block_types"], list[str])
            up = from_plain(config["up_block_types"], list[str])
            encoder = Encoder.Config(
                channels_in=from_plain(config["in_channels"], int),
                channels_hidden=widths,
                blocks_per_stage=layers,
                attention=tuple(kind.startswith("Attn") for kind in down),
                channels_latent=latent,
                num_groups=groups,
                checkpoint=checkpoint,
            )
            # Diffusers lists the decoder's stages coarsest first.
            decoder = Decoder.Config(
                channels_out=from_plain(config["out_channels"], int),
                channels_hidden=widths,
                blocks_per_stage=layers + 1,
                attention=tuple(kind.startswith("Attn") for kind in reversed(up)),
                channels_latent=latent,
                num_groups=groups,
                checkpoint=checkpoint,
            )
            if not from_plain(config.get("use_quant_conv"), bool, default=True):
                encoder.proj_out = None
            if not from_plain(config.get("use_post_quant_conv"), bool, default=True):
                decoder.proj_in = None
            return cls(encoder=encoder, decoder=decoder)

        def latent_shape(self) -> tuple[int, int, int]:
            """Return ``(channels, side, side)`` of the folded latent at ``image_size``."""
            half = self.encoder or self.decoder
            if half is None:
                raise ValueError("AutoencoderKL.Config needs an encoder or a decoder.")
            side = (
                self.image_size >> (len(half.channels_hidden) - 1)
            ) // self.patch_size
            return half.channels_latent * self.patch_size**2, side, side

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one ``decode(encode(image).sample())`` round trip at ``image_size``.

            Forward only; the parameters are still owned.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to both halves.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            if self.encoder is None or self.decoder is None:
                raise ValueError(
                    "AutoencoderKL.Config.cost prices both halves; one is None.",
                )
            channels, side, _ = self.latent_shape()
            area = batch_size * self.image_size**2
            pixels_in = area * self.encoder.channels_in
            pixels_out = area * self.decoder.channels_out
            latents = batch_size * channels * side * side
            full = (
                cost(
                    self.encoder,
                    **{
                        **kwargs,
                        "input_grid": self.image_size,
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                + cost(
                    self.decoder,
                    **{
                        **kwargs,
                        "input_grid": side * self.patch_size,
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
            if self.patch_size > 1:
                # Folding the mean and the log-variance, unfolding the draw: each
                # a strided copy, read once and written once.
                full += traffic(
                    "primal",
                    kernel="elementwise",
                    elements=6 * latents,
                    dtype=dtype,
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
            raise ValueError("AutoencoderKL needs an encoder or a decoder.")
        if config.patch_size < 1:
            raise ValueError(f"patch_size must be positive; got {config.patch_size}.")
        downsample = (1 << (len(halves[0].channels_hidden) - 1)) * config.patch_size
        if config.image_size % downsample:
            raise ValueError(
                f"image_size {config.image_size} must be divisible by {downsample}.",
            )
        if len({half.channels_latent for half in halves}) != 1:
            raise ValueError("The encoder and decoder disagree on channels_latent.")
        if len({len(half.channels_hidden) for half in halves}) != 1:
            raise ValueError(
                "The encoder and decoder disagree on their number of stages, so the "
                "decoder would not return image_size.",
            )
        self.encoder = None if config.encoder is None else config.encoder.make()
        self.decoder = None if config.decoder is None else config.decoder.make()
        self.patch = (config.patch_size, config.patch_size)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def load(
        cls,
        path_or_repo: Path | str = INVAE_REPO,
        *,
        revision: str = INVAE_REVISION,
    ) -> AutoencoderKL:
        """Build from a diffusers directory or Hub repository, weights included.

        Args:
          path_or_repo: Local directory holding ``config.json`` and
            ``diffusion_pytorch_model.safetensors``, or a Hub repository.
          revision: Commit to read a Hub repository at.

        Returns:
          model: Frozen autoencoder, its ``image_size`` and ``latent_norm`` the
            defaults.

        """
        config = cls.Config.from_hf(
            read_json(hub_or_local(path_or_repo, "config.json", revision=revision)),
            checkpoint=hub_or_local(
                path_or_repo,
                "diffusion_pytorch_model.safetensors",
                revision=revision,
            ),
        )
        return config.make()

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, x: Tensor, /) -> DiagonalGaussian:
        """Return the posterior over latents.

        Args:
          x: ``[..., 3, H, W]`` uint8 RGB.

        Returns:
          posterior: Diagonal Gaussian over the folded latent
            ``[..., channels_latent * patch_size**2, h, w]``.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder.

        """
        if self.encoder is None:
            raise ValueError("This AutoencoderKL was built without an encoder.")
        # A float image in [0, 1] would otherwise encode as near-black.
        if x.dtype != torch.uint8:
            raise TypeError(
                f"AutoencoderKL encodes uint8 RGB in [0, 255]; got {x.dtype}.",
            )
        # ``x / 127.5 - 1``, not ``rgb2float``'s ``(x - 127.5) / 127.5``: the two
        # round 128 of the 256 levels differently, and the weights saw this one.
        pixels = x.reshape(-1, *x.shape[-3:]).float() / 127.5 - 1
        moments = self.encoder(pixels)
        moments = moments.reshape(*x.shape[:-3], *moments.shape[1:])
        mean, logvar = moments.chunk(2, dim=-3)
        if self.patch != (1, 1):
            mean, logvar = patchify(mean, self.patch), patchify(logvar, self.patch)
        return DiagonalGaussian(mean, logvar)

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels_latent * patch_size**2, h, w]`` raw latents.

        Returns:
          image: ``[..., 3, H, W]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder.

        """
        if self.decoder is None:
            raise ValueError("This AutoencoderKL was built without a decoder.")
        if self.patch != (1, 1):
            z = unpatchify(z, self.patch)
        decoded = self.decoder(z.reshape(-1, *z.shape[-3:]))
        decoded = decoded.reshape(*z.shape[:-3], *decoded.shape[1:])
        return ((decoded + 1) / 2).clamp(0, 1)


def invae() -> AutoencoderKL.Config:
    """Return INVAE: 32 channels, 16x downsampling, REG's latent scale.

    Returns:
      config: The published model, weights pinned.

    """
    return AutoencoderKL.Config()


def sd_ft_mse() -> AutoencoderKL.Config:
    """Return SD-1.x's VAE fine-tuned 840k steps on MSE, EMA weights: 4 channels, 8x.

    Attention in CompVis's arithmetic, the code SD-1.x shipped with.

    Returns:
      config: The published model, weights pinned.

    References:
      https://huggingface.co/stabilityai/sd-vae-ft-mse-original
      Rombach et al. 2022. High-resolution image synthesis with latent
      diffusion models.

    """
    config = _ldm(
        HubFile.Config(
            repo_id="stabilityai/sd-vae-ft-mse-original",
            filename="vae-ft-mse-840000-ema-pruned.safetensors",
            revision="629b3ad3030ce36e15e70c5db7d91df0d60c627f",
            sha256="735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75",
        ),
        channels_latent=4,
        attn_kernel=LdmAttention.Config(),
    )
    config.latent_norm = ScaleLatents.Config(scale=0.18215)
    return config


def sdxl() -> AutoencoderKL.Config:
    """Return SDXL's VAE: SD-1.x's architecture retrained, 4 channels, 8x.

    Attention through SDPA, as Stability's ``sgm`` computes it.

    Returns:
      config: The published model, weights pinned.

    References:
      https://huggingface.co/stabilityai/sdxl-vae
      Podell et al. 2023. SDXL: Improving latent diffusion models for
      high-resolution image synthesis.

    """
    config = _ldm(
        HubFile.Config(
            repo_id="stabilityai/sdxl-vae",
            filename="sdxl_vae.safetensors",
            revision="6f5909a7e596173e25d4e97b07fd19cdf9611c76",
            sha256="63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e",
        ),
        channels_latent=4,
        attn_kernel=SdpaFused.Config(),
    )
    config.image_size = 1024
    config.latent_norm = ScaleLatents.Config(scale=0.13025)
    return config


def sdxl_fp16_fix() -> AutoencoderKL.Config:
    """Return SDXL's VAE fine-tuned to keep its activations finite in float16.

    The latents match :func:`sdxl`'s closely but not exactly.

    Returns:
      config: The published model, weights pinned.

    References:
      https://huggingface.co/madebyollin/sdxl-vae-fp16-fix

    """
    config = sdxl()
    checkpoint = HubFile.Config(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="sdxl_vae.safetensors",
        revision="207b116dae70ace3637169f1ddd2434b91b3a8cd",
        sha256="235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1",
    )
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = checkpoint
    config.decoder.checkpoint = checkpoint.copy_tree()
    return config


def flux1() -> AutoencoderKL.Config:
    """Return FLUX.1's VAE: 16 channels, 8x, no projection between the halves.

    Returns:
      config: The published model, weights pinned.

    References:
      https://huggingface.co/black-forest-labs/FLUX.1-dev
      https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/modules/autoencoder.py
        License: ``LICENSES/flux``.

    """
    config = _ldm(
        HubFile.Config(
            repo_id="black-forest-labs/FLUX.1-dev",
            filename="ae.safetensors",
            revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
            sha256="afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        ),
        channels_latent=16,
        attn_kernel=SdpaFused.Config(),
    )
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.proj_out = None
    config.decoder.proj_in = None
    config.image_size = 1024
    config.latent_norm = ShiftScaleLatents.Config(shift=0.1159, scale=0.3611)
    return config


def flux2() -> AutoencoderKL.Config:
    """Return FLUX.2's VAE: 32 channels at 8x, folded 2x2 to 128 channels at 16x.

    Its normalizer is the checkpoint's own frozen batch norm over the folded
    latent.

    Returns:
      config: The published model, weights pinned.

    References:
      https://huggingface.co/black-forest-labs/FLUX.2-dev
      https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/autoencoder.py
        License: ``LICENSES/flux2``.

    """
    checkpoint = HubFile.Config(
        repo_id="black-forest-labs/FLUX.2-dev",
        filename="ae.safetensors",
        revision="26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
        sha256="868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
    )
    config = _ldm(
        checkpoint,
        channels_latent=32,
        attn_kernel=SdpaFused.Config(),
    )
    config.image_size = 1024
    config.patch_size = 2
    stats = checkpoint.copy_tree()
    config.latent_norm = BatchNormLatents.Config(stats=stats, eps=1e-4)
    return config


# Every published latent-diffusion checkpoint shares this layout; each half gets its
# own copies, so a fork editing one half's checkpoint or kernel leaves the other's.
def _ldm(
    checkpoint: HubFile.Config,
    *,
    channels_latent: int,
    attn_kernel: Makeable[AttentionKernel],
) -> AutoencoderKL.Config:
    """Return the latent-diffusion 8x layout with this checkpoint, width, and kernel."""
    config = AutoencoderKL.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    for half in (config.encoder, config.decoder):
        half.channels_hidden = (128, 256, 512, 512)
        half.attention = (False, False, False, False)
        half.channels_latent = channels_latent
        half.attn_kernel = attn_kernel.copy_tree()
        half.checkpoint = checkpoint.copy_tree()
        half.remap = remap_ldm_state_dict
    return config


_ATTENTION_PROJECTION: Final = re.compile(r"\.to_(q|k|v|out)\.weight$")

_LDM_UP: Final = re.compile(r"^up\.(\d+)\.")

_LDM_RENAMES: Final = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"^down\.(\d+)\.block\.", r"down_blocks.\1.resnets."),
        (r"^down\.(\d+)\.attn\.", r"down_blocks.\1.attentions."),
        (r"^down\.(\d+)\.downsample\.", r"down_blocks.\1.downsamplers.0."),
        (r"^(up_blocks\.\d+)\.block\.", r"\1.resnets."),
        (r"^(up_blocks\.\d+)\.attn\.", r"\1.attentions."),
        (r"^(up_blocks\.\d+)\.upsample\.", r"\1.upsamplers.0."),
        (r"^mid\.block_1\.", "mid_block.resnets.0."),
        (r"^mid\.block_2\.", "mid_block.resnets.1."),
        (r"^mid\.attn_1\.", "mid_block.attentions.0."),
        (r"^norm_out\.", "conv_norm_out."),
        (r"\.nin_shortcut\.", ".conv_shortcut."),
        (r"(attentions\.\d+)\.norm\.", r"\1.group_norm."),
        (r"(attentions\.\d+)\.(q|k|v)\.", r"\1.to_\2."),
        (r"(attentions\.\d+)\.proj_out\.", r"\1.to_out."),
    )
)
"""Latent-diffusion names to this module's, applied in order after ``up`` reindexing."""

_SUPPORTED_BLOCKS: Final = {
    "down_block_types": {"DownEncoderBlock2D", "AttnDownEncoderBlock2D"},
    "up_block_types": {"UpDecoderBlock2D", "AttnUpDecoderBlock2D"},
}


def _reject_unsupported(config: Mapping[str, object]) -> None:
    """Raise on a diffusers field this port would otherwise silently drop."""
    if config.get("_class_name") != "AutoencoderKL":
        raise ValueError(
            f"Expected _class_name 'AutoencoderKL', got {config.get('_class_name')!r}.",
        )
    if from_plain(config.get("act_fn"), str, default="silu") != "silu":
        raise ValueError("Only the swish (silu) AutoencoderKL is supported.")
    if not from_plain(config.get("mid_block_add_attention"), bool, default=True):
        raise ValueError("mid_block_add_attention=false is unsupported.")
    for name, supported in _SUPPORTED_BLOCKS.items():
        kinds = set(from_plain(config[name], list[str]))
        if kinds - supported:
            raise ValueError(f"Unsupported {name}: {sorted(kinds - supported)}.")


def _swish(x: Tensor) -> Tensor:
    """Return ``x * sigmoid(x)``."""
    # Not ``functional.silu``: its fused kernel rounds differently from the
    # weights' reference, which moves latents off the existing corpora.
    return x * torch.sigmoid(x)


def _conv(
    channels_in: int,
    channels_out: int,
    *,
    kernel_size: int,
    stride: int = 1,
) -> Conv2d:
    """Return a biased convolution, same-padded unless strided."""
    return Conv2d.Config(
        channels_in,
        channels_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=0 if stride > 1 else kernel_size // 2,
        bias=True,
    ).make()


def _norm(channels: int, *, num_groups: int, eps: float) -> GroupNorm2d:
    """Return the reference's affine GroupNorm."""
    return GroupNorm2d.Config(
        channels,
        num_groups=num_groups,
        eps=eps,
        elementwise_affine=True,
    ).make()


class _Resnet(nn.Module):
    """Two GroupNorm, swish, 3x3 conv layers, added to the (projected) input."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        *,
        num_groups: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(channels_in, num_groups=num_groups, eps=eps)
        self.conv1 = _conv(channels_in, channels_out, kernel_size=3)
        self.norm2 = _norm(channels_out, num_groups=num_groups, eps=eps)
        self.conv2 = _conv(channels_out, channels_out, kernel_size=3)
        self.conv_shortcut = (
            _conv(channels_in, channels_out, kernel_size=1)
            if channels_in != channels_out
            else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(_swish(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _Attention(nn.Module):
    """Single-head self-attention over the grid, added to its input."""

    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        eps: float,
        kernel: Makeable[AttentionKernel],
    ) -> None:
        super().__init__()
        self.group_norm = _norm(channels, num_groups=num_groups, eps=eps)
        self.to_q = _conv(channels, channels, kernel_size=1)
        self.to_k = _conv(channels, channels, kernel_size=1)
        self.to_v = _conv(channels, channels, kernel_size=1)
        self.to_out = _conv(channels, channels, kernel_size=1)
        self.kernel = kernel.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.group_norm(x)
        b, c, height, width = x.shape
        # ``[N, S, 1, C]`` views of the projections, which :class:`LdmAttention`
        # turns back into the reference's own operands without a copy.
        q, k, v = (
            projection(h).reshape(b, c, -1).permute(0, 2, 1).unsqueeze(2)
            for projection in (self.to_q, self.to_k, self.to_v)
        )
        attended = self.kernel(q, k, v).squeeze(2).permute(0, 2, 1)
        return x + self.to_out(attended.reshape(b, c, height, width))


class _Downsample(nn.Module):
    """Halve the grid: pad bottom and right by one, then a stride-2 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _conv(channels, channels, kernel_size=3, stride=2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # One-sided: the checkpoint was trained with it, and Conv2d's own padding
        # is symmetric.
        return self.conv(functional.pad(x, (0, 1, 0, 1)))


class _Upsample(nn.Module):
    """Double the grid by nearest neighbour, then a 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _conv(channels, channels, kernel_size=3)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(functional.interpolate(x, scale_factor=2.0, mode="nearest"))


class _Stage(nn.Module):
    """One grid size: residual blocks, each optionally followed by attention, then a resample."""

    def __init__(self) -> None:
        super().__init__()
        self.resnets: nn.ModuleList[_Resnet] = nn.ModuleList()
        self.attentions: nn.ModuleList[_Attention] = nn.ModuleList()
        self.downsamplers: nn.ModuleList[_Downsample] = nn.ModuleList()
        self.upsamplers: nn.ModuleList[_Upsample] = nn.ModuleList()

    @override
    def forward(self, h: Tensor) -> Tensor:
        for index, resnet in enumerate(self.resnets):
            h = resnet(h)
            if self.attentions:
                h = self.attentions[index](h)
        for downsample in self.downsamplers:
            h = downsample(h)
        for upsample in self.upsamplers:
            h = upsample(h)
        return h


class _Middle(nn.Module):
    """Residual block, attention, residual block, at the coarsest grid."""

    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        eps: float,
        kernel: Makeable[AttentionKernel],
    ) -> None:
        super().__init__()
        self.resnets: nn.ModuleList[_Resnet] = nn.ModuleList(
            _Resnet(channels, channels, num_groups=num_groups, eps=eps)
            for _ in range(2)
        )
        self.attentions: nn.ModuleList[_Attention] = nn.ModuleList(
            [_Attention(channels, num_groups=num_groups, eps=eps, kernel=kernel)],
        )

    @override
    def forward(self, h: Tensor) -> Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](h)))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Layers:
    """Price the reference's layers at one batch size, group count, kernel, and dtype."""

    batch_size: int
    num_groups: int
    attn_kernel: Makeable[AttentionKernel]
    dtype: torch.dtype | None

    def conv(
        self,
        channels_in: int,
        channels_out: int,
        *,
        side: int,
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
          stride: Stride.
          padding: Padding; ``None`` pads same.

        Returns:
          cost: Forward and adjoint.

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

    def norm(self, channels: int, *, side: int) -> Cost:
        """Cost the affine GroupNorm.

        Args:
          channels: Channels normalized.
          side: Grid side.

        Returns:
          cost: Forward and adjoint.

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

    def resnet(self, channels_in: int, channels_out: int, *, side: int) -> Cost:
        """Cost ``_Resnet``, its shortcut projection included.

        Args:
          channels_in: Input channels.
          channels_out: Output channels.
          side: Grid side.

        Returns:
          cost: Forward and adjoint.

        """
        elements_in = self.batch_size * channels_in * side * side
        elements_out = self.batch_size * channels_out * side * side
        block = (
            self.norm(channels_in, side=side)
            + _swish_cost(elements_in, dtype=self.dtype)
            + self.conv(channels_in, channels_out, side=side, kernel_size=3)
            + self.norm(channels_out, side=side)
            + _swish_cost(elements_out, dtype=self.dtype)
            + self.conv(channels_out, channels_out, side=side, kernel_size=3)
            + SUM(channels=elements_out, dtype=self.dtype)
        )
        if channels_in != channels_out:
            block += self.conv(channels_in, channels_out, side=side, kernel_size=1)
        return block

    def attention(self, channels: int, *, side: int) -> Cost:
        """Cost ``_Attention``: single-head attention over the grid.

        Args:
          channels: Channels.
          side: Grid side.

        Returns:
          cost: Forward and adjoint.

        """
        return (
            self.norm(channels, side=side)
            + self.conv(channels, channels, side=side, kernel_size=1).tile(4, copies=4)
            + cost(
                self.attn_kernel,
                seq_len=side * side,
                batch_size=self.batch_size,
                dtype=self.dtype,
                num_heads=1,
                channels_head=channels,
            )
            + SUM(channels=self.batch_size * channels * side * side, dtype=self.dtype)
        )

    def middle(self, channels: int, *, side: int) -> Cost:
        """Cost ``_Middle``: block, attention, block."""
        return self.resnet(channels, channels, side=side).tile(
            2,
            copies=2,
        ) + self.attention(channels, side=side)

    def head(self, channels_in: int, *, channels_out: int, side: int) -> Cost:
        """Cost ``conv_norm_out``, swish, and ``conv_out``."""
        return (
            self.norm(channels_in, side=side)
            + _swish_cost(self.batch_size * channels_in * side * side, dtype=self.dtype)
            + self.conv(channels_in, channels_out, side=side, kernel_size=3)
        )

    def downsample(self, channels: int, *, side: int) -> Cost:
        """Cost ``_Downsample``: the one-sided pad's traffic, then the strided conv.

        Args:
          channels: Channels.
          side: Input grid side.

        Returns:
          cost: Forward and adjoint.

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
            channels,
            side=side + 1,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def upsample(self, channels: int, *, side: int) -> Cost:
        """Cost ``_Upsample``: the interpolation's traffic, then the conv.

        Args:
          channels: Channels.
          side: Input grid side.

        Returns:
          cost: Forward and adjoint.

        """
        elements = self.batch_size * channels * side * side
        return traffic(
            "primal",
            kernel="elementwise",
            elements=5 * elements,
            dtype=self.dtype,
        ) + self.conv(channels, channels, side=2 * side, kernel_size=3)


def _swish_cost(elements: int, *, dtype: torch.dtype | None) -> Cost:
    """Cost ``_swish``: a sigmoid, then its product with the input."""
    return cost(sigmoid, channels=elements, dtype=dtype) + SUM(
        channels=elements,
        dtype=dtype,
    )
