"""INVAE: a 32-channel convolutional KL autoencoder, 16x downsampling.

Build the published model with ``INVAE.load()``, or parse a diffusers
``config.json`` with :meth:`INVAE.Config.from_hf`.

MIT license and attribution: ``LICENSES/REG``.

References:
  https://github.com/SwayStar123/REG/blob/3c51606c801dd9e87ee9ef778782766ab7c379ca/models/invae.py
  https://huggingface.co/REPA-E/e2e-invae-hf
    Leng et al. 2025. REPA-E: Unlocking VAE for end-to-end tuning with latent
    diffusion transformers.

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

from priml.cost import Cost, cost, elementwise_cost, traffic
from priml.model.ae.checkpoint import (
    CheckpointFile,
    HubFile,
    hub_or_local,
    read_json,
    read_state,
)
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.latent_norm import LatentNormalizer, ScaleLatents
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import Conv2d, conv_cost
from priml.model.norm import GroupNorm2d
from priml.model.swiglu import sigmoid


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


REPO: Final = "REPA-E/e2e-invae-hf"
"""The published diffusers checkpoint."""

REVISION: Final = "523ddf74b061c01ed0aebb7904c6bdb2f33e06d0"
"""The commit of :data:`REPO` the defaults pin."""


def e2e_invae_hf() -> HubFile.Config:
    """Return the pinned diffusers safetensors both halves read by default.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id=REPO,
        filename="diffusion_pytorch_model.safetensors",
        revision=REVISION,
        sha256="6b1f754e414a841893fd242ca233c5aeec4bbdb7d2eeab2f7956adf48742e500",
    )


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

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=e2e_invae_hf,
        )
        """Diffusers weights; only ``encoder.*`` and ``quant_conv.*`` are read.

        ``None`` keeps the random initialization.
        """

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a square ``input_grid`` image, ``quant_conv`` included.

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
            moments = 2 * self.channels_latent
            return (
                total
                + layers.middle(width_in, side=side)
                + layers.head(width_in, channels_out=moments, side=side)
                + layers.conv(moments, moments, side=side, kernel_size=1)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        widths, groups, eps = config.channels_hidden, config.num_groups, config.eps
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
                        _Attention(width, num_groups=groups, eps=eps),
                    )
            if stage != len(widths) - 1:
                level.downsamplers.append(_Downsample(width))
            self.down_blocks.append(level)
        self.mid_block = _Middle(width_in, num_groups=groups, eps=eps)
        self.conv_norm_out = _norm(width_in, num_groups=groups, eps=eps)
        moments = 2 * config.channels_latent
        self.conv_out = _conv(width_in, moments, kernel_size=3)
        self.quant_conv = _conv(moments, moments, kernel_size=1)
        if config.checkpoint is not None:
            path = config.checkpoint.make().path()
            state = read_state(path, prefixes=("encoder.", "quant_conv."))
            self.load_state_dict(remap_hf_state_dict(state, half="encoder"))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[N, C_in, H, W]`` pixels in ``[-1, 1]`` to ``[N, 2 C, h, w]`` moments."""
        h = self.conv_in(x)
        for level in self.down_blocks:
            h = level(h)
        h = self.mid_block(h)
        h = self.conv_out(_swish(self.conv_norm_out(h)))
        return self.quant_conv(h)


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

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=e2e_invae_hf,
        )
        """Diffusers weights; only ``decoder.*`` and ``post_quant_conv.*`` are read.

        ``None`` keeps the random initialization.
        """

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a square ``input_grid`` latent, ``post_quant_conv`` included.

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
                dtype=dtype,
            )
            z, side = self.channels_latent, input_grid
            width_in = self.channels_hidden[-1]
            total = (
                layers.conv(z, z, side=side, kernel_size=1)
                + layers.conv(z, width_in, side=side, kernel_size=3)
                + layers.middle(width_in, side=side)
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
        z = config.channels_latent
        self.post_quant_conv = _conv(z, z, kernel_size=1)
        width_in = widths[-1]
        self.conv_in = _conv(z, width_in, kernel_size=3)
        self.mid_block = _Middle(width_in, num_groups=groups, eps=eps)
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
                        _Attention(width, num_groups=groups, eps=eps),
                    )
            if stage != 0:
                level.upsamplers.append(_Upsample(width))
            self.up_blocks.append(level)
        self.conv_norm_out = _norm(width_in, num_groups=groups, eps=eps)
        self.conv_out = _conv(width_in, config.channels_out, kernel_size=3)
        if config.checkpoint is not None:
            path = config.checkpoint.make().path()
            state = read_state(path, prefixes=("decoder.", "post_quant_conv."))
            self.load_state_dict(remap_hf_state_dict(state, half="decoder"))

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Map ``[N, C, h, w]`` latents to ``[N, C_out, H, W]`` pixels in ``[-1, 1]``."""
        h = self.mid_block(self.conv_in(self.post_quant_conv(z)))
        for level in self.up_blocks:
            h = level(h)
        return self.conv_out(_swish(self.conv_norm_out(h)))


class INVAE(nn.Module):
    """INVAE as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a diagonal Gaussian; call ``sample()`` or ``mode()`` on
    it. The model is frozen and always in evaluation mode.
    """

    class Config(Fig["INVAE"]):
        """The halves, the image side, and the published latent normalizer."""

        encoder: Encoder.Config | None = field(default_factory=Encoder.Config)
        """Pixels to posterior; ``None`` builds and loads no encoder."""

        decoder: Decoder.Config | None = field(default_factory=Decoder.Config)
        """Latents to pixels; ``None`` builds and loads no decoder."""

        image_size: int = 256
        """Side of the square image the checkpoint was trained at."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: ScaleLatents.Config(scale=0.3099),
        )
        """The scale the REG corpora train with; the autoencoder never applies it."""

        @classmethod
        def from_hf(
            cls,
            config: Mapping[str, object],
            *,
            checkpoint: Makeable[CheckpointFile] | None = None,
        ) -> Self:
            """Parse a diffusers ``AutoencoderKL`` ``config.json``.

            ``sample_size``, ``latents_mean``, and ``latents_std`` are not read.

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
            return cls(encoder=encoder, decoder=decoder)

        def latent_shape(self) -> tuple[int, int, int]:
            """Return ``(channels_latent, side, side)`` at ``image_size``."""
            half = self.encoder or self.decoder
            if half is None:
                raise ValueError("INVAE.Config needs an encoder or a decoder.")
            side = self.image_size >> (len(half.channels_hidden) - 1)
            return half.channels_latent, side, side

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
                raise ValueError("INVAE.Config.cost prices both halves; one is None.")
            channels, side, _ = self.latent_shape()
            area = batch_size * self.image_size**2
            pixels_in = area * self.encoder.channels_in
            pixels_out = area * self.decoder.channels_out
            latents = batch_size * channels * side * side
            f32 = torch.float32
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
                        "input_grid": side,
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                # ``image.float() / 127.5 - 1`` in.
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=pixels_in,
                    dtype=torch.uint8,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=pixels_in,
                    dtype=f32,
                )
                + _pointwise(pixels_in, flops=1, dtype=f32).tile(2)
                # Clamp the log-variance; ``exp(0.5 * logvar)``.
                + _pointwise(latents, flops=2, dtype=dtype)
                + _pointwise(latents, flops=1, dtype=dtype).tile(2)
                # ``mean + std * noise``: the noise written, a product, a sum.
                + traffic("primal", kernel="elementwise", elements=latents, dtype=dtype)
                + _pointwise(latents, flops=1, inputs=2, dtype=dtype).tile(2)
                # ``(x + 1) / 2`` and a two-sided clamp out.
                + _pointwise(pixels_out, flops=1, dtype=dtype).tile(2)
                + _pointwise(pixels_out, flops=2, dtype=dtype)
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
            raise ValueError("INVAE needs an encoder or a decoder.")
        downsample = 1 << (len(halves[0].channels_hidden) - 1)
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
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def load(
        cls,
        path_or_repo: Path | str = REPO,
        *,
        revision: str = REVISION,
    ) -> INVAE:
        """Build from a diffusers directory or Hub repository, weights included.

        Args:
          path_or_repo: Local directory holding ``config.json`` and
            ``diffusion_pytorch_model.safetensors``, or a Hub repository.
          revision: Commit to read a Hub repository at.

        Returns:
          model: Frozen INVAE.

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
          posterior: Diagonal Gaussian over ``[..., channels_latent, h, w]``.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder.

        """
        if self.encoder is None:
            raise ValueError("This INVAE was built without an encoder.")
        # A float image in [0, 1] would otherwise encode as near-black.
        if x.dtype != torch.uint8:
            raise TypeError(f"INVAE encodes uint8 RGB in [0, 255]; got {x.dtype}.")
        # ``x / 127.5 - 1``, not ``rgb2float``'s ``(x - 127.5) / 127.5``: the two
        # round 128 of the 256 levels differently, and the weights saw this one.
        pixels = x.reshape(-1, *x.shape[-3:]).float() / 127.5 - 1
        moments = self.encoder(pixels)
        moments = moments.reshape(*x.shape[:-3], *moments.shape[1:])
        mean, logvar = moments.chunk(2, dim=-3)
        return DiagonalGaussian(mean, logvar)

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels_latent, h, w]`` raw latents.

        Returns:
          image: ``[..., 3, H, W]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder.

        """
        if self.decoder is None:
            raise ValueError("This INVAE was built without a decoder.")
        decoded = self.decoder(z.reshape(-1, *z.shape[-3:]))
        decoded = decoded.reshape(*z.shape[:-3], *decoded.shape[1:])
        return ((decoded + 1) / 2).clamp(0, 1)


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


_ATTENTION_PROJECTION: Final = re.compile(r"\.to_(q|k|v|out)\.weight$")

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
    for name in ("mid_block_add_attention", "use_quant_conv", "use_post_quant_conv"):
        if not from_plain(config.get(name), bool, default=True):
            raise ValueError(f"{name}=false is unsupported.")
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

    def __init__(self, channels: int, *, num_groups: int, eps: float) -> None:
        super().__init__()
        self.group_norm = _norm(channels, num_groups=num_groups, eps=eps)
        self.to_q = _conv(channels, channels, kernel_size=1)
        self.to_k = _conv(channels, channels, kernel_size=1)
        self.to_v = _conv(channels, channels, kernel_size=1)
        self.to_out = _conv(channels, channels, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.group_norm(x)
        q, k, v = self.to_q(h), self.to_k(h), self.to_v(h)
        b, c, height, width = q.shape
        # The reference's products and its scale, a multiply by ``c ** -0.5``:
        # ``scaled_dot_product_attention`` divides and rounds differently.
        scores = torch.bmm(q.reshape(b, c, -1).permute(0, 2, 1), k.reshape(b, c, -1))
        weights = functional.softmax(scores * cast(float, c ** (-0.5)), dim=2)
        attended = torch.bmm(v.reshape(b, c, -1), weights.permute(0, 2, 1))
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

    def __init__(self, channels: int, *, num_groups: int, eps: float) -> None:
        super().__init__()
        self.resnets: nn.ModuleList[_Resnet] = nn.ModuleList(
            _Resnet(channels, channels, num_groups=num_groups, eps=eps)
            for _ in range(2)
        )
        self.attentions: nn.ModuleList[_Attention] = nn.ModuleList(
            [_Attention(channels, num_groups=num_groups, eps=eps)],
        )

    @override
    def forward(self, h: Tensor) -> Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](h)))


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
            + _pointwise(elements_out, flops=1, inputs=2, dtype=self.dtype)
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
