"""VTP: MiniMax's visual tokenizer as a 64-channel autoencoder, 16x downsampling.

A DINOv3-style ViT encoder with a linear bottleneck, and a ViT pixel decoder.
Build a published model with ``VTP.load()``, or parse its ``config.json`` with
:meth:`VTP.Config.from_hf`.

MIT license with an attribution clause: ``LICENSES/VTP``.

References:
  https://github.com/MiniMax-AI/VTP/tree/5ce1eb67010fff3c1eed483352483be6a1838556/vtp/models
  https://huggingface.co/MiniMaxAI/VTP-Large-f16d64
    Yao et al. 2025. Towards Scalable Pre-training of Visual Tokenizers for
    Generation.

"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import field
from typing import TYPE_CHECKING, Final, Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.math.basic import ceil_multiple
from priml.math.pixel import rgb2float
from priml.model.ae.checkpoint import (
    CheckpointFile,
    HubFile,
    UrlFile,
    hub_or_local,
    read_json,
    read_state,
)
from priml.model.ae.distribution import PointMass
from priml.model.ae.latent_norm import ChannelLatentStats, LatentNormalizer
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import Conv2d, conv_cost
from priml.model.linear import Linear
from priml.model.norm import LayerNorm, RMSNorm
from priml.model.swiglu import SwiGLU


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from priml.model.custom_types import TensorModule


_LATENT_STATS_ROOT: Final = (
    "https://raw.githubusercontent.com/MiniMax-AI/VTP/"
    "5ce1eb67010fff3c1eed483352483be6a1838556/generation/latent_stats"
)

PUBLISHED: Final = {
    "small": (
        "MiniMaxAI/VTP-Small-f16d64",
        "378967941e66f7f09a9e10218aa8710a930cc635",
        "5442083e078aadc65e51c7893d10177e360139866d4e09ab955b1c7b563eeaee",
        "vtp_s",
        "9f4bb851a6226843c0bdb9b25330faa2a8428e5c10a77f79a974f145cf848eff",
    ),
    "base": (
        "MiniMaxAI/VTP-Base-f16d64",
        "21fe3a9cb53f01a2c7f3b622efdc8d2534a7c79d",
        "c7fdbb1507cecbcbb35eab4ea9fcbca8dc80b5e0d64cfe0c37b7baacdbb0fa05",
        "vtp_b",
        "f7c030310f3eeaa38d2d0838f38eeab1d9922fcaf183a0fe8dd1f1287366c4a7",
    ),
    "large": (
        "MiniMaxAI/VTP-Large-f16d64",
        "d1726ca9abd5ed0b3ad938c508c635d8b1a51a31",
        "de5df2006083a9536c4d3ea36c6ae2181ec604e06550f1b2d7cece3b16aac32f",
        "vtp_l",
        "0ad1fadbfb8959f3a93147a4db668b6d704ae36ff320bfad53cd3d568de7b0fd",
    ),
}
"""Size -> (repository, revision, weights sha256, stats directory, stats sha256)."""


def published_weights(size: str) -> HubFile.Config:
    """Return the pinned ``model.safetensors`` of a published size.

    Args:
      size: ``"small"``, ``"base"``, or ``"large"``.

    Returns:
      source: The pinned Hub file; both halves read it.

    """
    repo, revision, sha256, _, _ = PUBLISHED[size]
    return HubFile.Config(
        repo_id=repo,
        filename="model.safetensors",
        revision=revision,
        sha256=sha256,
    )


def published_latent_norm(size: str) -> ChannelLatentStats.Config:
    """Return the per-channel statistics published with a size.

    Args:
      size: ``"small"``, ``"base"``, or ``"large"``.

    Returns:
      config: The normalizer.

    """
    _, _, _, directory, sha256 = PUBLISHED[size]
    return ChannelLatentStats.Config(
        stats=UrlFile.Config(
            url=f"{_LATENT_STATS_ROOT}/{directory}/latents_stats.pt",
            sha256=sha256,
        ),
    )


class Encoder(nn.Module):
    """ImageNet-normalized pixels to a bottlenecked latent per patch.

    A CLS token joins the patch tokens through RMSNorm SwiGLU blocks with axial
    rotary attention and a final norm; each patch token is then projected,
    bias-free, to ``channels_out``.
    """

    class Config(Fig["Encoder"]):
        """Architecture and weights; the defaults are VTP-Large."""

        patch_size: int = 16
        """Pixels per patch side; the latent grid is the image divided by it."""

        channels_hidden: int = 1024
        """Token width."""

        num_layers: int = 24
        """Transformer blocks."""

        heads: int = 16
        """Attention heads; ``channels_hidden`` must be divisible by ``4 * heads``."""

        expansion: float = 4.0
        """Nominal MLP width over ``channels_hidden``; SwiGLU keeps 2/3, rounded up to 8."""

        channels_out: int = 64
        """Latent channels."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: published_weights("large"),
        )
        """Published safetensors; only ``trunk.*`` is read. ``None`` keeps the random init."""

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            dtype_autocast: torch.dtype | None = None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward and backward on ``[batch_size, 3, S, S]``.

            Under autocast the products, attention, and MLP run in
            ``dtype_autocast``; the CLS token, residual stream, and norms stay
            in ``dtype``.

            Args:
              input_grid: Image side ``S``.
              batch_size: Images per invocation.
              dtype: The weights' dtype; ``None`` is torch's default.
              dtype_autocast: The autocast dtype; ``None`` runs all in ``dtype``.
              **kwargs: The open bus, unread.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            Raises:
              ValueError: ``input_grid`` is not a multiple of ``patch_size``.

            """
            del kwargs
            if input_grid % self.patch_size:
                raise ValueError(
                    f"input_grid {input_grid} must be divisible by "
                    f"patch_size {self.patch_size}.",
                )
            side = input_grid // self.patch_size
            width = self.channels_hidden
            compute = dtype if dtype_autocast is None else dtype_autocast
            patch_embed = conv_cost(
                channels_in=3,
                channels_out=width,
                kernel_size=self.patch_size,
                ndim=2,
                groups=1,
                bias=True,
                input_grid=(input_grid, input_grid),
                batch_size=batch_size,
                stride=self.patch_size,
                padding=0,
                dtype=compute,
            )
            # ``cls_token + 0 * mask_token``: two maps over one row owning both
            # tokens. Back, the mask token's gradient is the zero product and the
            # class token's is the expanded batch summed.
            class_token = elementwise_cost(
                primal=2 * width,
                adjoint=width,
                channels=width,
                params=2 * width,
                inputs=1,
                outputs=2,
                adjoint_inputs=1,
                adjoint_outputs=1,
                dtype=dtype,
            ) + reduction_cost(
                input_elements=batch_size * width,
                output_groups=width,
                dtype=dtype,
                phase="adjoint",
            )
            blocks = _transformer_cost(
                norm_config=RMSNorm.Config(width, eps=1e-5, elementwise_affine=True),
                channels=width,
                heads=self.heads,
                expansion=self.expansion,
                num_layers=self.num_layers,
                side=side,
                prefix=1,
                batch_size=batch_size,
                dtype=compute,
                dtype_stream=dtype,
            )
            bottleneck = matmul_cost(
                channels_in=width,
                channels_out=self.channels_out,
                rows=batch_size * side * side,
                dtype=compute,
            )
            return patch_embed + class_token + blocks + bottleneck

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_out == config.channels_hidden:
            raise ValueError(
                f"channels_out {config.channels_out} must differ from "
                f"channels_hidden {config.channels_hidden}.",
            )
        width = config.channels_hidden
        self.patch_size = config.patch_size
        self.patch_embed = Conv2d.Config(
            3,
            width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
        ).make()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, width))
        self.rope = _Rotary(width, num_heads=config.heads)
        self.blocks = nn.ModuleList(
            _Block(
                width,
                num_heads=config.heads,
                expansion=config.expansion,
                norm=lambda: RMSNorm.Config(
                    width,
                    eps=1e-5,
                    elementwise_affine=True,
                ).make(),
            )
            for _ in range(config.num_layers)
        )
        self.norm = RMSNorm.Config(width, eps=1e-5, elementwise_affine=True).make()
        self.bottleneck = Linear.Config(width, config.channels_out).make()
        if config.checkpoint is not None:
            state = read_state(config.checkpoint.make().path(), prefixes=("trunk.",))
            self.load_state_dict(remap_hf_state_dict(state, half="encoder"))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[N, 3, H, W]`` normalized pixels to ``[N, channels_out, h, w]``."""
        tokens = self.patch_embed(x)
        batch, width, height, grid_width = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        # ``0 * mask_token`` keeps an inf or NaN mask token visible.
        cls_token = self.cls_token + 0 * self.mask_token
        tokens = torch.cat([cls_token.expand(batch, -1, -1), tokens], dim=1)
        rope = self.rope(height=height, width=grid_width)
        for block in self.blocks:
            tokens = block(tokens, rope=rope)
        patches = self.norm(tokens)[:, 1:]
        latent = self.bottleneck(patches.reshape(-1, width))
        latent = latent.reshape(batch, height * grid_width, -1)
        return latent.transpose(1, 2).reshape(batch, -1, height, grid_width)


class Decoder(nn.Module):
    """Latents to ImageNet-normalized pixels.

    A 1x1 convolution widens the latent, LayerNorm SwiGLU blocks with axial
    rotary attention mix it, and a 1x1 convolution plus pixel shuffle unfold
    each token into an ``upscale_factor``-square RGB patch.
    """

    class Config(Fig["Decoder"]):
        """Architecture and weights; the defaults are VTP-Large."""

        channels_in: int = 64
        """Latent channels."""

        channels_hidden: int = 1024
        """Token width."""

        num_layers: int = 24
        """Transformer blocks."""

        heads: int = 16
        """Attention heads; ``channels_hidden`` must be divisible by ``4 * heads``."""

        expansion: float = 4.0
        """Nominal MLP width over ``channels_hidden``; SwiGLU keeps 2/3, rounded up to 8."""

        upscale_factor: int = 16
        """Pixels per latent side."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: published_weights("large"),
        )
        """Published safetensors; only ``pixel_decoder.*`` is read. ``None`` keeps the random init."""

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward and backward on ``[batch_size, channels_in, s, s]``.

            Args:
              input_grid: Latent side ``s``.
              batch_size: Latents per invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del kwargs
            width = self.channels_hidden
            grid = (input_grid, input_grid)
            proj_in = conv_cost(
                channels_in=self.channels_in,
                channels_out=width,
                kernel_size=1,
                ndim=2,
                groups=1,
                bias=True,
                input_grid=grid,
                batch_size=batch_size,
                padding=0,
                dtype=dtype,
            )
            blocks = _transformer_cost(
                norm_config=LayerNorm.Config(width, eps=1e-6, elementwise_affine=True),
                channels=width,
                heads=self.heads,
                expansion=self.expansion,
                num_layers=self.num_layers,
                side=input_grid,
                prefix=0,
                batch_size=batch_size,
                dtype=dtype,
                dtype_stream=dtype,
            )
            proj_out = conv_cost(
                channels_in=width,
                channels_out=3 * self.upscale_factor**2,
                kernel_size=1,
                ndim=2,
                groups=1,
                bias=True,
                input_grid=grid,
                batch_size=batch_size,
                padding=0,
                dtype=dtype,
            )
            return proj_in + blocks + proj_out

    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.channels_hidden
        self.proj_in = Conv2d.Config(
            config.channels_in,
            width,
            kernel_size=1,
            padding=0,
            bias=True,
        ).make()
        self.rope = _Rotary(width, num_heads=config.heads)
        self.blocks = nn.ModuleList(
            _Block(
                width,
                num_heads=config.heads,
                expansion=config.expansion,
                norm=lambda: LayerNorm.Config(
                    width,
                    eps=1e-6,
                    elementwise_affine=True,
                ).make(),
            )
            for _ in range(config.num_layers)
        )
        self.norm = LayerNorm.Config(width, eps=1e-6, elementwise_affine=True).make()
        self.proj_out = Conv2d.Config(
            width,
            3 * config.upscale_factor**2,
            kernel_size=1,
            padding=0,
            bias=True,
        ).make()
        self.pixel_shuffle = nn.PixelShuffle(config.upscale_factor)
        if config.checkpoint is not None:
            state = read_state(
                config.checkpoint.make().path(),
                prefixes=("pixel_decoder.",),
            )
            self.load_state_dict(remap_hf_state_dict(state, half="decoder"))

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Map ``[N, channels_in, h, w]`` latents to ``[N, 3, h * up, w * up]`` pixels."""
        batch, _, height, width = z.shape
        tokens = self.proj_in(z).flatten(2).transpose(1, 2)
        rope = self.rope(height=height, width=width)
        for block in self.blocks:
            tokens = block(tokens, rope=rope)
        tokens = self.norm(tokens)
        grid = tokens.transpose(1, 2).reshape(batch, -1, height, width)
        return self.pixel_shuffle(self.proj_out(grid))


class VTP(nn.Module):
    """VTP as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a point mass; ``decode`` returns pixels in ``[0, 1]``.
    The model is frozen and always in evaluation mode.
    """

    class Config(Fig["VTP"]):
        """The halves, the image side, pixel statistics, precision, and latent norm."""

        encoder: Encoder.Config | None = field(default_factory=Encoder.Config)
        """Pixels to latents; ``None`` builds and loads no encoder."""

        decoder: Decoder.Config | None = field(default_factory=Decoder.Config)
        """Latents to pixels; ``None`` builds and loads no decoder."""

        image_size: int = 256
        """Side of the square image the checkpoint was trained at."""

        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """Per-channel RGB mean subtracted before encoding."""

        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """Per-channel RGB standard deviation divided out before encoding."""

        dtype_autocast: torch.dtype | None = None
        """Autocast dtype for the encoder; ``None`` encodes in the weights' dtype.

        Set, ``encode`` returns latents in this dtype, and ``decode`` casts
        them back to the weights' dtype.
        """

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: published_latent_norm("large"),
        )
        """Per-channel standardization; the autoencoder itself never applies it."""

        @classmethod
        def from_hf(
            cls,
            config: Mapping[str, object],
            *,
            checkpoint: Makeable[CheckpointFile] | None = None,
        ) -> Self:
            """Parse a published VTP ``config.json``.

            Args:
              config: The ``config.json`` object.
              checkpoint: Weights both halves read; ``None`` keeps a random init.

            Returns:
              config: The architecture.

            Raises:
              ValueError: A field names an architecture this port does not build.

            """
            _reject_unsupported(config)
            encoder = Encoder.Config(
                patch_size=from_plain(config["vision_patch_size"], int),
                channels_hidden=from_plain(config["vision_embed_dim"], int),
                num_layers=from_plain(config["vision_depth"], int),
                heads=from_plain(config["vision_num_heads"], int),
                expansion=from_plain(config["vision_mlp_ratio"], float, strict=False),
                channels_out=from_plain(config["vision_feature_bottleneck"], int),
                checkpoint=checkpoint,
            )
            decoder = Decoder.Config(
                channels_in=encoder.channels_out,
                channels_hidden=from_plain(config["decoder_embed_dim"], int),
                num_layers=from_plain(config["decoder_depth"], int),
                heads=from_plain(config["decoder_num_heads"], int),
                expansion=from_plain(
                    config.get("decoder_ffn_ratio", 4.0),
                    float,
                    strict=False,
                ),
                upscale_factor=encoder.patch_size,
                checkpoint=checkpoint,
            )
            parsed = cls(
                encoder=encoder,
                decoder=decoder,
                image_size=from_plain(config["image_size"], int),
            )
            # Each published size has its own statistics; match by architecture. No
            # statistics exist for any other, so its normalizer refuses to build
            # rather than standardize with another size's.
            parsed.latent_norm = ChannelLatentStats.Config()
            for size in PUBLISHED:
                published = _published(size)
                if _architecture(published) == _architecture(parsed):
                    parsed.latent_norm = published.latent_norm
                    break
            return parsed

        def latent_shape(self) -> tuple[int, int, int]:
            """Return the latent's shape at ``image_size``.

            Returns:
              shape: ``(channels, side, side)``.

            """
            if self.encoder is not None:
                side = self.image_size // self.encoder.patch_size
                return self.encoder.channels_out, side, side
            if self.decoder is None:
                raise ValueError("VTP.Config needs an encoder or a decoder.")
            side = self.image_size // self.decoder.upscale_factor
            return self.decoder.channels_in, side, side

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one ``decode(encode(image).mode())`` round trip at ``image_size``.

            Forward only; the parameters are still owned.

            Args:
              batch_size: Images per invocation.
              dtype: The weights' dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to both halves.

            Returns:
              cost: Primal integer FLOPs and logical bytes.

            """
            if self.encoder is None or self.decoder is None:
                raise ValueError("VTP.Config.cost prices both halves; one is None.")
            channels, side, _ = self.latent_shape()
            full = cost(
                self.encoder,
                **{
                    **kwargs,
                    "input_grid": self.image_size,
                    "batch_size": batch_size,
                    "dtype": dtype,
                    "dtype_autocast": self.dtype_autocast,
                },
            ) + cost(
                self.decoder,
                **{
                    **kwargs,
                    "input_grid": side,
                    "batch_size": batch_size,
                    "dtype": dtype,
                },
            )
            autocast = self.dtype_autocast
            if autocast is not None and autocast != resolve_dtype(dtype):
                latents = batch_size * channels * side * side
                full += traffic(
                    "primal",
                    kernel="elementwise",
                    elements=latents,
                    dtype=autocast,
                ) + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=latents,
                    dtype=dtype,
                )
            pixels = batch_size * 3 * self.image_size**2
            stats = len(self.pixel_mean)
            # Each map reads the image and writes it once; the subtracted or divided
            # maps also read one statistic per channel.
            pixel_maps = (
                traffic(
                    "primal",
                    kernel="elementwise",
                    elements=pixels,
                    dtype=torch.uint8,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=7 * pixels + 2 * stats,
                    flops=3 * pixels,
                    dtype=torch.float32,
                )
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=6 * pixels + 2 * stats,
                    flops=4 * pixels,
                    dtype=torch.float32,
                )
            )
            return Cost(
                cells=(full.only("primal") + pixel_maps).cells,
                params=full.params,
                params_active=full.params_active,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.encoder is None and config.decoder is None:
            raise ValueError("VTP needs an encoder or a decoder.")
        stats = (*config.pixel_mean, *config.pixel_std)
        if any(math.isnan(v) or math.isinf(v) for v in stats) or 0 in config.pixel_std:
            raise ValueError(
                "VTP needs finite pixel statistics and a nonzero std; got mean "
                f"{config.pixel_mean}, std {config.pixel_std}.",
            )
        if config.encoder is not None:
            patch = config.encoder.patch_size
        else:
            if config.decoder is None:
                raise ValueError("Expected config.decoder is not None.")
            patch = config.decoder.upscale_factor
        if config.image_size % patch:
            raise ValueError(
                f"image_size {config.image_size} must be divisible by patch_size {patch}.",
            )
        if config.encoder is not None and config.decoder is not None:
            encoder, decoder = config.encoder, config.decoder
            if (decoder.channels_in, decoder.upscale_factor) != (
                encoder.channels_out,
                encoder.patch_size,
            ):
                raise ValueError(
                    f"The decoder reads {decoder.channels_in} channels at upscale "
                    f"{decoder.upscale_factor}; the encoder writes "
                    f"{encoder.channels_out} at patch {encoder.patch_size}.",
                )
        self.patch_size = patch
        self.dtype_autocast = config.dtype_autocast
        self.encoder = None if config.encoder is None else config.encoder.make()
        self.decoder = None if config.decoder is None else config.decoder.make()
        mean = torch.tensor(config.pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(config.pixel_std, dtype=torch.float32).view(-1, 1, 1)
        # Inverted as ``(x - (-m / s)) / (1 / s)``, the published evaluation's
        # inverse Normalize: ``x * s + m`` differs in the last bit.
        inverse_mean = [
            -m / s for m, s in zip(config.pixel_mean, config.pixel_std, strict=True)
        ]
        inverse_std = [1 / s for s in config.pixel_std]
        self.pixel_mean: Tensor
        self.pixel_std: Tensor
        self.inverse_mean: Tensor
        self.inverse_std: Tensor
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)
        self.register_buffer(
            "inverse_mean",
            torch.tensor(inverse_mean, dtype=torch.float32).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "inverse_std",
            torch.tensor(inverse_std, dtype=torch.float32).view(-1, 1, 1),
            persistent=False,
        )
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def load(
        cls,
        path_or_repo: Path | str = PUBLISHED["large"][0],
        *,
        revision: str = PUBLISHED["large"][1],
    ) -> VTP:
        """Build from a VTP directory or Hub repository, weights included.

        Args:
          path_or_repo: Local directory holding ``config.json`` and
            ``model.safetensors``, or a Hub repository.
          revision: Commit to read a Hub repository at.

        Returns:
          model: Frozen VTP.

        """
        return cls.Config.from_hf(
            read_json(hub_or_local(path_or_repo, "config.json", revision=revision)),
            checkpoint=hub_or_local(
                path_or_repo,
                "model.safetensors",
                revision=revision,
            ),
        ).make()

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, x: Tensor, /) -> PointMass:
        """Return the bottleneck latents of uint8 images, as a point mass.

        Args:
          x: ``[..., 3, H, W]`` uint8 RGB, ``H`` and ``W`` multiples of the patch.

        Returns:
          latent: Point mass on ``[..., channels, H / patch, W / patch]``, in
            ``dtype_autocast`` when set.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder, or a side is
            not a multiple of the patch.

        """
        if self.encoder is None:
            raise ValueError("This VTP was built without an encoder.")
        # A float image in [0, 1] would otherwise encode as near-black.
        if x.dtype != torch.uint8:
            raise TypeError(f"VTP encodes uint8 RGB in [0, 255]; got {x.dtype}.")
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Image sides {height}x{width} must be multiples of "
                f"patch_size {self.patch_size}.",
            )
        unit = rgb2float(
            x.reshape(-1, *x.shape[-3:]),
            float_dtype=torch.float32,
            unit_interval=True,
        )
        pixels = (unit - self.pixel_mean) / self.pixel_std
        autocast = (
            nullcontext()
            if self.dtype_autocast is None
            else torch.autocast(x.device.type, dtype=self.dtype_autocast)
        )
        with autocast:
            latent = self.encoder(pixels)
        return PointMass(latent.reshape(*x.shape[:-3], *latent.shape[1:]))

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels, h, w]`` raw latents in any floating dtype.

        Returns:
          image: ``[..., 3, h * patch, w * patch]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder.

        """
        if self.decoder is None:
            raise ValueError("This VTP was built without a decoder.")
        flat = z.reshape(-1, *z.shape[-3:]).to(self.decoder.proj_in.weight.dtype)
        pixels = self.decoder(flat)
        image = ((pixels - self.inverse_mean) / self.inverse_std).clamp(0, 1)
        return image.reshape(*z.shape[:-3], *image.shape[1:])


def vtp_small() -> VTP.Config:
    """Return VTP-Small-f16d64: 384 wide, 12 blocks, 6 heads in each half."""
    return _published("small")


def vtp_base() -> VTP.Config:
    """Return VTP-Base-f16d64: 768 wide, 12 blocks, 12 heads in each half."""
    return _published("base")


def vtp_large() -> VTP.Config:
    """Return VTP-Large-f16d64, the :class:`VTP.Config` defaults: 1024 wide, 24 blocks."""
    return _published("large")


def remap_hf_state_dict(state: Mapping[str, Tensor], *, half: str) -> dict[str, Tensor]:
    """Rename one half's published tensors to this module's names.

    The text tower and CLIP projections the published files also carry are
    ignored; any other unknown key is refused.

    Args:
      state: Published tensors; keys outside ``half`` are ignored.
      half: ``"encoder"`` or ``"decoder"``.

    Returns:
      state: This half's ``state_dict``.

    Raises:
      ValueError: A tensor in ``half`` this module has no place for.

    """
    prefix = {"encoder": "trunk.", "decoder": "pixel_decoder."}[half]
    remapped: dict[str, Tensor] = {}
    unknown: list[str] = []
    for key, tensor in state.items():
        if not key.startswith(prefix):
            continue
        name = key.removeprefix(prefix)
        for published, ours in _RENAMES:
            if name.startswith(published):
                name = ours + name.removeprefix(published)
                break
        if name.endswith(
            (".mlp.w1.weight", ".mlp.w1.bias", ".mlp.w2.weight", ".mlp.w2.bias"),
        ):
            continue
        if ".mlp.w3." in name:
            name = name.replace(".mlp.w3.", ".mlp.down_proj.")
        elif name == "rope.periods" or name.startswith(_KNOWN):
            pass
        else:
            unknown.append(key)
            continue
        remapped[name] = tensor
    for key in state:
        if key.startswith(prefix) and ".mlp.w1." in key:
            base = key.removeprefix(prefix).rsplit(".mlp.w1.", 1)[0]
            kind = key.rsplit(".", 1)[1]
            # The published gate ``w1`` and value ``w2``, fused into one projection.
            remapped[f"{base}.mlp.up_proj.{kind}"] = torch.cat(
                [state[key], state[key.replace(".mlp.w1.", ".mlp.w2.")]],
            )
    if unknown:
        raise ValueError(f"No {half} parameter takes {sorted(unknown)}.")
    return remapped


_RENAMES: Final = (
    ("rope_embed.periods", "rope.periods"),
    ("patch_embed.proj.", "patch_embed."),
    ("feature_bottleneck.", "bottleneck."),
)

_KNOWN: Final = (
    "cls_token",
    "mask_token",
    "patch_embed.",
    "blocks.",
    "norm.",
    "bottleneck.",
    "proj_in.",
    "proj_out.",
)


def _published(size: str) -> VTP.Config:
    """Return a published size's architecture, weights, and statistics."""
    width, depth, heads = {
        "small": (384, 12, 6),
        "base": (768, 12, 12),
        "large": (1024, 24, 16),
    }[size]
    return VTP.Config(
        encoder=Encoder.Config(
            channels_hidden=width,
            num_layers=depth,
            heads=heads,
            checkpoint=published_weights(size),
        ),
        decoder=Decoder.Config(
            channels_hidden=width,
            num_layers=depth,
            heads=heads,
            checkpoint=published_weights(size),
        ),
        latent_norm=published_latent_norm(size),
    )


def _architecture(config: VTP.Config) -> tuple[object, ...]:
    """Return what distinguishes one published size from another, weights aside."""
    return (
        *(
            (half.channels_hidden, half.num_layers, half.heads, half.expansion)
            for half in (config.encoder, config.decoder)
            if half is not None
        ),
        config.image_size,
    )


def _reject_unsupported(config: Mapping[str, object]) -> None:
    """Raise on a published field this port would otherwise silently drop."""
    if config.get("model_type") != "vtp":
        raise ValueError(
            f"Expected model_type 'vtp', got {config.get('model_type')!r}.",
        )
    expected = {
        "vision_ffn_layer": "swiglu",
        "decoder_ffn_layer": "swiglu",
        "vision_norm_layer": "rmsnorm",
        "decoder_norm_layer": "layernorm",
    }
    for name, value in expected.items():
        if config.get(name, value) != value:
            raise ValueError(f"Only {name}={value!r} is supported.")
    for name in ("vision_use_qk_norm", "decoder_use_qk_norm"):
        if from_plain(config.get(name), bool, default=False):
            raise ValueError(f"{name}=true is unsupported.")
    for name in ("vision_init_values", "decoder_init_values"):
        if config.get(name) is not None:
            raise ValueError(f"{name} (LayerScale) is unsupported.")


def _swiglu_channels_hidden(channels_nominal: int) -> int:
    """Return SwiGLU's hidden width: 2/3 of nominal, rounded up to a multiple of 8."""
    return ceil_multiple(int(channels_nominal * 2 / 3), multiple=8)


class _Rotary(nn.Module):
    """Axial 2D rotary angles over coordinates normalized per side to ``[-1, 1]``.

    Computed in bfloat16, which rounds the rotated queries and keys to
    bfloat16 whatever the activation dtype.
    """

    # Not priml's ``RoPE``: it rotates in the activation dtype, which differs from
    # the published weights' bfloat16 rotation in a sixth of the rotated values.

    def __init__(self, width: int, *, num_heads: int, base: float = 100.0) -> None:
        super().__init__()
        if width % (4 * num_heads):
            raise ValueError(
                f"width {width} must be divisible by 4 * num_heads ({4 * num_heads}).",
            )
        self.channels_head = width // num_heads
        exponents = torch.arange(self.channels_head // 4, dtype=torch.bfloat16)
        self.periods: Tensor
        self.register_buffer(
            "periods",
            base ** (2 * exponents / (self.channels_head // 2)),
            persistent=True,
        )

    @override
    def forward(self, *, height: int, width: int) -> tuple[Tensor, Tensor]:
        """Return ``(sin, cos)``, each ``[height * width, channels_head]`` bfloat16."""
        device, dtype = self.periods.device, torch.bfloat16
        coords_h = torch.arange(0.5, height, device=device, dtype=dtype) / height
        coords_w = torch.arange(0.5, width, device=device, dtype=dtype) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


def _rotate_after_prefix(x: Tensor, *, rope: tuple[Tensor, Tensor]) -> Tensor:
    """Rotate the trailing ``sin.shape[-2]`` tokens of ``x`` in the table's dtype."""
    sin, cos = rope
    rotating = x.to(sin.dtype)
    prefix = x.shape[-2] - sin.shape[-2]
    head, tail = rotating[..., :prefix, :], rotating[..., prefix:, :]
    first, second = tail.chunk(2, dim=-1)
    rotated = tail * cos + torch.cat([-second, first], dim=-1) * sin
    return torch.cat((head, rotated), dim=-2).to(x.dtype)


class _Attention(nn.Module):
    """Multi-head attention rotating every token after the leading prefix."""

    def __init__(self, width: int, *, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = Linear.Config(width, 3 * width, bias=True).make()
        self.proj = Linear.Config(width, width, bias=True).make()

    @override
    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            width // self.num_heads,
        )
        q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
        q, k = (_rotate_after_prefix(t, rope=rope) for t in (q, k))
        attended = functional.scaled_dot_product_attention(q, k, v)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class _Block(nn.Module):
    """Pre-norm rotary attention then SwiGLU, each added to the residual stream."""

    def __init__(
        self,
        width: int,
        *,
        num_heads: int,
        expansion: float,
        norm: Callable[[], TensorModule],
    ) -> None:
        super().__init__()
        self.norm1 = norm()
        self.attn = _Attention(width, num_heads=num_heads)
        self.norm2 = norm()
        self.mlp = SwiGLU.Config(
            width,
            channels_hidden=_swiglu_channels_hidden(int(width * expansion)),
            bias=True,
        ).make()

    @override
    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        return x + self.mlp(self.norm2(x))


# The sequence is ``prefix`` unrotated tokens then the ``side ** 2`` grid. Each block is
# two norms, the biased ``qkv`` and ``proj`` projections, the rotation, the two
# attention products, the biased SwiGLU, and two residual adds; the rotary table is
# built once and shared by every block. The norms and residual adds run in
# ``dtype_stream``, the rest in ``dtype``.
def _transformer_cost(
    *,
    norm_config: RMSNorm.Config | LayerNorm.Config,
    channels: int,
    heads: int,
    expansion: float,
    num_layers: int,
    side: int,
    prefix: int,
    batch_size: int,
    dtype: torch.dtype | None,
    dtype_stream: torch.dtype | None,
) -> Cost:
    """Cost the rotary table, ``num_layers`` blocks, and the final norm."""
    patches = side * side
    tokens = patches + prefix
    rows = batch_size * tokens
    channels_head = channels // heads
    ffn = SwiGLU.Config(
        channels,
        channels_out=channels,
        channels_hidden=_swiglu_channels_hidden(int(channels * expansion)),
        bias=True,
    )
    block = (
        norm_config.cost(
            seq_len=tokens,
            batch_size=batch_size,
            dtype=dtype_stream,
        ).tile(2, copies=2)
        + matmul_cost(
            channels_in=channels,
            channels_out=3 * channels,
            bias=True,
            rows=rows,
            dtype=dtype,
        )
        + _rotation_cost(
            rotated_rows=batch_size * patches,
            rows=rows,
            channels=channels,
            dtype=dtype,
        )
        + attention_kernel_cost(
            seq_len=tokens,
            batch_size=batch_size,
            dtype=dtype,
            num_heads=heads,
            channels_head=channels_head,
        )
        + matmul_cost(
            channels_in=channels,
            channels_out=channels,
            bias=True,
            rows=rows,
            dtype=dtype,
        )
        + ffn.cost(seq_len=tokens, batch_size=batch_size, dtype=dtype)
        + elementwise_cost(
            primal=rows * 2 * channels,
            adjoint=rows * 2 * channels,
            channels=2 * channels,
            rows=rows,
            inputs=2,
            dtype=dtype_stream,
        )
    )
    return (
        _rope_table_cost(side=side, channels_head=channels_head)
        + block.tile(num_layers, copies=num_layers)
        + norm_config.cost(seq_len=tokens, batch_size=batch_size, dtype=dtype_stream)
    )


# ``x * cos + rotate_half(x) * sin`` is two products and an add per channel, and
# ``rotate_half`` negates half the channels: 7/2 operations each way. The rotation
# runs in bfloat16, so queries and keys of every token are cast there and back
# unless already bfloat16.
def _rotation_cost(
    *,
    rotated_rows: int,
    rows: int,
    channels: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost rotating the queries and keys of ``rotated_rows`` tokens, plus the casts."""
    width = 2 * channels
    rotations = 7 * width // 2
    rotation = elementwise_cost(
        primal=rotated_rows * rotations,
        adjoint=rotated_rows * rotations,
        channels=width,
        inputs=6,
        outputs=3,
        adjoint_inputs=6,
        adjoint_outputs=3,
        rows=rotated_rows,
        dtype=torch.bfloat16,
    )
    dt = resolve_dtype(dtype)
    if dt == torch.bfloat16:
        return rotation
    return sum(
        (
            traffic(
                phase,
                kernel="elementwise",
                elements=2 * rows * width,
                dtype=cast_dtype,
            )
            for phase in ("primal", "adjoint")
            for cast_dtype in (dt, torch.bfloat16)
        ),
        rotation,
    )


# Over an ``s x s`` grid of ``N`` positions and head width ``c``: the per-side
# coordinate divides (2s), the ``2x - 1`` and ``2 pi x`` maps over both axes (6N),
# the divide by the ``c / 4`` periods (Nc / 2), and cos and sin of the tiled angles
# (2Nc). The meshgrid stack and the tile are layout. No gradient reaches the table.
def _rope_table_cost(*, side: int, channels_head: int) -> Cost:
    """Cost one bfloat16 rotary ``(sin, cos)`` table over a ``side x side`` grid."""
    positions = side * side
    angles = positions * channels_head
    coordinates = 6 * side + 14 * positions + channels_head // 4
    return traffic(
        "primal",
        kernel="elementwise",
        elements=coordinates + angles // 2 + 4 * angles,
        flops=2 * side + 6 * positions + angles // 2 + 2 * angles,
        dtype=torch.bfloat16,
    )
