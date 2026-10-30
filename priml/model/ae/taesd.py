"""TAESD: a tiny distilled stand-in for SD-1.x's and SDXL's autoencoders.

Deterministic convolutional halves with SD's latent API: 4 channels at 8x, in
the same diffusion space, since its latent scale is 1. Pixels are in ``[0, 1]``
rather than ``[-1, 1]``. Each half is published as its own file, whose tensors
are named by their index in the reference's ``nn.Sequential``.

Build a published model from :func:`taesd` or :func:`taesdxl`.

MIT license and attribution: ``LICENSES/taesd``.

References:
  https://github.com/madebyollin/taesd/blob/fdbae2df989db95761f482efb8977e1182fb68d2/taesd.py
  https://huggingface.co/madebyollin/taesd

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, map_cost, traffic
from priml.model.ae.checkpoint import CheckpointFile, HubFile, read_state
from priml.model.ae.custom_types import StateDictRemap
from priml.model.ae.distribution import PointMass
from priml.model.ae.frozen_cost import CLAMP, POINTWISE, SUM
from priml.model.ae.latent_norm import LatentNormalizer, ScaleLatents
from priml.model.conv import Conv2d, conv_cost
from priml.model.swiglu import relu


if TYPE_CHECKING:
    from collections.abc import Mapping


TAESD_REVISION: Final = "614f76814bbe30edbe2e627ace1c2234c81a2c0e"
"""The commit of ``madebyollin/taesd`` the defaults pin."""

TAESDXL_REVISION: Final = "b20258aaef75ef61e659c1e0f14f251cf0ad153e"
"""The commit of ``madebyollin/taesdxl`` the defaults pin."""

_STAGES: Final = 3
"""Stride-2 stages: 8x, SD's latent grid."""

_BLOCKS_PER_STAGE: Final = 3
"""Residual blocks in each stage the reference repeats."""

_LATENT_BOUND: Final = 3.0
"""The decoder first squashes latents into ``(-3, 3)``: ``3 tanh(z / 3)``."""

_SQUASH: Final = map_cost(
    primal=3,
    adjoint=0,
    inputs=3,
    outputs=3,
    adjoint_inputs=0,
    adjoint_outputs=0,
)
"""``tanh(z / 3) * 3``: a divide, a tanh, a multiply, per element."""


def remap_taesd_state_dict(
    state: Mapping[str, Tensor],
    *,
    half: str,
) -> dict[str, Tensor]:
    """Rename one half's published tensors, ``nn.Sequential`` indices, to its ``layers``.

    Args:
      state: One half's published file; it holds no other half.
      half: Unread: each half is its own file.

    Returns:
      state: This half's ``state_dict``.

    """
    del half
    return {f"layers.{key}": tensor for key, tensor in state.items()}


class Encoder(nn.Module):
    """Pixels in ``[0, 1]`` to latents: a block, then three stride-2 stages."""

    class Config(Fig["Encoder"]):
        """Widths and weights."""

        channels_in: int = 3
        """Image channels."""

        channels_hidden: int = 64
        """Width of every block."""

        channels_latent: int = 4
        """Latent channels."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: HubFile.Config(
                repo_id="madebyollin/taesd",
                filename="taesd_encoder.safetensors",
                revision=TAESD_REVISION,
                sha256="160d90c61c3a5ce50fe2cbe6404b3429f5763772d61b38523cc37e1525b4e19f",
            ),
        )
        """Weights; ``None`` keeps the random initialization."""

        remap: StateDictRemap = remap_taesd_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a square ``input_grid`` image.

            Args:
              input_grid: Image side.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            del kwargs
            width, side = self.channels_hidden, input_grid
            layers = _Layers(batch_size=batch_size, dtype=dtype)
            total = layers.conv(self.channels_in, width, side=side) + layers.block(
                width,
                side=side,
            )
            for _ in range(_STAGES):
                total += layers.conv(width, width, side=side, stride=2, bias=False)
                side //= 2
                total += layers.block(width, side=side).tile(
                    _BLOCKS_PER_STAGE,
                    copies=_BLOCKS_PER_STAGE,
                )
            return total + layers.conv(width, self.channels_latent, side=side)

    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.channels_hidden
        layers: list[nn.Module] = [_conv(config.channels_in, width), _Block(width)]
        for _ in range(_STAGES):
            layers.append(_conv(width, width, stride=2, bias=False))
            layers.extend(_Block(width) for _ in range(_BLOCKS_PER_STAGE))
        layers.append(_conv(width, config.channels_latent))
        # The reference's own indices, parameter-free modules included, so a remap
        # only adds a prefix.
        self.layers = nn.Sequential(*layers)
        if config.checkpoint is not None:
            state = read_state(config.checkpoint.make().path())
            self.load_state_dict(config.remap(state, half="encoder"))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[N, C_in, H, W]`` pixels in ``[0, 1]`` to ``[N, C, h, w]`` latents."""
        return self.layers(x)


class Decoder(nn.Module):
    """Latents to pixels: a clamp, three stages doubling the grid, then a block."""

    class Config(Fig["Decoder"]):
        """Widths and weights."""

        channels_out: int = 3
        """Image channels."""

        channels_hidden: int = 64
        """Width of every block."""

        channels_latent: int = 4
        """Latent channels."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: HubFile.Config(
                repo_id="madebyollin/taesd",
                filename="taesd_decoder.safetensors",
                revision=TAESD_REVISION,
                sha256="f0fb51dd10d41c26612c070fa0b52ea0215a5ff90792134b4971109dd713c019",
            ),
        )
        """Weights; ``None`` keeps the random initialization."""

        remap: StateDictRemap = remap_taesd_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        def cost(
            self,
            *,
            input_grid: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a square ``input_grid`` latent.

            Args:
              input_grid: Latent side.
              batch_size: Latents in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            del kwargs
            width, side = self.channels_hidden, input_grid
            layers = _Layers(batch_size=batch_size, dtype=dtype)
            latents = batch_size * self.channels_latent * side * side
            total = _SQUASH(channels=latents, dtype=dtype)
            total += layers.conv(self.channels_latent, width, side=side) + layers.relu(
                width,
                side=side,
            )
            for _ in range(_STAGES):
                total += layers.block(width, side=side).tile(
                    _BLOCKS_PER_STAGE,
                    copies=_BLOCKS_PER_STAGE,
                )
                # Nearest upsampling reads the grid once and writes it four times.
                total += traffic(
                    "primal",
                    kernel="elementwise",
                    elements=5 * batch_size * width * side * side,
                    dtype=dtype,
                )
                side *= 2
                total += layers.conv(width, width, side=side, bias=False)
            return (
                total
                + layers.block(width, side=side)
                + layers.conv(width, self.channels_out, side=side)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.channels_hidden
        layers: list[nn.Module] = [
            _Clamp(),
            _conv(config.channels_latent, width),
            nn.ReLU(),
        ]
        for _ in range(_STAGES):
            layers.extend(_Block(width) for _ in range(_BLOCKS_PER_STAGE))
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(_conv(width, width, bias=False))
        layers.extend([_Block(width), _conv(width, config.channels_out)])
        # The reference's own indices, as in ``Encoder``.
        self.layers = nn.Sequential(*layers)
        if config.checkpoint is not None:
            state = read_state(config.checkpoint.make().path())
            self.load_state_dict(config.remap(state, half="decoder"))

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Map ``[N, C, h, w]`` latents to ``[N, C_out, H, W]`` pixels, unclamped."""
        return self.layers(z)


class TAESD(nn.Module):
    """TAESD as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a point mass: the encoder is deterministic. The model is
    frozen and always in evaluation mode.
    """

    class Config(Fig["TAESD"]):
        """The halves, the image side, and the latent normalizer.

        The defaults are TAESD's published model, which :func:`taesd` returns.
        """

        encoder: Encoder.Config | None = field(default_factory=Encoder.Config)
        """Pixels to latents; ``None`` builds and loads no encoder."""

        decoder: Decoder.Config | None = field(default_factory=Decoder.Config)
        """Latents to pixels; ``None`` builds and loads no decoder."""

        image_size: int = 512
        """Side of the square image the latent space it mirrors is native to."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=ScaleLatents.Config,
        )
        """The identity: TAESD's latents are already in SD's diffusion space."""

        def latent_shape(self) -> tuple[int, int, int]:
            """Return ``(channels_latent, side, side)`` at ``image_size``."""
            half = self.encoder or self.decoder
            if half is None:
                raise ValueError("TAESD.Config needs an encoder or a decoder.")
            side = self.image_size >> _STAGES
            return half.channels_latent, side, side

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
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to both halves.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            if self.encoder is None or self.decoder is None:
                raise ValueError("TAESD.Config.cost prices both halves; one is None.")
            _, side, _ = self.latent_shape()
            area = batch_size * self.image_size**2
            pixels_in = area * self.encoder.channels_in
            pixels_out = area * self.decoder.channels_out
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
                # ``image.float() / 255`` in.
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
                    dtype=torch.float32,
                )
                + POINTWISE(channels=pixels_in, dtype=torch.float32)
                + CLAMP(channels=pixels_out, dtype=dtype)
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
            raise ValueError("TAESD needs an encoder or a decoder.")
        if config.image_size % (1 << _STAGES):
            raise ValueError(
                f"image_size {config.image_size} must be divisible by {1 << _STAGES}.",
            )
        if len({half.channels_latent for half in halves}) != 1:
            raise ValueError("The encoder and decoder disagree on channels_latent.")
        self.encoder = None if config.encoder is None else config.encoder.make()
        self.decoder = None if config.decoder is None else config.decoder.make()
        self.eval()
        self.requires_grad_(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, x: Tensor, /) -> PointMass:
        """Return the latent.

        Args:
          x: ``[..., 3, H, W]`` uint8 RGB.

        Returns:
          latent: Point mass at ``[..., channels_latent, h, w]``.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder.

        """
        if self.encoder is None:
            raise ValueError("This TAESD was built without an encoder.")
        # A float image in [0, 1] would otherwise be taken for a uint8 one.
        if x.dtype != torch.uint8:
            raise TypeError(f"TAESD encodes uint8 RGB in [0, 255]; got {x.dtype}.")
        # ``x / 255``, the reference's ``to_tensor``.
        pixels = x.reshape(-1, *x.shape[-3:]).float() / 255
        latent = self.encoder(pixels)
        return PointMass(latent.reshape(*x.shape[:-3], *latent.shape[1:]))

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels_latent, h, w]`` latents.

        Returns:
          image: ``[..., 3, H, W]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder.

        """
        if self.decoder is None:
            raise ValueError("This TAESD was built without a decoder.")
        decoded = self.decoder(z.reshape(-1, *z.shape[-3:]))
        return decoded.reshape(*z.shape[:-3], *decoded.shape[1:]).clamp(0, 1)


def taesd() -> TAESD.Config:
    """Return TAESD, the tiny stand-in for SD-1.x's VAE.

    Returns:
      config: The published model, weights pinned.

    """
    return TAESD.Config()


def taesdxl() -> TAESD.Config:
    """Return TAESDXL, the tiny stand-in for SDXL's VAE.

    Returns:
      config: The published model, weights pinned.

    References:
      https://huggingface.co/madebyollin/taesdxl

    """
    config = TAESD.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = HubFile.Config(
        repo_id="madebyollin/taesdxl",
        filename="taesdxl_encoder.safetensors",
        revision=TAESDXL_REVISION,
        sha256="9f37c0b28f72ec4ca835dc7dbf05255bdf323cde8cf12a304674f106466c98ef",
    )
    config.decoder.checkpoint = HubFile.Config(
        repo_id="madebyollin/taesdxl",
        filename="taesdxl_decoder.safetensors",
        revision=TAESDXL_REVISION,
        sha256="f6013131e7eb412ef20113f1acc2ea7d3e47e53196ca0530fa65d9b61d814b61",
    )
    config.image_size = 1024
    return config


def _conv(
    channels_in: int,
    channels_out: int,
    *,
    stride: int = 1,
    bias: bool = True,
) -> Conv2d:
    """Return the reference's 3x3 convolution, padded by one on every side."""
    return Conv2d.Config(
        channels_in,
        channels_out,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
    ).make()


class _Clamp(nn.Module):
    """Squash into ``(-3, 3)``: ``3 tanh(x / 3)``."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Not ``softcap``: it computes a narrower input in float32, where the
        # reference rounds at the input's width.
        return torch.tanh(x / _LATENT_BOUND) * _LATENT_BOUND


class _Block(nn.Module):
    """Three 3x3 convolutions with ReLUs between, added to the input, then ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(channels, channels),
            nn.ReLU(),
            _conv(channels, channels),
            nn.ReLU(),
            _conv(channels, channels),
        )
        self.skip = nn.Identity()
        self.fuse = nn.ReLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Layers:
    """Price the reference's layers at one batch size and dtype."""

    batch_size: int
    dtype: torch.dtype | None

    def conv(
        self,
        channels_in: int,
        channels_out: int,
        *,
        side: int,
        stride: int = 1,
        bias: bool = True,
    ) -> Cost:
        """Cost one 3x3 convolution on a square ``side`` grid."""
        return conv_cost(
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=3,
            ndim=2,
            groups=1,
            bias=bias,
            input_grid=(side, side),
            batch_size=self.batch_size,
            stride=stride,
            padding=1,
            dtype=self.dtype,
        )

    def relu(self, channels: int, *, side: int) -> Cost:
        """Cost one ReLU over a ``side`` grid."""
        return cost(
            relu,
            channels=self.batch_size * channels * side * side,
            dtype=self.dtype,
        )

    def block(self, channels: int, *, side: int) -> Cost:
        """Cost ``_Block``: three convolutions, two ReLUs, the sum, a ReLU."""
        elements = self.batch_size * channels * side * side
        return (
            self.conv(channels, channels, side=side).tile(3, copies=3)
            + self.relu(channels, side=side).tile(3)
            + SUM(channels=elements, dtype=self.dtype)
        )
