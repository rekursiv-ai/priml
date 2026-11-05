"""RAE: a frozen DINOv2-with-registers encoder and a ViT pixel decoder.

The latent is the encoder's patch-token grid. Build the published model with
``RAE.load()``, or parse the two ``config.json`` files with
:meth:`RAE.Config.from_hf`.

MIT license and attribution: ``LICENSES/RAE``.

References:
  https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/rae.py
  https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/decoders/decoder.py
  https://huggingface.co/nyu-visionx/RAE-collections
    Zheng et al. 2025. Diffusion Transformers with Representation
    Autoencoders.

"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Final, Self, cast, override

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
    map_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.math.pixel import rgb2float
from priml.math.position_embedding import sincos_position_table
from priml.model.ae.checkpoint import (
    CheckpointFile,
    HubFile,
    hub_or_local,
    read_json,
    read_state,
)
from priml.model.ae.distribution import PointMass
from priml.model.ae.latent_norm import ElementwiseLatentStats, LatentNormalizer
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import Conv2d, conv_cost
from priml.model.linear import Linear
from priml.model.norm import LayerNorm


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


ENCODER_REPO: Final = "facebook/dinov2-with-registers-base"
"""The published DINOv2-B with registers."""

ENCODER_REVISION: Final = "a1d738ccfa7ae170945f210395d99dde8adb1805"
"""The commit of :data:`ENCODER_REPO` the defaults pin."""

DECODER_REPO: Final = "nyu-visionx/RAE-collections"
"""The published RAE decoders and latent statistics."""

DECODER_REVISION: Final = "1be4f03273523431f099a934da4cf1940dc6039f"
"""The commit of :data:`DECODER_REPO` the defaults pin."""


def dinov2_with_registers_base() -> HubFile.Config:
    """Return the pinned DINOv2-B (registers) safetensors, final-norm affine included.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id=ENCODER_REPO,
        filename="model.safetensors",
        revision=ENCODER_REVISION,
        sha256="7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362",
    )


def vit_xl_n08() -> HubFile.Config:
    """Return the pinned ViT-XL decoder trained on DINOv2-B (registers) latents.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id=DECODER_REPO,
        filename="decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        revision=DECODER_REVISION,
        sha256="5fedf7c9660476a709e122cef18385c917532914a23f034388e1c1a52bde2be6",
    )


class Encoder(nn.Module):
    """DINOv2 with registers: ImageNet-normalized pixels to one token per patch.

    The final layer norm has no affine; the CLS and register tokens are dropped.
    """

    class Config(Fig["Encoder"]):
        """Architecture and weights; the defaults are the published DINOv2-B."""

        channels_hidden: int = 768
        """Token width, which is the latent's channel count (HF ``hidden_size``)."""

        num_layers: int = 12
        """Transformer blocks (HF ``num_hidden_layers``)."""

        heads: int = 12
        """Attention heads (HF ``num_attention_heads``)."""

        expansion: int = 4
        """MLP width over ``channels_hidden`` (HF ``mlp_ratio``)."""

        patch_size: int = 14
        """Pixels per patch side."""

        image_size: int = 518
        """Side the position table was trained at; other sides interpolate it."""

        num_register_tokens: int = 4
        """Register tokens after CLS; both are dropped before the patches."""

        eps: float = 1e-6
        """Layer-norm epsilon (HF ``layer_norm_eps``)."""

        activation: TensorFn = functional.gelu
        """MLP activation; HF's ``"gelu"`` is this exact-erf form."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=dinov2_with_registers_base,
        )
        """HF safetensors; ``None`` keeps the random initialization."""

        @classmethod
        def from_hf(cls, config: Mapping[str, object]) -> Self:
            """Parse a ``Dinov2WithRegistersModel`` ``config.json``; no checkpoint.

            Args:
              config: The ``config.json`` object.

            Returns:
              config: The architecture.

            Raises:
              ValueError: A field names an architecture this port does not build.

            """
            if config.get("model_type") != "dinov2_with_registers":
                raise ValueError(
                    "Expected model_type 'dinov2_with_registers', got "
                    f"{config.get('model_type')!r}.",
                )
            if from_plain(config.get("use_swiglu_ffn"), bool, default=False):
                raise ValueError("The SwiGLU DINOv2 MLP is unsupported.")
            if not from_plain(config.get("qkv_bias"), bool, default=True):
                raise ValueError("Bias-free query, key, and value are unsupported.")
            if from_plain(config.get("hidden_act"), str, default="gelu") != "gelu":
                raise ValueError("Only the exact GELU DINOv2 MLP is supported.")
            return cls(
                channels_hidden=from_plain(config["hidden_size"], int),
                num_layers=from_plain(config["num_hidden_layers"], int),
                heads=from_plain(config["num_attention_heads"], int),
                expansion=from_plain(config["mlp_ratio"], int),
                patch_size=from_plain(config["patch_size"], int),
                image_size=from_plain(config["image_size"], int),
                num_register_tokens=from_plain(config["num_register_tokens"], int),
                eps=from_plain(config["layer_norm_eps"], float),
                checkpoint=None,
            )

        def cost(
            self,
            *,
            input_grid: int | tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward and its adjoint.

            Args:
              input_grid: ``(height, width)`` pixels in; a scalar is square.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            Raises:
              TypeError: ``activation`` has no cost.

            """
            del kwargs
            height, width = (
                (input_grid, input_grid) if isinstance(input_grid, int) else input_grid
            )
            channels = self.channels_hidden
            grid = (height // self.patch_size, width // self.patch_size)
            patches = grid[0] * grid[1]
            table_grid = self.image_size // self.patch_size
            positions = table_grid * table_grid
            seq_len = 1 + self.num_register_tokens + patches
            rows = seq_len * batch_size
            channels_mlp = channels * self.expansion
            layer = _vit_layer_cost(
                channels=channels,
                channels_mlp=channels_mlp,
                num_heads=self.heads,
                seq_len=seq_len,
                batch_size=batch_size,
                layer_scale=True,
                activation=_activation_cost(
                    self.activation,
                    channels=rows * channels_mlp,
                    dtype=dtype,
                ),
                dtype=dtype,
            )
            patch = conv_cost(
                channels_in=3,
                channels_out=channels,
                kernel_size=self.patch_size,
                ndim=2,
                groups=1,
                bias=True,
                input_grid=(height, width),
                batch_size=batch_size,
                stride=self.patch_size,
                padding=0,
                dtype=dtype,
            )
            with_cls = batch_size * (1 + patches) * channels
            embeddings = (
                _concatenate_cost(elements=with_cls, dtype=dtype)
                + _concatenate_cost(elements=rows * channels, dtype=dtype)
                # The table broadcasts over the batch, so its gradient sums there.
                + elementwise_cost(
                    primal=with_cls,
                    adjoint=0,
                    channels=(1 + patches) * channels,
                    rows=batch_size,
                    inputs=2,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=with_cls,
                    output_groups=(1 + patches) * channels,
                    dtype=dtype,
                    phase="adjoint",
                )
                # CLS and registers are expanded over the batch likewise.
                + reduction_cost(
                    input_elements=batch_size
                    * (1 + self.num_register_tokens)
                    * channels,
                    output_groups=(1 + self.num_register_tokens) * channels,
                    dtype=dtype,
                    phase="adjoint",
                )
            )
            if patches != positions or height != width:
                embeddings += _bicubic_cost(
                    planes=channels,
                    grid_in=(table_grid, table_grid),
                    grid_out=grid,
                    antialias=True,
                    dtype=torch.float32,
                ) + _concatenate_cost(elements=(1 + patches) * channels, dtype=dtype)
                if resolve_dtype(dtype) != torch.float32:
                    # The table in, the resized grid back out, each phase.
                    embeddings += sum(
                        (
                            traffic(
                                phase,
                                kernel="elementwise",
                                elements=(positions + patches) * channels,
                                dtype=cast_dtype,
                            )
                            for phase in ("primal", "adjoint")
                            for cast_dtype in (dtype, torch.float32)
                        ),
                        Cost(),
                    )
            final_norm = cost(
                LayerNorm.Config(channels, elementwise_affine=False),
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
            )
            # The CLS and mask tokens, the registers, and the position table, whose
            # first row is the CLS token's.
            tables = (3 + self.num_register_tokens + positions) * channels
            return (
                patch
                + embeddings
                + layer.tile(self.num_layers, copies=self.num_layers)
                + final_norm
                + Cost(params=tables, params_active=tables - channels)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        channels = config.channels_hidden
        table_grid = config.image_size // config.patch_size
        self.patch_size = config.patch_size
        self.num_leading_tokens = 1 + config.num_register_tokens
        # Registration order is the checkpoint's, so a random initialization and
        # ``state_dict`` order follow it.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.mask_token = nn.Parameter(torch.zeros(1, channels))
        self.patch_embed = Conv2d.Config(
            3,
            channels,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
        ).make()
        self.position_embeddings = nn.Parameter(
            torch.randn(1, table_grid * table_grid + 1, channels),
        )
        self.register_tokens = nn.Parameter(
            torch.zeros(1, config.num_register_tokens, channels),
        )
        self.layers = nn.ModuleList(
            _Block(
                channels=channels,
                channels_mlp=channels * config.expansion,
                num_heads=config.heads,
                activation=config.activation,
                eps=config.eps,
                layer_scale=True,
            )
            for _ in range(config.num_layers)
        )
        self.norm = LayerNorm.Config(channels, eps=config.eps).make()
        if config.checkpoint is not None:
            state = read_state(
                config.checkpoint.make().path(),
                prefixes=("embeddings.", "encoder.", "layernorm."),
            )
            self.load_state_dict(remap_encoder_state_dict(state))

    @override
    def forward(self, image: Tensor) -> Tensor:
        """Return the normalized patch tokens.

        Args:
          image: ``[N, 3, H, W]`` ImageNet-normalized pixels.

        Returns:
          tokens: ``[N, (H / patch) * (W / patch), channels_hidden]``.

        """
        batch, _, height, width = image.shape
        tokens = self.patch_embed(image.to(self.patch_embed.weight.dtype))
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = tokens + self._positions(tokens, height=height, width=width)
        tokens = torch.cat(
            (
                tokens[:, :1],
                self.register_tokens.expand(batch, -1, -1),
                tokens[:, 1:],
            ),
            dim=1,
        )
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)[:, self.num_leading_tokens :]

    def _positions(self, tokens: Tensor, *, height: int, width: int) -> Tensor:
        """Return the position table, resized to the input's patch grid when it differs."""
        num_patches = tokens.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        cls_position = self.position_embeddings[:, 0]
        grid = math.isqrt(num_positions)
        channels = tokens.shape[-1]
        table = self.position_embeddings[:, 1:].reshape(1, grid, grid, channels)
        dtype = table.dtype
        # HF resizes in float32 whatever the table's dtype, with antialiasing.
        resized = functional.interpolate(
            table.permute(0, 3, 1, 2).to(torch.float32),
            size=(height // self.patch_size, width // self.patch_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype)
        resized = resized.permute(0, 2, 3, 1).view(1, -1, channels)
        return torch.cat((cls_position.unsqueeze(0), resized), dim=1)


class Decoder(nn.Module):
    """ViTMAE pixel decoder: latent tokens to ImageNet-normalized pixels.

    The position table is a fixed 2D sine-cosine one, stored as a frozen parameter.
    """

    class Config(Fig["Decoder"]):
        """Architecture and weights; the defaults are the published ViT-XL."""

        channels_in: int = 768
        """Latent token width; :class:`RAE` checks it against the encoder's."""

        num_patches: int = 256
        """Latent tokens, a square; :class:`RAE` checks it against the encoder's grid."""

        channels_hidden: int = 1152
        """Transformer width (``decoder_hidden_size``)."""

        channels_hidden_mlp: int = 4096
        """MLP hidden width (``decoder_intermediate_size``)."""

        num_layers: int = 28
        """Transformer blocks (``decoder_num_hidden_layers``)."""

        heads: int = 16
        """Attention heads (``decoder_num_attention_heads``)."""

        patch_size: int = 16
        """Output pixels per latent token side."""

        eps: float = 1e-12
        """Layer-norm epsilon (``layer_norm_eps``)."""

        activation: TensorFn = functional.gelu
        """MLP activation; the reference's ``"gelu"`` is the exact erf form."""

        checkpoint: Makeable[CheckpointFile] | None = field(default_factory=vit_xl_n08)
        """Bare decoder state dict; ``None`` keeps the random initialization."""

        @classmethod
        def from_hf(cls, config: Mapping[str, object], *, patch_size: int) -> Self:
            """Parse an RAE ViTMAE decoder ``config.json``; no checkpoint.

            Args:
              config: The ``config.json`` object.
              patch_size: Output pixels per latent token side.

            Returns:
              config: The architecture.

            Raises:
              ValueError: A field names an architecture this port does not build.

            """
            if config.get("model_type") != "vit_mae":
                raise ValueError(
                    f"Expected model_type 'vit_mae', got {config.get('model_type')!r}.",
                )
            if from_plain(config.get("hidden_act"), str, default="gelu") != "gelu":
                raise ValueError("Only the exact GELU ViTMAE decoder is supported.")
            if not from_plain(config.get("qkv_bias"), bool, default=True):
                raise ValueError("Bias-free query, key, and value are unsupported.")
            return cls(
                channels_in=from_plain(config["hidden_size"], int),
                channels_hidden=from_plain(config["decoder_hidden_size"], int),
                channels_hidden_mlp=from_plain(
                    config["decoder_intermediate_size"],
                    int,
                ),
                num_layers=from_plain(config["decoder_num_hidden_layers"], int),
                heads=from_plain(config["decoder_num_attention_heads"], int),
                patch_size=patch_size,
                eps=from_plain(config["layer_norm_eps"], float),
                checkpoint=None,
            )

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on ``num_patches`` tokens and its adjoint.

            Args:
              batch_size: Latent grids in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            Raises:
              TypeError: ``activation`` carries no cost and is not torch's GELU.

            """
            del kwargs
            channels = self.channels_hidden
            seq_len = self.num_patches + 1
            rows = seq_len * batch_size
            layer = _vit_layer_cost(
                channels=channels,
                channels_mlp=self.channels_hidden_mlp,
                num_heads=self.heads,
                seq_len=seq_len,
                batch_size=batch_size,
                layer_scale=False,
                activation=_activation_cost(
                    self.activation,
                    channels=rows * self.channels_hidden_mlp,
                    dtype=dtype,
                ),
                dtype=dtype,
            )
            embed = matmul_cost(
                channels_in=self.channels_in,
                channels_out=channels,
                bias=True,
                rows=batch_size * self.num_patches,
                dtype=dtype,
            )
            tokens = (
                _concatenate_cost(elements=rows * channels, dtype=dtype)
                + elementwise_cost(
                    primal=rows * channels,
                    adjoint=0,
                    channels=seq_len * channels,
                    rows=batch_size,
                    inputs=2,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    dtype=dtype,
                )
                # The CLS token is expanded over the batch, so its gradient sums there.
                + reduction_cost(
                    input_elements=batch_size * channels,
                    output_groups=channels,
                    dtype=dtype,
                    phase="adjoint",
                )
            )
            head = cost(
                LayerNorm.Config(channels, elementwise_affine=True),
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
            ) + matmul_cost(
                channels_in=channels,
                channels_out=self.patch_size**2 * 3,
                bias=True,
                rows=rows,
                dtype=dtype,
            )
            tables = (seq_len + 1) * channels
            return (
                embed
                + tokens
                + layer.tile(self.num_layers, copies=self.num_layers)
                + head
                + Cost(params=tables, params_active=tables)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        n = config.num_patches
        if n <= 0 or math.isqrt(n) ** 2 != n:
            raise ValueError(f"num_patches {n} must be a positive square.")
        self.num_patches = n
        self.patch_size = config.patch_size
        self.grid_size = math.isqrt(n)
        channels = config.channels_hidden
        # Registration order is the reference's, trainable CLS token last.
        self.decoder_embed = _linear(config.channels_in, channels)
        self.decoder_pos_embed = nn.Parameter(
            sincos_position_table(channels, grid=self.grid_size).unsqueeze(0),
            requires_grad=False,
        )
        self.decoder_layers = nn.ModuleList(
            _Block(
                channels=channels,
                channels_mlp=config.channels_hidden_mlp,
                num_heads=config.heads,
                activation=config.activation,
                eps=config.eps,
                layer_scale=False,
            )
            for _ in range(config.num_layers)
        )
        self.decoder_norm = LayerNorm.Config(
            channels,
            eps=config.eps,
            elementwise_affine=True,
        ).make()
        self.decoder_pred = _linear(channels, config.patch_size**2 * 3)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        if config.checkpoint is not None:
            state = read_state(config.checkpoint.make().path())
            self.load_state_dict(remap_decoder_state_dict(state))

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Reconstruct ImageNet-normalized pixels.

        Args:
          tokens: ``[N, num_patches, channels_in]`` latent tokens, row-major.

        Returns:
          pixels: ``[N, 3, side, side]`` with ``side = patch_size * sqrt(num_patches)``.

        Raises:
          ValueError: The token count is not ``num_patches``.

        """
        if tokens.shape[1] != self.num_patches:
            raise ValueError(
                f"Decoder was built for {self.num_patches} latent tokens, got "
                f"{tokens.shape[1]}; the reference would resize them silently.",
            )
        embedded = self.decoder_embed(tokens)
        cls_token = self.trainable_cls_token.expand(embedded.shape[0], -1, -1)
        hidden = torch.cat([cls_token, embedded], dim=1) + self.decoder_pos_embed
        for layer in self.decoder_layers:
            hidden = layer(hidden)
        logits = self.decoder_pred(self.decoder_norm(hidden))[:, 1:, :]
        grid, patch = self.grid_size, self.patch_size
        patches = logits.reshape(logits.shape[0], grid, grid, patch, patch, 3)
        patches = torch.einsum("nhwpqc->nchpwq", patches)
        return patches.reshape(logits.shape[0], 3, grid * patch, grid * patch)


class RAE(nn.Module):
    """RAE as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a point mass; ``decode`` returns ``image_size`` pixels.
    The model is frozen and always in evaluation mode.
    """

    class Config(Fig["RAE"]):
        """The halves, the two image sides, pixel statistics, and the latent normalizer."""

        encoder: Encoder.Config | None = field(default_factory=Encoder.Config)
        """Representation encoder; ``None`` builds and loads none."""

        decoder: Decoder.Config | None = field(default_factory=Decoder.Config)
        """Pixel decoder; ``None`` builds and loads none."""

        image_size: int = 256
        """Side of the square image the decoder reconstructs."""

        encoder_image_size: int = 224
        """Side the encoder sees; other inputs are resized bicubically first."""

        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """ImageNet mean, from the encoder's ``preprocessor_config.json``."""

        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """ImageNet deviation, from the encoder's ``preprocessor_config.json``."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: ElementwiseLatentStats.Config(
                stats=HubFile.Config(
                    repo_id=DECODER_REPO,
                    filename="stats/dinov2/wReg_base/imagenet1k/stat.pt",
                    revision=DECODER_REVISION,
                    sha256="84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59",
                ),
            ),
        )
        """The published ImageNet statistics; the autoencoder itself never applies them."""

        @classmethod
        def from_hf(
            cls,
            encoder: Mapping[str, object],
            decoder: Mapping[str, object],
            *,
            decoder_patch_size: int,
        ) -> Self:
            """Parse the encoder's and the decoder's ``config.json``; no checkpoints.

            Args:
              encoder: DINOv2-with-registers ``config.json``.
              decoder: The reference's ViTMAE decoder ``config.json``.
              decoder_patch_size: Output pixels per latent token side.

            Returns:
              config: Both halves at the default image sides.

            """
            config = cls(
                encoder=Encoder.Config.from_hf(encoder),
                decoder=Decoder.Config.from_hf(decoder, patch_size=decoder_patch_size),
            )
            config.image_size = decoder_patch_size * config.latent_shape()[1]
            return config

        def latent_shape(self) -> tuple[int, int, int]:
            """Return the latent's shape at ``encoder_image_size``.

            Returns:
              shape: ``(channels, grid, grid)``.

            """
            if self.encoder is not None:
                grid = self.encoder_image_size // self.encoder.patch_size
                return self.encoder.channels_hidden, grid, grid
            if self.decoder is None:
                raise ValueError("RAE.Config needs an encoder or a decoder.")
            grid = math.isqrt(self.decoder.num_patches)
            return self.decoder.channels_in, grid, grid

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
              dtype: Activation dtype of both halves; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to both halves.

            Returns:
              cost: Forward FLOPs and logical bytes; no adjoint cells.

            """
            if self.encoder is None or self.decoder is None:
                raise ValueError("RAE.Config.cost prices both halves; one is None.")
            side, encoder_side = self.image_size, self.encoder_image_size
            pixels = batch_size * 3 * side * side
            encoder_pixels = batch_size * 3 * encoder_side * encoder_side
            f32 = torch.float32
            # ``float() / 255`` as one map: uint8 in, float32 out.
            full = traffic(
                "primal",
                kernel="elementwise",
                elements=pixels,
                dtype=torch.uint8,
            ) + traffic(
                "primal",
                kernel="elementwise",
                elements=pixels,
                flops=pixels,
                dtype=f32,
            )
            if side != encoder_side:
                full += _bicubic_cost(
                    planes=batch_size * 3,
                    grid_in=(side, side),
                    grid_out=(encoder_side, encoder_side),
                    antialias=False,
                    dtype=f32,
                )
            # Normalize (subtract, divide), then denormalize (multiply, add)
            # and clamp (max, min): two operations each.
            full += (
                elementwise_cost(
                    primal=2 * encoder_pixels,
                    adjoint=0,
                    channels=encoder_pixels,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    dtype=f32,
                )
                + cost(
                    self.encoder,
                    **{
                        **kwargs,
                        "input_grid": (encoder_side, encoder_side),
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                + cost(self.decoder, batch_size=batch_size, dtype=dtype, **kwargs)
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=2 * pixels,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=2 * pixels,
                    adjoint=0,
                    channels=pixels,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    dtype=dtype,
                ).tile(2)
            )
            return Cost(
                cells=full.only("primal").cells,
                params=full.params,
                params_active=full.params_active,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.encoder is None and config.decoder is None:
            raise ValueError("RAE needs an encoder or a decoder.")
        mean, std = config.pixel_mean, config.pixel_std
        if any(math.isnan(v) or math.isinf(v) for v in (*mean, *std)) or 0 in std:
            raise ValueError(
                "RAE needs finite pixel statistics and a nonzero std; got mean "
                f"{mean}, std {std}.",
            )
        self.latent_shape = config.latent_shape()
        channels, grid, _ = self.latent_shape
        if config.encoder is not None:
            patch = config.encoder.patch_size
            if config.encoder_image_size % patch:
                raise ValueError(
                    f"encoder_image_size {config.encoder_image_size} must be "
                    f"divisible by the encoder patch {patch}.",
                )
        if config.decoder is not None:
            decoder = config.decoder
            if (decoder.channels_in, decoder.num_patches) != (channels, grid * grid):
                raise ValueError(
                    f"The decoder reads {decoder.channels_in}-wide tokens on "
                    f"{decoder.num_patches} patches; the encoder produces "
                    f"{channels}-wide tokens on {grid * grid}.",
                )
            if decoder.patch_size * grid != config.image_size:
                raise ValueError(
                    f"image_size {config.image_size} must be the decoder patch "
                    f"{decoder.patch_size} times the latent grid {grid}.",
                )
        self.encoder_image_size = config.encoder_image_size
        # Encoder before decoder, as the reference registers them.
        self.encoder = None if config.encoder is None else config.encoder.make()
        self.pixel_mean: Tensor
        self.pixel_std: Tensor
        self.register_buffer(
            "pixel_mean",
            torch.tensor(mean).view(3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(std).view(3, 1, 1),
            persistent=False,
        )
        self.decoder = None if config.decoder is None else config.decoder.make()
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def load(
        cls,
        encoder: Path | str = ENCODER_REPO,
        decoder: Path | str = DECODER_REPO,
        *,
        encoder_revision: str = ENCODER_REVISION,
        decoder_revision: str = DECODER_REVISION,
    ) -> RAE:
        """Build the published DINOv2-B (registers) RAE, weights included.

        The decoder's architecture is the default :class:`Decoder.Config`.

        Args:
          encoder: Local directory holding ``config.json`` and
            ``model.safetensors``, or a Hub repository.
          decoder: Local directory, or Hub repository, holding
            ``decoders/dinov2/wReg_base/ViTXL_n08/model.pt``.
          encoder_revision: Commit to read a Hub encoder at.
          decoder_revision: Commit to read a Hub decoder at.

        Returns:
          model: Frozen RAE.

        """
        encoder_config = Encoder.Config.from_hf(
            read_json(hub_or_local(encoder, "config.json", revision=encoder_revision)),
        )
        encoder_config.checkpoint = hub_or_local(
            encoder,
            "model.safetensors",
            revision=encoder_revision,
        )
        decoder_config = Decoder.Config(
            checkpoint=hub_or_local(
                decoder,
                "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
                revision=decoder_revision,
            ),
        )
        config = cls.Config(encoder=encoder_config, decoder=decoder_config)
        return config.make()

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, x: Tensor, /) -> PointMass:
        """Return the encoder's patch tokens as a grid, as a point mass.

        Args:
          x: ``[..., 3, H, W]`` uint8 RGB.

        Returns:
          latent: Point mass on ``[..., channels, grid, grid]`` raw latents.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder.

        """
        if self.encoder is None:
            raise ValueError("This RAE was built without an encoder.")
        # A float image in [0, 1] would otherwise encode as near-black.
        if x.dtype != torch.uint8:
            raise TypeError(f"RAE encodes uint8 RGB in [0, 255]; got {x.dtype}.")
        pixels = rgb2float(
            x.reshape(-1, *x.shape[-3:]),
            float_dtype=torch.float32,
            unit_interval=True,
        )
        side = self.encoder_image_size
        if pixels.shape[-2:] != (side, side):
            pixels = functional.interpolate(
                pixels,
                size=(side, side),
                mode="bicubic",
                align_corners=False,
            )
        tokens = self.encoder((pixels - self.pixel_mean) / self.pixel_std)
        batch, count, channels = tokens.shape
        grid = math.isqrt(count)
        latent = tokens.transpose(1, 2).view(batch, channels, grid, grid)
        return PointMass(latent.reshape(*x.shape[:-3], *latent.shape[1:]))

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode raw latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels, grid, grid]`` raw latents.

        Returns:
          image: ``[..., 3, image_size, image_size]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder, or the grid is
            not the one the decoder was built for.

        """
        if self.decoder is None:
            raise ValueError("This RAE was built without a decoder.")
        if tuple(z.shape[-3:]) != self.latent_shape:
            raise ValueError(
                f"RAE decodes latents of shape {self.latent_shape}, got "
                f"{tuple(z.shape[-3:])}; the reference would reshape them silently.",
            )
        flat = z.reshape(-1, *self.latent_shape)
        tokens = flat.flatten(2).transpose(1, 2)
        pixels = self.decoder(tokens) * self.pixel_std + self.pixel_mean
        return pixels.clamp(0, 1).reshape(*z.shape[:-3], *pixels.shape[1:])


def rae_dinov2_base() -> RAE.Config:
    """Return the published DINOv2-B (registers) RAE, the :class:`RAE.Config` defaults.

    References:
      https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/configs/stage1/pretrained/DINOv2-B.yaml

    Returns:
      config: Encoder, ``ViTXL_n08`` decoder, and ImageNet latent statistics.

    """
    return RAE.Config()


def remap_encoder_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename HF ``Dinov2WithRegistersModel`` tensors to :class:`Encoder`'s names.

    The final norm's weight and bias are dropped.

    Args:
      state: The published safetensors.

    Returns:
      state: :class:`Encoder`'s ``state_dict``.

    Raises:
      KeyError: A tensor this encoder has no place for.

    """
    remapped: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key in {"layernorm.weight", "layernorm.bias"}:
            continue
        name = _ENCODER_NAMES.get(key)
        if name is None and key.startswith("encoder.layer."):
            index, rest = key.removeprefix("encoder.layer.").split(".", 1)
            name = f"layers.{index}.{_ENCODER_LAYER_NAMES[rest]}"
        if name is None:
            raise KeyError(f"No encoder parameter takes {key}.")
        remapped[name] = tensor
    return remapped


def remap_decoder_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename published RAE decoder tensors to :class:`Decoder`'s names.

    Args:
      state: The published decoder state dict.

    Returns:
      state: :class:`Decoder`'s ``state_dict``.

    Raises:
      KeyError: A tensor this decoder has no place for.

    """
    remapped: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if not key.startswith("decoder_layers."):
            remapped[key] = tensor
            continue
        index, rest = key.removeprefix("decoder_layers.").split(".", 1)
        name = _DECODER_LAYER_NAMES.get(rest)
        if name is None:
            raise KeyError(f"No decoder parameter takes {key}.")
        remapped[f"decoder_layers.{index}.{name}"] = tensor
    return remapped


_ENCODER_NAMES: Final = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.register_tokens": "register_tokens",
    "embeddings.position_embeddings": "position_embeddings",
    "embeddings.patch_embeddings.projection.weight": "patch_embed.weight",
    "embeddings.patch_embeddings.projection.bias": "patch_embed.bias",
}

# The published names, then transformers 5.19's for the same tensors.
_ENCODER_LAYER_NAMES: Final = {
    **{
        f"{prefix}{role}.{kind}": f"{block}.{kind}"
        for prefix, role, block in (
            ("attention.attention.", "query", "attention.query"),
            ("attention.attention.", "key", "attention.key"),
            ("attention.attention.", "value", "attention.value"),
            ("attention.output.", "dense", "attention.out"),
            ("attention.", "q_proj", "attention.query"),
            ("attention.", "k_proj", "attention.key"),
            ("attention.", "v_proj", "attention.value"),
            ("attention.", "o_proj", "attention.out"),
            ("", "norm1", "norm1"),
            ("", "norm2", "norm2"),
            ("mlp.", "fc1", "mlp_in"),
            ("mlp.", "fc2", "mlp_out"),
        )
        for kind in ("weight", "bias")
    },
    "layer_scale1.lambda1": "scale1",
    "layer_scale2.lambda1": "scale2",
}

_DECODER_LAYER_NAMES: Final = {
    f"{published}.{kind}": f"{ours}.{kind}"
    for published, ours in (
        ("attention.attention.query", "attention.query"),
        ("attention.attention.key", "attention.key"),
        ("attention.attention.value", "attention.value"),
        ("attention.output.dense", "attention.out"),
        ("layernorm_before", "norm1"),
        ("layernorm_after", "norm2"),
        ("intermediate.dense", "mlp_in"),
        ("output.dense", "mlp_out"),
    )
    for kind in ("weight", "bias")
}


def _linear(channels_in: int, channels_out: int) -> Linear:
    """Return a biased priml ``Linear``."""
    return Linear.Config(channels_in, channels_out, bias=True).make()


class _Attention(nn.Module):
    """Multi-head attention as plain matmul, scale, softmax, matmul."""

    # Not priml's ``Attention``: its fused query-key-value projection and fused
    # kernel round differently, which moves latents off the published weights'.

    def __init__(self, channels: int, *, num_heads: int, divide: bool) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"{num_heads} heads do not divide width {channels}.")
        self.num_heads = num_heads
        self.channels_head = channels // num_heads
        self.divide = divide
        self.query = _linear(channels, channels)
        self.key = _linear(channels, channels)
        self.value = _linear(channels, channels)
        self.out = _linear(channels, channels)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        query, key, value = (
            self._heads(projection(hidden))
            for projection in (self.query, self.key, self.value)
        )
        scores = torch.matmul(query, key.transpose(-1, -2))
        # The decoder's reference divides by the correctly rounded root, the
        # encoder's multiplies by ``d ** -0.5``: at width 72 the two disagree in
        # most elements, so each half keeps its own.
        if self.divide:
            scores = scores / math.sqrt(self.channels_head)  # noqa: TID251 -- The reference's correctly rounded root; ``** 0.5`` goes through ``pow``, which need not round the same.
        else:
            scores = scores * cast(float, self.channels_head**-0.5)
        context = torch.matmul(functional.softmax(scores, dim=-1), value)
        context = context.transpose(1, 2).contiguous()
        return self.out(context.reshape(*context.shape[:-2], -1))

    def _heads(self, projected: Tensor) -> Tensor:
        """Split ``[N, T, C]`` into ``[N, heads, T, channels_head]``."""
        split = projected.view(
            *projected.shape[:-1],
            self.num_heads,
            self.channels_head,
        )
        return split.transpose(1, 2)


class _Block(nn.Module):
    """Pre-norm attention then MLP, each added to the residual, optionally scaled."""

    def __init__(
        self,
        *,
        channels: int,
        channels_mlp: int,
        num_heads: int,
        activation: TensorFn,
        eps: float,
        layer_scale: bool,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm.Config(channels, eps=eps, elementwise_affine=True).make()
        self.attention = _Attention(
            channels,
            num_heads=num_heads,
            divide=not layer_scale,
        )
        self.norm2 = LayerNorm.Config(channels, eps=eps, elementwise_affine=True).make()
        self.mlp_in = _linear(channels, channels_mlp)
        self.activation = activation
        self.mlp_out = _linear(channels_mlp, channels)
        self.scale1 = nn.Parameter(torch.ones(channels)) if layer_scale else None
        self.scale2 = nn.Parameter(torch.ones(channels)) if layer_scale else None

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        attended = self.attention(self.norm1(hidden))
        if self.scale1 is not None:
            attended = attended * self.scale1
        hidden = attended + hidden
        expanded = self.mlp_out(self.activation(self.mlp_in(self.norm2(hidden))))
        if self.scale2 is not None:
            expanded = expanded * self.scale2
        return expanded + hidden


# Two affine layer norms, separate query/key/value projections, the attention products,
# the output projection, the MLP around ``activation`` (already costed by the caller),
# two residual adds, and, with ``layer_scale``, the two per-channel LayerScale
# multiplies DINOv2 applies before each add.
def _vit_layer_cost(
    *,
    channels: int,
    channels_mlp: int,
    num_heads: int,
    seq_len: int,
    batch_size: int,
    layer_scale: bool,
    activation: Cost,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one pre-norm ViT block."""
    rows = seq_len * batch_size
    norms = cost(
        LayerNorm.Config(channels, elementwise_affine=True),
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).tile(2, copies=2)
    square = matmul_cost(
        channels_in=channels,
        channels_out=channels,
        bias=True,
        rows=rows,
        dtype=dtype,
    )
    attention = attention_kernel_cost(
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
        num_heads=num_heads,
        channels_head=channels // num_heads,
    )
    mlp = matmul_cost(
        channels_in=channels,
        channels_out=channels_mlp,
        bias=True,
        rows=rows,
        dtype=dtype,
    ) + matmul_cost(
        channels_in=channels_mlp,
        channels_out=channels,
        bias=True,
        rows=rows,
        dtype=dtype,
    )
    residual_adds = elementwise_cost(
        primal=2 * rows * channels,
        adjoint=2 * rows * channels,
        channels=2 * channels,
        rows=rows,
        inputs=2,
        dtype=dtype,
    )
    # A multiply forward; back, the input gradient and the parameter product.
    scales = (
        elementwise_cost(
            primal=rows * channels,
            adjoint=2 * rows * channels,
            channels=channels,
            params=channels,
            rows=rows,
            dtype=dtype,
        ).tile(2, copies=2)
        if layer_scale
        else Cost()
    )
    return (
        norms
        + square.tile(4, copies=4)
        + attention
        + mlp
        + activation
        + residual_adds
        + scales
    )


# Torch's exact-erf GELU -- the reference's, and HF's ``"gelu"`` -- is priced here at
# eight operations each way, the count ``priml.baselines.sudoku`` gives exact GELU,
# rather than registered on torch's function with :func:`~priml.cost.set_cost`:
# registering would impose this module's count on every importer. Any other activation
# must cost itself.
def _activation_cost(
    activation: TensorFn,
    *,
    channels: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost ``activation`` over ``channels`` elements."""
    if activation is functional.gelu:
        return map_cost(primal=8, adjoint=8)(channels=channels, dtype=dtype)
    return cost(activation, channels=channels, dtype=dtype)


def _concatenate_cost(*, elements: int, dtype: torch.dtype | None) -> Cost:
    """Cost a concatenation writing ``elements``: a copy, and its split back."""
    return traffic(
        "primal",
        kernel="selection",
        elements=2 * elements,
        dtype=dtype,
    ) + traffic("adjoint", kernel="selection", elements=2 * elements, dtype=dtype)


# A width pass then a height pass, each output a weighted sum of ``taps`` inputs
# (``taps`` multiplies, ``taps - 1`` adds). The cubic kernel spans four inputs;
# antialiasing a downscale widens it by the scale, as torch's ``_upsample_bicubic2d_aa``
# does. The adjoint scatters the same weights back, so it costs the same.
def _bicubic_cost(
    *,
    planes: int,
    grid_in: tuple[int, int],
    grid_out: tuple[int, int],
    antialias: bool,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost a separable bicubic resample of ``planes`` images and its transpose."""
    (height_in, width_in), (height_out, width_out) = grid_in, grid_out
    taps_height = _bicubic_taps(height_in, size_out=height_out, antialias=antialias)
    taps_width = _bicubic_taps(width_in, size_out=width_out, antialias=antialias)
    middle = planes * height_in * width_out
    out = planes * height_out * width_out
    flops = (2 * taps_width - 1) * middle + (2 * taps_height - 1) * out
    elements = planes * height_in * width_in + 2 * middle + out
    return traffic(
        "primal",
        kernel="elementwise",
        elements=elements,
        flops=flops,
        dtype=dtype,
    ) + traffic(
        "adjoint",
        kernel="elementwise",
        elements=elements,
        flops=flops,
        dtype=dtype,
    )


def _bicubic_taps(size_in: int, size_out: int, *, antialias: bool) -> int:
    """Return how many inputs one bicubic output reads along an axis."""
    if not antialias or size_in <= size_out:
        return 4
    return min(size_in, math.ceil(4 * size_in / size_out))
