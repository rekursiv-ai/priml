"""RAE: a frozen representation encoder with a trained ViT pixel decoder.

The encoder is Hugging Face's ``Dinov2WithRegistersModel`` with its final
layer norm stripped of its affine, as the reference strips it; its patch tokens
ARE the latent. The decoder is the reference's MAE-style ``GeneralDecoder``,
ported natively so the published state dict loads under ``strict=True`` with
no dependence on Hugging Face's internal ViTMAE classes. :class:`RAE` is the
configured wrapper every consumer uses.

Deviations from the reference, each deliberate:

- ``encode`` takes uint8 and divides by 255 first, which is what the
  reference's ``ToTensor`` callers feed it.
- No latent normalization: the autoencoder returns RAW latents, and the
  published statistics are ``RAE.Config.latent_norm``.
- No ``noise_tau`` noising: it is a decoder-training augmentation, inert at
  the published inference setting (``noise_tau: 0``).
- ``decode`` clamps to ``[0, 1]``, as the ``Autoencoder`` protocol requires;
  the reference leaves clamping to its caller.
- A latent grid the decoder was not built for raises; the reference silently
  resizes it bilinearly.
- The CLS and register tokens dropped before the patches are counted from
  ``num_register_tokens``; the reference hardcodes 5, the same count for every
  published checkpoint.
- Inference only: the decoder's dropouts (all ``p=0``), gradient
  checkpointing, attention outputs, and the ``drop_cls_token`` and
  ``interpolate_pos_encoding`` paths are not ported.

MIT license and attribution: ``priml/model/vision_ae/RAE-LICENSE``.

``scripts/reference_parity.py`` proves it bit-identical to the reference below.

References:
  https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/rae.py
  https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/decoders/decoder.py
  https://huggingface.co/nyu-visionx/RAE-collections
    Zheng et al. 2025. Diffusion Transformers with Representation
    Autoencoders.

"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, Self, cast, override, runtime_checkable

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

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
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import conv_cost
from priml.model.custom_types import ChannelsIn, propagate_attr
from priml.model.norm import LayerNorm
from priml.model.vision_ae.checkpoint import HubFile
from priml.model.vision_ae.custom_types import (
    CheckpointFile,
    LatentNormalizer,
    require_uint8,
)
from priml.model.vision_ae.latent_norm import ElementwiseLatentStats


if TYPE_CHECKING:
    from collections.abc import Callable

    from safetensors.torch import load_file

    import transformers
else:
    from wrapt import lazy_import

    load_file = lazy_import("safetensors.torch", "load_file")
    transformers = lazy_import("transformers")


@runtime_checkable
class PatchEncoder(Protocol):
    """Maps normalized pixels to one token per patch, in row-major order."""

    def __call__(self, image: Tensor, /) -> Tensor:
        """Return ``[B, N, C]`` patch tokens for ``[B, 3, H, W]`` normalized pixels."""
        ...


class PatchEncoderConfig(Makeable[PatchEncoder], Protocol):
    """What :class:`RAE` reads from its encoder's config without building it."""

    channels_hidden: int
    """Token width, which is the latent's channel count."""

    patch_size: int
    """Pixels per patch side."""

    pixel_mean: tuple[float, float, float]
    """Per-channel mean subtracted from ``[0, 1]`` pixels."""

    pixel_std: tuple[float, float, float]
    """Per-channel deviation the centered pixels are divided by."""


class Dinov2WithRegisters(nn.Module):
    """DINOv2 with registers, final norm without affine, leading tokens dropped.

    Built from literal architecture values rather than a Hub config lookup, so
    building touches the network only for the weights.
    """

    class Config(Fig["Dinov2WithRegisters"]):
        """Architecture, attention kernel, pixel statistics, and weights."""

        channels_hidden: int = 768
        """Token width (HF ``hidden_size``)."""

        num_layers: int = 12
        """Transformer blocks (HF ``num_hidden_layers``)."""

        heads: int = 12
        """Attention heads (HF ``num_attention_heads``)."""

        expansion: int = 4
        """MLP width over ``channels_hidden`` (HF ``mlp_ratio``, which HF types as an integer)."""

        patch_size: int = 14
        """Pixels per patch side."""

        image_size: int = 518
        """Side the position table was trained at; other sides interpolate it."""

        num_register_tokens: int = 4
        """Register tokens inserted after CLS; both are dropped before the patches."""

        layerscale_value: float = 1.0
        """Initial LayerScale; the checkpoint overwrites it."""

        eps: float = 1e-6
        """Layer-norm epsilon (HF ``layer_norm_eps``)."""

        activation: TensorFn = functional.gelu
        """MLP activation; the checkpoint's HF ``"gelu"`` is this exact-erf form."""

        qkv_bias: bool = True
        """Whether the query, key, and value projections carry a bias."""

        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """ImageNet mean, from the checkpoint's ``preprocessor_config.json``."""

        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """ImageNet deviation, from the checkpoint's ``preprocessor_config.json``."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: HubFile.Config(
                repo_id="facebook/dinov2-with-registers-base",
                filename="model.safetensors",
                revision="a1d738ccfa7ae170945f210395d99dde8adb1805",
                sha256="7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362",
            ),
        )
        """Published safetensors, final-norm affine included.

        ``None`` keeps the random initialization.
        """

        def cost(
            self,
            *,
            input_grid: int | tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward and its adjoint, as any trainable module.

            Counts the patch convolution, the CLS/register concatenations, the
            position-table interpolation whenever HF runs it (a patch count
            other than the table's, or a non-square input), the position add,
            every layer, and the affine-free final norm, all over the CLS and
            register tokens too. HF interpolates in float32 whatever ``dtype``,
            casting the table there and the result back. Dropping the leading
            tokens is a view. The mask token is owned but never read.

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
            # Analytical rather than a hooked meta-device forward (the imagenet
            # precedent): every value HF builds from is a field here, so the
            # formula reads nothing HF owns and costing never imports
            # ``transformers``; the FlopCounterMode test pins it to HF's kernels.
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
                qkv_bias=self.qkv_bias,
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
        hf_config = transformers.Dinov2WithRegistersConfig(
            hidden_size=config.channels_hidden,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.heads,
            mlp_ratio=config.expansion,
            # A placeholder for HF's string-only schema; ``config.activation``
            # replaces it on every layer below.
            hidden_act="gelu",
            layer_norm_eps=config.eps,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=3,
            qkv_bias=config.qkv_bias,
            layerscale_value=config.layerscale_value,
            num_register_tokens=config.num_register_tokens,
            use_swiglu_ffn=False,
            # Inert: the wrapper never leaves evaluation mode.
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            # The reference's ``from_pretrained`` resolves the fused ``"sdpa"``,
            # which ``host_agnostic_numerics`` cannot reach inside; ``"eager"`` is
            # the plain matmul-softmax sequence. Measured on the tiny parity
            # fixture under ``host_agnostic_numerics``, the two differ in 6 of 64
            # latent values by at most 3 float32 ULP.
            attn_implementation="eager",
        )
        self.encoder: transformers.Dinov2WithRegistersModel = (
            transformers.Dinov2WithRegistersModel(hf_config)
        )
        mlp_class = transformers.models.dinov2_with_registers.modeling_dinov2_with_registers.Dinov2WithRegistersMLP
        for mlp in self.encoder.modules():
            if isinstance(mlp, mlp_class):
                # HF registered its ``GELUActivation`` as a child module; a plain
                # function cannot overwrite a child, so the child goes first.
                del mlp.activation
                mlp.activation = config.activation
        if config.checkpoint is not None:
            state = load_file(str(config.checkpoint.make().path()))
            self.encoder.load_state_dict(state)
        # After loading, as the reference strips it: the checkpoint carries the
        # affine, and the latent is the normalized token without it.
        layernorm = cast("nn.LayerNorm", self.encoder.layernorm)
        layernorm.elementwise_affine = False
        layernorm.register_parameter("weight", None)
        layernorm.register_parameter("bias", None)
        # CLS, then the registers: counted, not restated, so changing the register
        # count cannot leave a register among the latent's patch tokens.
        self.num_leading_tokens = 1 + config.num_register_tokens

    @override
    def forward(self, image: Tensor) -> Tensor:
        """Return the patch tokens.

        Args:
          image: ``[B, 3, H, W]`` normalized pixels.

        Returns:
          tokens: ``[B, (H / patch) * (W / patch), channels_hidden]``.

        """
        # Hugging Face's ``__call__`` is typed by a ParamSpec its stubs leave open.
        encode = cast("Callable[[Tensor], _EncoderOutput]", self.encoder)
        return encode(image).last_hidden_state[:, self.num_leading_tokens :]


class ViTMAESelfAttention(nn.Module):
    """Multi-head attention as plain matmul, scale, softmax, matmul."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"{num_heads} heads do not divide width {channels}.")
        self.num_heads = num_heads
        self.channels_head = channels // num_heads
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """Attend over tokens.

        Args:
          hidden: ``[B, N, C]`` tokens.

        Returns:
          context: ``[B, N, C]`` attended values, heads concatenated.

        """
        query = self._heads(self.query(hidden))
        key = self._heads(self.key(hidden))
        value = self._heads(self.value(hidden))
        # Divide by the root rather than multiply by its reciprocal: the
        # reference does, and the two round differently.
        root = math.sqrt(self.channels_head)  # noqa: TID251 -- The reference's correctly rounded root; ``** 0.5`` goes through ``pow``, which need not round the same.
        scores = torch.matmul(query, key.transpose(-1, -2)) / root
        context = torch.matmul(functional.softmax(scores, dim=-1), value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.view(*context.shape[:-2], self.num_heads * self.channels_head)

    def _heads(self, projected: Tensor) -> Tensor:
        """Split ``[B, N, C]`` into ``[B, heads, N, channels_head]``."""
        split = projected.view(
            *projected.shape[:-1],
            self.num_heads,
            self.channels_head,
        )
        return split.permute(0, 2, 1, 3)


class ViTMAESelfOutput(nn.Module):
    """Attention output projection; the residual lives in the layer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(channels, channels)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """Project the attended values."""
        return self.dense(hidden)


class ViTMAEAttention(nn.Module):
    """Attention followed by its output projection."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(channels, num_heads=num_heads)
        self.output = ViTMAESelfOutput(channels)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """Attend and project."""
        return self.output(self.attention(hidden))


class ViTMAEIntermediate(nn.Module):
    """MLP expansion and activation."""

    def __init__(
        self,
        channels: int,
        channels_hidden: int,
        activation: TensorFn,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(channels, channels_hidden)
        self.activation = activation

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """Expand and activate."""
        return self.activation(self.dense(hidden))


class ViTMAEOutput(nn.Module):
    """MLP contraction plus the second residual."""

    def __init__(self, channels: int, channels_hidden: int) -> None:
        super().__init__()
        self.dense = nn.Linear(channels_hidden, channels)

    @override
    def forward(self, hidden: Tensor, residual: Tensor) -> Tensor:
        """Contract and add the residual."""
        return self.dense(hidden) + residual


class ViTMAELayer(nn.Module):
    """Pre-norm transformer block with the reference's module names."""

    def __init__(
        self,
        *,
        channels: int,
        channels_hidden: int,
        num_heads: int,
        activation: TensorFn,
        eps: float,
    ) -> None:
        super().__init__()
        # Registration order is the reference's; it fixes the checkpoint key
        # order and the draw order of a random initialization.
        self.attention = ViTMAEAttention(channels, num_heads=num_heads)
        self.intermediate = ViTMAEIntermediate(
            channels,
            channels_hidden=channels_hidden,
            activation=activation,
        )
        self.output = ViTMAEOutput(channels, channels_hidden=channels_hidden)
        self.layernorm_before = nn.LayerNorm(channels, eps=eps)
        self.layernorm_after = nn.LayerNorm(channels, eps=eps)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """Run attention and MLP, each with a residual.

        Args:
          hidden: ``[B, N, C]`` tokens.

        Returns:
          hidden: ``[B, N, C]`` tokens.

        """
        hidden = self.attention(self.layernorm_before(hidden)) + hidden
        return self.output(
            self.intermediate(self.layernorm_after(hidden)),
            residual=hidden,
        )


class GeneralDecoder(nn.Module):
    """ViT pixel decoder: latent tokens plus a learned CLS token to patches.

    The position table is the fixed 2D sine-cosine one, stored as a frozen
    parameter so it lives in the checkpoint as the reference saves it.
    """

    class Config(Fig["GeneralDecoder"]):
        """ViT-XL dimensions by default, and weights."""

        channels_in: int = -1
        """Latent token width; :class:`RAE` sets the encoder's."""

        num_patches: int = -1
        """Latent tokens, a square; :class:`RAE` sets the encoder's grid."""

        channels_hidden: int = 1152
        """Transformer width."""

        channels_hidden_mlp: int = 4096
        """MLP hidden width."""

        num_layers: int = 28
        """Transformer blocks."""

        heads: int = 16
        """Attention heads."""

        patch_size: int = 16
        """Output pixels per latent token side."""

        eps: float = 1e-12
        """Layer-norm epsilon, the ViTMAE configuration's."""

        activation: TensorFn = functional.gelu
        """MLP activation; the reference's ``"gelu"`` is the exact erf form."""

        checkpoint: Makeable[CheckpointFile] | None = None
        """Bare decoder state dict; ``None`` keeps the random initialization."""

        @override
        def finalize(self) -> Self:
            n = self.num_patches
            if n <= 0 or math.isqrt(n) ** 2 != n:
                raise ValueError(
                    f"num_patches {self.num_patches} must be a positive square.",
                )
            if self.channels_hidden % 4:
                raise ValueError(
                    f"channels_hidden {self.channels_hidden} must be divisible by "
                    "4 for the 2D sine-cosine table.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on ``num_patches`` tokens and its adjoint.

            Every layer and the prediction head run over the CLS token too;
            dropping it afterwards is a view. The position table is owned but
            frozen, so it adds without a gradient; ``unpatchify`` is not part
            of the forward and is the caller's.

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
                qkv_bias=True,
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
        self.num_patches = config.num_patches
        self.patch_size = config.patch_size
        self.grid_size = math.isqrt(config.num_patches)
        # Registration order is the reference's, trainable CLS token last.
        self.decoder_embed = nn.Linear(config.channels_in, config.channels_hidden)
        self.decoder_pos_embed = nn.Parameter(
            sincos_position_table(
                config.channels_hidden,
                grid=self.grid_size,
            ).unsqueeze(0),
            requires_grad=False,
        )
        self.decoder_layers = nn.ModuleList(
            ViTMAELayer(
                channels=config.channels_hidden,
                channels_hidden=config.channels_hidden_mlp,
                num_heads=config.heads,
                activation=config.activation,
                eps=config.eps,
            )
            for _ in range(config.num_layers)
        )
        self.decoder_norm = nn.LayerNorm(config.channels_hidden, eps=config.eps)
        self.decoder_pred = nn.Linear(config.channels_hidden, config.patch_size**2 * 3)
        self.trainable_cls_token = nn.Parameter(
            torch.zeros(1, 1, config.channels_hidden),
        )
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

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Predict each patch's pixels.

        Args:
          tokens: ``[B, num_patches, channels_in]`` latent tokens, row-major.

        Returns:
          logits: ``[B, num_patches, patch_size**2 * 3]`` normalized pixels.

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
        logits = self.decoder_pred(self.decoder_norm(hidden))
        return logits[:, 1:, :]

    def unpatchify(self, logits: Tensor) -> Tensor:
        """Tile per-patch pixels into an image.

        Args:
          logits: ``[B, num_patches, patch_size**2 * 3]``.

        Returns:
          image: ``[B, 3, side, side]`` with ``side = patch_size * sqrt(num_patches)``.

        """
        grid, patch = self.grid_size, self.patch_size
        patches = logits.reshape(logits.shape[0], grid, grid, patch, patch, 3)
        patches = torch.einsum("nhwpqc->nchpwq", patches)
        return patches.reshape(logits.shape[0], 3, grid * patch, grid * patch)


class RAE(nn.Module):
    """RAE behind the ``Autoencoder`` contract.

    Deterministic: the latent is the encoder's patch-token grid itself. Frozen:
    the wrapper stays in evaluation mode whatever its parent does. ``decode``
    always returns ``image_size`` pixels, whatever size was encoded.
    """

    class Config(Fig["RAE"]):
        """Geometry, encoder, decoder, and the published latent normalizer.

        The defaults are the published DINOv2-B (registers) model.
        """

        image_size: int = 256
        """Side of the square image the decoder reconstructs."""

        encoder_image_size: int = 224
        """Side the encoder sees; other inputs are resized bicubically first."""

        encoder: PatchEncoderConfig = field(default_factory=Dinov2WithRegisters.Config)
        """Representation encoder; its tokens are the latent."""

        decoder: GeneralDecoder.Config = field(
            default_factory=lambda: GeneralDecoder.Config(
                checkpoint=HubFile.Config(
                    repo_id="nyu-visionx/RAE-collections",
                    filename="decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
                    revision="1be4f03273523431f099a934da4cf1940dc6039f",
                    sha256="5fedf7c9660476a709e122cef18385c917532914a23f034388e1c1a52bde2be6",
                ),
            ),
        )
        """Pixel decoder, sized from the encoder's tokens; the published ViT-XL weights."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: ElementwiseLatentStats.Config(
                stats=HubFile.Config(
                    repo_id="nyu-visionx/RAE-collections",
                    filename="stats/dinov2/wReg_base/imagenet1k/stat.pt",
                    revision="1be4f03273523431f099a934da4cf1940dc6039f",
                    sha256="84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59",
                ),
            ),
        )
        """The published ImageNet statistics; the autoencoder itself never applies them."""

        @override
        def finalize(self) -> Self:
            if self.encoder_image_size % self.encoder.patch_size:
                raise ValueError(
                    f"encoder_image_size {self.encoder_image_size} must be "
                    f"divisible by the encoder patch {self.encoder.patch_size}.",
                )
            grid = self.encoder_image_size // self.encoder.patch_size
            propagate_attr(
                self.decoder,
                name="channels_in",
                value=self.encoder.channels_hidden,
                protocol=ChannelsIn,
            )
            propagate_attr(self.decoder, name="num_patches", value=grid * grid)
            if self.decoder.patch_size * grid != self.image_size:
                raise ValueError(
                    f"image_size {self.image_size} must be the decoder patch "
                    f"{self.decoder.patch_size} times the latent grid {grid}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one ``decode(encode(image))`` round trip at ``image_size``.

            ``encode``: the uint8-to-unit map, the bicubic resize to
            ``encoder_image_size`` when the sides differ, the pixel
            normalization, and the encoder. ``decode``: the decoder, its
            ``unpatchify`` copy, the denormalization, and the clamp. The
            grid reshapes between them are views. Preprocessing is float32
            whatever ``dtype``, as ``encode`` casts it.

            The children cost as ordinary modules, adjoint included; the
            wrapper is frozen and never leaves evaluation mode, so it keeps
            their forward work only, and every parameter it owns.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype of the encoder and decoder; ``None``
                is torch's default.
              **kwargs: The open bus, forwarded to both children.

            Returns:
              cost: Forward FLOPs and logical bytes; no adjoint cells.

            Raises:
              TypeError: The encoder or decoder has no cost.

            """
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

        def latent_shape(self) -> tuple[int, int, int]:
            """Return ``(channels_hidden, grid, grid)`` at ``encoder_image_size``."""
            grid = self.encoder_image_size // self.encoder.patch_size
            return self.encoder.channels_hidden, grid, grid

    def __init__(self, config: Config) -> None:
        super().__init__()
        mean, std = config.encoder.pixel_mean, config.encoder.pixel_std
        if any(math.isnan(v) or math.isinf(v) for v in (*mean, *std)) or 0 in std:
            raise ValueError(
                "RAE needs finite pixel statistics and a nonzero std; got mean "
                f"{mean}, std {std}.",
            )
        self.encoder_image_size = config.encoder_image_size
        self.latent_shape = config.latent_shape()
        # Encoder before decoder, as the reference registers them.
        self.encoder = config.encoder.make()
        self.encoder_mean: Tensor
        self.encoder_std: Tensor
        self.register_buffer(
            "encoder_mean",
            torch.tensor(mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor(std).view(1, 3, 1, 1),
            persistent=False,
        )
        self.decoder = config.decoder.make()
        self.eval()
        self.requires_grad_(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, image: Tensor, /) -> Tensor:
        """Return the encoder's patch tokens as a grid.

        Args:
          image: ``[B, 3, H, W]`` uint8 RGB.

        Returns:
          latent: ``[B, channels_hidden, grid, grid]`` raw latent.

        Raises:
          TypeError: ``image`` is not uint8.

        """
        require_uint8(image)
        pixels = rgb2float(image, float_dtype=torch.float32, unit_interval=True)
        _, _, height, width = pixels.shape
        side = self.encoder_image_size
        if height != side or width != side:
            pixels = functional.interpolate(
                pixels,
                size=(side, side),
                mode="bicubic",
                align_corners=False,
            )
        tokens = self.encoder((pixels - self.encoder_mean) / self.encoder_std)
        batch, count, channels = tokens.shape
        grid = math.isqrt(count)
        return tokens.transpose(1, 2).view(batch, channels, grid, grid)

    def decode(self, latent: Tensor, /) -> Tensor:
        """Decode a raw latent to RGB in ``[0, 1]``.

        Args:
          latent: ``[B, channels_hidden, grid, grid]`` raw latent.

        Returns:
          image: ``[B, 3, image_size, image_size]`` float RGB, clamped.

        Raises:
          ValueError: The grid is not the one the decoder was built for.

        """
        batch, channels, height, width = latent.shape
        if (channels, height, width) != self.latent_shape:
            raise ValueError(
                f"RAE decodes latents of shape {self.latent_shape}, got "
                f"{(channels, height, width)}; the reference would reshape them silently.",
            )
        tokens = latent.view(batch, channels, height * width).transpose(1, 2)
        pixels = self.decoder.unpatchify(self.decoder(tokens))
        return (pixels * self.encoder_std + self.encoder_mean).clamp(0, 1)


def rae_dinov2_base() -> RAE.Config:
    """Return the published DINOv2-B (registers) RAE, the :class:`RAE.Config` defaults.

    References:
      https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/configs/stage1/pretrained/DINOv2-B.yaml

    Returns:
      config: Encoder, ``ViTXL_n08`` decoder, and ImageNet latent statistics.

    """
    return RAE.Config()


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
    qkv_bias: bool,
    layer_scale: bool,
    activation: Cost,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one pre-norm ViT block: HF's DINOv2 layer, or the port's ViTMAE one."""
    rows = seq_len * batch_size
    norms = cost(
        LayerNorm.Config(channels, elementwise_affine=True),
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).tile(2, copies=2)
    qkv = matmul_cost(
        channels_in=channels,
        channels_out=channels,
        bias=qkv_bias,
        rows=rows,
        dtype=dtype,
    ).tile(3, copies=3)
    attention = attention_kernel_cost(
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
        num_heads=num_heads,
        channels_head=channels // num_heads,
    )
    projections = (
        matmul_cost(
            channels_in=channels,
            channels_out=channels,
            bias=True,
            rows=rows,
            dtype=dtype,
        )
        + matmul_cost(
            channels_in=channels,
            channels_out=channels_mlp,
            bias=True,
            rows=rows,
            dtype=dtype,
        )
        + matmul_cost(
            channels_in=channels_mlp,
            channels_out=channels,
            bias=True,
            rows=rows,
            dtype=dtype,
        )
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
    return norms + qkv + attention + projections + activation + residual_adds + scales


# Torch's exact-erf GELU -- the reference's, and HF's ``"gelu"`` -- is priced here at
# eight operations each way, the count ``priml.baselines.sudoku`` gives exact GELU,
# rather than registered on torch's function with :func:`~priml.cost.set_cost`:
# registering would impose this module's count on every importer, and ``cost(gelu)``
# would work only once this module had been imported. Any other activation must cost
# itself.
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
# does. The adjoint scatters the same weights back, so it costs the same. Per-output
# weights are not counted.
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


class _EncoderOutput(Protocol):
    """The part of Hugging Face's model output the encoder reads."""

    last_hidden_state: Tensor
