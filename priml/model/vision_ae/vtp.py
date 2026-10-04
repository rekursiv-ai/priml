"""VTP: MiniMax's visual tokenizer, as the 64-channel autoencoder LightningDiT trains on.

Only the reconstruction path is ported: the DINOv3-style vision trunk with its
feature bottleneck, and the pixel decoder. The text tower and CLIP projection
the published checkpoints also carry are discarded at load, and the
training-only machinery (stochastic depth, RoPE coordinate augmentation, token
masking) is dropped, since none of it runs in the reference's evaluation
forward. Module and parameter names are the reference's, so the published
``trunk.*`` and ``pixel_decoder.*`` keys load unchanged, and every forward issues
the reference's operations in the reference's order -- including its bfloat16
rotary embedding, which rounds the queries and keys to bfloat16 whatever the
activation dtype. :class:`VTP` is the configured wrapper every consumer uses.

MIT license with an attribution clause: ``priml/model/vision_ae/VTP-LICENSE``.

``scripts/reference_parity.py`` proves it bit-identical to the reference below.

References:
  https://github.com/MiniMax-AI/VTP/tree/5ce1eb67010fff3c1eed483352483be6a1838556/vtp/models
  https://huggingface.co/MiniMaxAI/VTP-Large-f16d64
    Yao et al. 2025. Towards Scalable Pre-training of Visual Tokenizers for
    Generation.

"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, Final, Self, cast, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.math.basic import ceil_multiple
from priml.math.pixel import rgb2float
from priml.model import norm as priml_norm
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import conv_cost
from priml.model.custom_types import ChannelsIn, TensorModule, propagate_attr
from priml.model.swiglu import SwiGLU
from priml.model.vision_ae.checkpoint import HubFile, UrlFile
from priml.model.vision_ae.custom_types import (
    CheckpointFile,
    LatentNormalizer,
    require_uint8,
)
from priml.model.vision_ae.latent_norm import ChannelLatentStats


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from safetensors.torch import load_file
else:
    from wrapt import lazy_import

    load_file = lazy_import("safetensors.torch", "load_file")


def rope_rotate_half(x: Tensor) -> Tensor:
    """Return ``[-x2, x1]`` for the halves ``x1, x2`` of the last axis."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Rotate ``x`` by the angles whose sines and cosines are given."""
    return (x * cos) + (rope_rotate_half(x) * sin)


class RMSNorm(nn.Module):
    """The reference's RMSNorm: normalize in float32, cast back, then scale.

    Not :class:`priml.model.norm.RMSNorm`: the two agree bit for bit in float32,
    but on bfloat16 activations -- the encoder under ``dtype_autocast`` -- this
    one promotes to the float32 weight and returns different bits.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        """Set the scale to one."""
        nn.init.constant_(self.weight, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        normed = x.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class PatchEmbed(nn.Module):
    """Non-overlapping patches to tokens: ``[B, C, H, W] -> [B, h, w, D]``."""

    def __init__(self, *, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        height, width = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        return x.reshape(-1, height, width, self.embed_dim)

    def reset_parameters(self) -> None:
        """Draw weight and bias from the reference's fan-in uniform."""
        # The reference's own spelling: ``pow`` is not required to round like the
        # correctly-rounded ``sqrt``, and the bound scales every initial draw.
        bound = math.sqrt(1 / (self.in_chans * (self.patch_size**2)))  # noqa: TID251 -- Matches the reference's initialization bits.
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)


class RopePositionEmbedding(nn.Module):
    """Axial 2D rotary embedding over coordinates normalized per side to ``[-1, 1]``.

    Computed in ``dtype`` -- bfloat16 for every published checkpoint -- as the
    reference computes it; ``periods`` is a persistent buffer the checkpoints
    store in that dtype.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if embed_dim % (4 * num_heads):
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 4 * num_heads "
                f"({4 * num_heads}).",
            )
        self.base = base
        self.channels_head = embed_dim // num_heads
        self.dtype = dtype
        self.periods: Tensor
        self.register_buffer(
            "periods",
            torch.empty(self.channels_head // 4, dtype=dtype),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Set ``periods`` to ``base ** (2i / (channels_head / 2))``."""
        exponents = torch.arange(
            self.channels_head // 4,
            device=self.periods.device,
            dtype=self.dtype,
        )
        self.periods.data = self.base ** (2 * exponents / (self.channels_head // 2))

    @override
    def forward(self, *, height: int, width: int) -> tuple[Tensor, Tensor]:
        device, dtype = self.periods.device, self.dtype
        coords_h = torch.arange(0.5, height, device=device, dtype=dtype) / height
        coords_w = torch.arange(0.5, width, device=device, dtype=dtype) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos


class SelfAttention(nn.Module):
    """Multi-head self-attention with rotary queries and keys after a prefix."""

    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    @override
    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        qkv = self.qkv(x)
        batch, tokens, _ = qkv.shape
        channels = self.qkv.in_features
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
        q, k = self.apply_rope(q, k, rope)
        x = functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape([batch, tokens, channels])
        return self.proj(x)

    def apply_rope(
        self,
        q: Tensor,
        k: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Rotate the trailing ``sin.shape[-2]`` tokens in the rope's dtype.

        Args:
          q: ``[B, heads, N, channels_head]`` queries.
          k: ``[B, heads, N, channels_head]`` keys.
          rope: ``(sin, cos)``, each ``[n, channels_head]`` with ``n <= N``.

        Returns:
          rotated: Queries and keys, cast back to their own dtypes.

        """
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        q = q.to(dtype=sin.dtype)
        k = k.to(dtype=sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        q = torch.cat(
            (q[:, :, :prefix, :], rope_apply(q[:, :, prefix:, :], sin, cos)),
            dim=-2,
        )
        k = torch.cat(
            (k[:, :, :prefix, :], rope_apply(k[:, :, prefix:, :], sin, cos)),
            dim=-2,
        )
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


class SwiGLUFFN(nn.Module):
    """``w3(silu(w1(x)) * w2(x))``, hidden width ``2/3`` of nominal, aligned up to 8.

    Not :class:`priml.model.swiglu.SwiGLU`, whose fused ``up_proj`` and
    ``down_proj`` would neither load the checkpoint's biased ``w1``, ``w2``, and
    ``w3`` nor take this width rule.
    """

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        swiglu_hidden_features = _swiglu_channels_hidden(hidden_features)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=True)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=True)
        self.w3 = nn.Linear(swiglu_hidden_features, in_features, bias=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(functional.silu(x1) * x2)


type NormFactory = Callable[[int], TensorModule]
"""Builds a normalization over the given channel width."""


class SelfAttentionBlock(nn.Module):
    """Pre-norm attention then SwiGLU, each added back to the residual stream."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        expansion: float,
        norm: NormFactory,
    ) -> None:
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads)
        self.norm2 = norm(dim)
        self.mlp = SwiGLUFFN(dim, int(dim * expansion))

    @override
    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        x_attn = x + self.attn(self.norm1(x), rope)
        return x_attn + self.mlp(self.norm2(x_attn))


def init_weights_vit(module: nn.Module) -> None:
    """Apply the reference's per-module initialization to one module."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, (nn.LayerNorm, PatchEmbed, RMSNorm)):
        module.reset_parameters()


def init_weights_post(module: nn.Module) -> None:
    """Apply ``VTPModel``'s post-construction pass to one module.

    Hugging Face's ``post_init`` runs it over the whole model after the trunk
    and decoder have initialized themselves, so its draws replace theirs.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # Unconditional, as the reference's: a bias-free LayerNorm raises there too.
        nn.init.ones_(module.weight)
        nn.init.zeros_(cast("Tensor", module.bias))
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


class DinoVisionTransformerWithBottleneck(nn.Module):
    """VTP's vision trunk: pixels to bottlenecked patch latents.

    A CLS token joins the patch tokens through the RMSNorm SwiGLU blocks and the
    final norm, then is dropped; each normalized patch token is projected, bias
    free, to ``channels_out``.
    """

    class Config(Fig["DinoVisionTransformerWithBottleneck"]):
        """Trunk size."""

        patch_size: int = 16
        """Side of a square patch; the latent grid is the image divided by it."""

        channels_hidden: int = 1024
        """Token width."""

        num_layers: int = 24
        """Transformer blocks."""

        heads: int = 16
        """Attention heads; ``channels_hidden`` must be divisible by ``4 * heads``."""

        expansion: float = 4.0
        """Nominal FFN width over ``channels_hidden``, before SwiGLU's 2/3 and alignment."""

        channels_out: int = 64
        """Bottleneck width: latent channels."""

        @override
        def finalize(self) -> Self:
            # The reference skips the projection at equal widths; this port always
            # builds it, so it refuses the one case where the two would differ.
            if self.channels_out == self.channels_hidden:
                raise ValueError(
                    f"channels_out {self.channels_out} must differ from "
                    f"channels_hidden {self.channels_hidden}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            image_size: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one trainable forward and backward on ``[batch_size, 3, S, S]``.

            An ordinary module: every parameter and the input receive
            gradients, so both phases are counted; :class:`VTP`, which freezes
            it, keeps the primal alone. The patch embedding, the blocks over
            the ``(S / patch_size) ** 2`` patches plus the class token, the
            final norm, and the bias-free bottleneck over the patches. Layout
            copies (the class-token concatenation, the head transposes, the
            output reshape) move no arithmetic and are not counted, as in
            :class:`priml.model.attention.self_attention.SelfAttention`.

            Args:
              image_size: Side ``S`` of the square input image, in pixels.
              batch_size: Images per invocation.
              dtype: Activation dtype; ``None`` is torch's default. The rotary
                embedding runs in bfloat16 whatever it is.
              **kwargs: The open bus, unread.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            Raises:
              ValueError: ``image_size`` is not a multiple of ``patch_size``.

            """
            del kwargs
            if image_size % self.patch_size:
                raise ValueError(
                    f"image_size {image_size} must be divisible by "
                    f"patch_size {self.patch_size}.",
                )
            side = image_size // self.patch_size
            width = self.channels_hidden
            patch_embed = conv_cost(
                channels_in=3,
                channels_out=width,
                kernel_size=self.patch_size,
                ndim=2,
                groups=1,
                bias=True,
                input_grid=(image_size, image_size),
                batch_size=batch_size,
                stride=self.patch_size,
                padding=0,
                dtype=dtype,
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
                norm=priml_norm.RMSNorm.Config(
                    width,
                    eps=1e-5,
                    elementwise_affine=True,
                ),
                channels=width,
                heads=self.heads,
                expansion=self.expansion,
                num_layers=self.num_layers,
                side=side,
                prefix=1,
                batch_size=batch_size,
                dtype=dtype,
            )
            bottleneck = matmul_cost(
                channels_in=width,
                channels_out=self.channels_out,
                rows=batch_size * side * side,
                dtype=dtype,
            )
            return patch_embed + class_token + blocks + bottleneck

    def __init__(self, config: Config) -> None:
        super().__init__()
        dim = config.channels_hidden
        self.embed_dim = dim
        self.patch_size = config.patch_size
        # Registration order is the reference's: it fixes the state_dict order and
        # the draw order of the initialization below.
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=dim,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.rope_embed = RopePositionEmbedding(dim, num_heads=config.heads)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(
                dim,
                num_heads=config.heads,
                expansion=config.expansion,
                norm=RMSNorm,
            )
            for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(dim)
        self.mask_token = nn.Parameter(torch.empty(1, dim))
        self.feature_bottleneck = nn.Linear(dim, config.channels_out, bias=False)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mask_token)
        for module in self.modules():
            init_weights_vit(module)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Encode normalized pixels.

        Args:
          x: ``[B, 3, H, W]`` ImageNet-normalized RGB.

        Returns:
          latent: ``[B, channels_out, H / patch_size, W / patch_size]``.

        """
        x = self.patch_embed(x)
        batch, height, width, _ = x.shape
        x = x.flatten(1, 2)
        # ``0 * mask_token`` is the reference's unmasked branch; it keeps an inf or
        # NaN mask token visible rather than silently skipping it.
        cls_token = self.cls_token + 0 * self.mask_token
        x = torch.cat([cls_token.expand(batch, -1, -1), x], dim=1)
        rope = self.rope_embed(height=height, width=width)
        for block in self.blocks:
            x = block(x, rope)
        x_norm_patch = self.norm(x)[:, 1:]
        num_patches = x_norm_patch.shape[1]
        patch_tokens = self.feature_bottleneck(x_norm_patch.reshape(-1, self.embed_dim))
        patch_tokens = patch_tokens.reshape(batch, num_patches, -1)
        return patch_tokens.transpose(1, 2).reshape(batch, -1, height, width)


class DinoV3PixelDecoder(nn.Module):
    """VTP's pixel decoder: latents to ImageNet-normalized pixels.

    A 1x1 convolution widens the latent, LayerNorm SwiGLU blocks with rotary
    attention mix it, and a 1x1 convolution plus pixel shuffle unfold each token
    into an ``upscale_factor``-square RGB patch.
    """

    class Config(Fig["DinoV3PixelDecoder"]):
        """Decoder size."""

        channels_in: int = -1
        """Latent channels; :class:`VTP` sets the trunk's ``channels_out``."""

        channels_hidden: int = 1024
        """Token width."""

        num_layers: int = 24
        """Transformer blocks."""

        heads: int = 16
        """Attention heads; ``channels_hidden`` must be divisible by ``4 * heads``."""

        expansion: float = 4.0
        """Nominal FFN width over ``channels_hidden``, before SwiGLU's 2/3 and alignment."""

        upscale_factor: int = -1
        """Pixels per latent side; :class:`VTP` sets the trunk's ``patch_size``."""

        def cost(
            self,
            *,
            latent_size: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one trainable forward and backward on ``[batch_size, channels_in, s, s]``.

            An ordinary module, counted in both phases like the trunk: the
            input 1x1 convolution, the blocks and final norm over the ``s ** 2``
            latent tokens, and the output 1x1 convolution. The pixel shuffle
            and the token transposes are layout and move no arithmetic.

            Args:
              latent_size: Side ``s`` of the square latent grid, in tokens.
              batch_size: Latents per invocation.
              dtype: Activation dtype; ``None`` is torch's default. The rotary
                embedding runs in bfloat16 whatever it is.
              **kwargs: The open bus, unread.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            Raises:
              ValueError: ``channels_in`` or ``upscale_factor`` is unset, which
                :class:`VTP` resolves from its trunk.

            """
            del kwargs
            if -1 in (self.channels_in, self.upscale_factor):
                raise ValueError(
                    "channels_in and upscale_factor must be set; VTP.Config "
                    "propagates them from its trunk.",
                )
            width = self.channels_hidden
            grid = (latent_size, latent_size)
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
                norm=priml_norm.LayerNorm.Config(
                    width,
                    eps=1e-6,
                    elementwise_affine=True,
                ),
                channels=width,
                heads=self.heads,
                expansion=self.expansion,
                num_layers=self.num_layers,
                side=latent_size,
                prefix=0,
                batch_size=batch_size,
                dtype=dtype,
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
        dim = config.channels_hidden
        self.embed_dim = dim
        self.proj_in = nn.Conv2d(config.channels_in, dim, kernel_size=1, bias=True)
        self.rope_embed = RopePositionEmbedding(dim, num_heads=config.heads)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(
                dim,
                num_heads=config.heads,
                expansion=config.expansion,
                norm=layer_norm,
            )
            for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.upscale_factor = config.upscale_factor
        self.proj_out = nn.Conv2d(
            dim,
            3 * self.upscale_factor**2,
            kernel_size=1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        for conv in (self.proj_in, self.proj_out):
            nn.init.trunc_normal_(conv.weight, std=0.02)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for module in self.modules():
            init_weights_vit(module)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Decode latents.

        Args:
          x: ``[B, channels_in, h, w]`` raw latent.

        Returns:
          pixels: ``[B, 3, h * upscale_factor, w * upscale_factor]``,
            ImageNet-normalized.

        """
        batch, _, height, width = x.shape
        x = self.proj_in(x)
        x = x.flatten(2).transpose(1, 2)
        rope = self.rope_embed(height=height, width=width)
        for block in self.blocks:
            assert isinstance(block, SelfAttentionBlock)
            x = block(x, rope)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(batch, self.embed_dim, height, width)
        x = self.proj_out(x)
        return self.pixel_shuffle(x)


class VTP(nn.Module):
    """VTP behind the ``Autoencoder`` contract.

    ``encode`` takes uint8 RGB, normalizes it with the ImageNet statistics the
    checkpoints were trained on, and returns the trunk's raw bottleneck latent;
    ``decode`` inverts the pixel normalization and clamps. Deterministic: there
    is no posterior. Frozen: the wrapper stays in evaluation mode whatever its
    parent does.
    """

    class Config(Fig["VTP"]):
        """Trunk and decoder sizes, pixel statistics, precision, weights, and latent norm."""

        trunk: DinoVisionTransformerWithBottleneck.Config = field(
            default_factory=DinoVisionTransformerWithBottleneck.Config,
        )
        """Vision encoder; its ``channels_out`` and ``patch_size`` shape the latent."""

        pixel_decoder: DinoV3PixelDecoder.Config = field(
            default_factory=DinoV3PixelDecoder.Config,
        )
        """Decoder; its input width and upscale follow the trunk."""

        image_size: int = 256
        """Side of the square image the checkpoint was trained at."""

        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """Per-channel RGB mean subtracted before encoding."""

        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """Per-channel RGB standard deviation divided out before encoding."""

        dtype_autocast: torch.dtype | None = None
        """Autocast dtype for the encoder alone; ``None`` encodes in the weights' dtype.

        The reference's evaluation autocasts the encoder to its precision flag
        and decodes in float32. Set, ``encode`` returns latents in this dtype;
        ``decode`` never autocasts, so it wants them cast back to float32.
        """

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=lambda: HubFile.Config(
                repo_id="MiniMaxAI/VTP-Large-f16d64",
                filename="model.safetensors",
                revision="d1726ca9abd5ed0b3ad938c508c635d8b1a51a31",
                sha256="de5df2006083a9536c4d3ea36c6ae2181ec604e06550f1b2d7cece3b16aac32f",
            ),
        )
        """Published weights; ``None`` keeps the random initialization."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=lambda: ChannelLatentStats.Config(
                stats=UrlFile.Config(
                    url=f"{_LATENT_STATS_ROOT}/vtp_l/latents_stats.pt",
                    sha256="0ad1fadbfb8959f3a93147a4db668b6d704ae36ff320bfad53cd3d568de7b0fd",
                ),
            ),
        )
        """Per-channel standardization by the statistics published with ``checkpoint``.

        The autoencoder itself never applies it.
        """

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.pixel_decoder,
                "channels_in",
                self.trunk.channels_out,
                protocol=ChannelsIn,
            )
            propagate_attr(self.pixel_decoder, "upscale_factor", self.trunk.patch_size)
            if self.image_size % self.trunk.patch_size:
                raise ValueError(
                    f"image_size {self.image_size} must be divisible by "
                    f"patch_size {self.trunk.patch_size}.",
                )
            return super().finalize()

        def latent_shape(self) -> tuple[int, int, int]:
            """Return ``(trunk.channels_out, side, side)`` at ``image_size``."""
            side = self.image_size // self.trunk.patch_size
            return self.trunk.channels_out, side, side

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one reconstruction ``decode(encode(image))`` at ``image_size``.

            Frozen, so only forward work is kept -- the trunk and decoder costs
            restricted to their primal cells -- while ``params`` still counts
            every weight, as :class:`priml.model.dinov2.DinoV2Teacher` does. The
            pixel maps around them run in float32 whatever ``dtype`` is:
            ``float()``, ``/ 255``, ``- mean``, ``/ std`` in; ``- inverse_mean``,
            ``/ inverse_std``, and the two-sided clamp out.

            Args:
              batch_size: Images per invocation.
              dtype: The weights' dtype, which the decoder runs in; the trunk
                runs in ``dtype_autocast`` when set. ``None`` is torch's default.
              **kwargs: The open bus, unread: ``image_size`` is this config's.

            Returns:
              cost: Primal integer FLOPs and logical bytes, owning every
                parameter of the trunk and the decoder.

            """
            del kwargs
            full = self.trunk.cost(
                image_size=self.image_size,
                batch_size=batch_size,
                dtype=dtype if self.dtype_autocast is None else self.dtype_autocast,
            ) + self.pixel_decoder.cost(
                latent_size=self.image_size // self.trunk.patch_size,
                batch_size=batch_size,
                dtype=dtype,
            )
            pixels = batch_size * 3 * self.image_size**2
            channels = len(self.pixel_mean)
            # Each map reads the image and writes it once; the two subtracted or
            # divided maps also read one statistic per channel.
            pixel_maps = (
                traffic("primal", "elementwise", elements=pixels, dtype=torch.uint8)
                + traffic(
                    "primal",
                    "elementwise",
                    elements=7 * pixels + 2 * channels,
                    flops=3 * pixels,
                    dtype=torch.float32,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=6 * pixels + 2 * channels,
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
        # The reference registers the trunk, then the pixel decoder.
        self.trunk = config.trunk.make()
        self.pixel_decoder = config.pixel_decoder.make()
        self.patch_size = config.trunk.patch_size
        self.dtype_autocast = config.dtype_autocast
        # torchvision's Normalize and the reference's inverse Normalize, whose
        # statistics round to float32 as these do: ``(x - (-m / s)) / (1 / s)`` is
        # not ``x * s + m`` in the last bit.
        mean = torch.tensor(config.pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(config.pixel_std, dtype=torch.float32).view(-1, 1, 1)
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
        # ``self.modules()`` is pre-order and HF's pass post-order, but only leaves
        # draw, and both reach the leaves in the same order.
        for module in self.modules():
            init_weights_post(module)
        if config.checkpoint is not None:
            state = _autoencoder_state(load_file(config.checkpoint.make().path()))
            self.load_state_dict(state)
        self.eval()
        self.requires_grad_(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, image: Tensor, /) -> Tensor:
        """Return the bottleneck latent of a uint8 image batch.

        Args:
          image: ``[B, 3, H, W]`` uint8 RGB, ``H`` and ``W`` multiples of
            ``patch_size``.

        Returns:
          latent: ``[B, channels_out, H / patch_size, W / patch_size]`` raw
            latent, in ``dtype_autocast`` when set.

        Raises:
          TypeError: ``image`` is not uint8.
          ValueError: A side is not a multiple of ``patch_size``.

        """
        require_uint8(image)
        height, width = image.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Image sides {height}x{width} must be multiples of "
                f"patch_size {self.patch_size}.",
            )
        # Divide then normalize, as torchvision's ToTensor then Normalize do.
        unit = rgb2float(image, float_dtype=torch.float32, unit_interval=True)
        pixels = (unit - self.pixel_mean) / self.pixel_std
        autocast = (
            nullcontext()
            if self.dtype_autocast is None
            else torch.autocast(image.device.type, dtype=self.dtype_autocast)
        )
        with autocast:
            return self.trunk(pixels)

    def decode(self, latent: Tensor, /) -> Tensor:
        """Decode a raw latent to RGB in ``[0, 1]``.

        Args:
          latent: ``[B, channels_out, h, w]`` raw latent, in the weights' dtype.

        Returns:
          image: ``[B, 3, h * patch_size, w * patch_size]`` float RGB, clamped.

        """
        decoded = self.pixel_decoder(latent)
        return ((decoded - self.inverse_mean) / self.inverse_std).clamp(0, 1)


def vtp_small() -> VTP.Config:
    """Return VTP-Small-f16d64: 384 wide, 12 blocks, 6 heads in trunk and decoder."""
    config = VTP.Config()
    config.trunk.channels_hidden = 384
    config.trunk.num_layers = 12
    config.trunk.heads = 6
    config.pixel_decoder.channels_hidden = 384
    config.pixel_decoder.num_layers = 12
    config.pixel_decoder.heads = 6
    config.checkpoint = HubFile.Config(
        repo_id="MiniMaxAI/VTP-Small-f16d64",
        filename="model.safetensors",
        revision="378967941e66f7f09a9e10218aa8710a930cc635",
        sha256="5442083e078aadc65e51c7893d10177e360139866d4e09ab955b1c7b563eeaee",
    )
    config.latent_norm = ChannelLatentStats.Config(
        stats=UrlFile.Config(
            url=f"{_LATENT_STATS_ROOT}/vtp_s/latents_stats.pt",
            sha256="9f4bb851a6226843c0bdb9b25330faa2a8428e5c10a77f79a974f145cf848eff",
        ),
    )
    return config


def vtp_base() -> VTP.Config:
    """Return VTP-Base-f16d64: 768 wide, 12 blocks, 12 heads in trunk and decoder."""
    config = VTP.Config()
    config.trunk.channels_hidden = 768
    config.trunk.num_layers = 12
    config.trunk.heads = 12
    config.pixel_decoder.channels_hidden = 768
    config.pixel_decoder.num_layers = 12
    config.pixel_decoder.heads = 12
    config.checkpoint = HubFile.Config(
        repo_id="MiniMaxAI/VTP-Base-f16d64",
        filename="model.safetensors",
        revision="21fe3a9cb53f01a2c7f3b622efdc8d2534a7c79d",
        sha256="c7fdbb1507cecbcbb35eab4ea9fcbca8dc80b5e0d64cfe0c37b7baacdbb0fa05",
    )
    config.latent_norm = ChannelLatentStats.Config(
        stats=UrlFile.Config(
            url=f"{_LATENT_STATS_ROOT}/vtp_b/latents_stats.pt",
            sha256="f7c030310f3eeaa38d2d0838f38eeab1d9922fcaf183a0fe8dd1f1287366c4a7",
        ),
    )
    return config


def vtp_large() -> VTP.Config:
    """Return VTP-Large-f16d64, the :class:`VTP.Config` defaults: 1024 wide, 24 blocks."""
    return VTP.Config()


_LATENT_STATS_ROOT: Final = (
    "https://raw.githubusercontent.com/MiniMax-AI/VTP/"
    "5ce1eb67010fff3c1eed483352483be6a1838556/generation/latent_stats"
)

# The published checkpoints are the whole VTP model; everything outside the trunk and
# pixel decoder is its CLIP text tower and projections, which reconstruction never
# reads. Named exactly, so a checkpoint carrying anything else is refused rather than
# half loaded.
_TEXT_TOWER_PREFIX: Final = "text_transformer."
_TEXT_TOWER_KEYS: Final = frozenset(
    {
        "token_embedding.weight",
        "positional_embedding",
        "ln_final.weight",
        "ln_final.bias",
        "text_projection",
        "visual_proj.weight",
        "logit_scale",
    },
)


def _autoencoder_state(checkpoint: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Keep the ``trunk.*`` and ``pixel_decoder.*`` entries; discard the text tower."""
    kept: dict[str, Tensor] = {}
    unexpected: list[str] = []
    for key, value in checkpoint.items():
        if key.startswith(("trunk.", "pixel_decoder.")):
            kept[key] = value
        elif not key.startswith(_TEXT_TOWER_PREFIX) and key not in _TEXT_TOWER_KEYS:
            unexpected.append(key)
    if unexpected:
        raise ValueError(
            f"Checkpoint holds keys neither the autoencoder nor the text tower "
            f"explains: {sorted(unexpected)}.",
        )
    return kept


def _swiglu_channels_hidden(hidden_features: int) -> int:
    """Return SwiGLU's hidden width: ``2/3`` of nominal, aligned up to 8."""
    return ceil_multiple(int(hidden_features * 2 / 3), 8)


def _transformer_cost(
    *,
    norm: priml_norm.RMSNorm.Config | priml_norm.LayerNorm.Config,
    channels: int,
    heads: int,
    expansion: float,
    num_layers: int,
    side: int,
    prefix: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost the rotary table, ``num_layers`` blocks, and the final norm.

    The sequence is ``prefix`` unrotated tokens then the ``side ** 2`` grid.
    Each block is two norms, the biased ``qkv`` and ``proj`` projections, the
    rotation, the two attention products, the biased split-gate SwiGLU, and two
    residual adds; the table is built once and shared by every block.
    """
    patches = side * side
    tokens = patches + prefix
    rows = batch_size * tokens
    channels_head = channels // heads
    ffn = SwiGLU.Config(
        channels,
        channels,
        channels_hidden=_swiglu_channels_hidden(int(channels * expansion)),
        bias=True,
        split_gate_projection=True,
    )
    block = (
        norm.cost(seq_len=tokens, batch_size=batch_size, dtype=dtype).tile(
            2,
            copies=2,
        )
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
            dtype=dtype,
        )
    )
    return (
        _rope_table_cost(side=side, channels_head=channels_head)
        + block.tile(num_layers, copies=num_layers)
        + norm.cost(seq_len=tokens, batch_size=batch_size, dtype=dtype)
    )


# ``x * cos + rotate_half(x) * sin`` is two products and an add per channel, and
# ``rotate_half`` negates half the channels: 7/2 operations each way. The rotation
# runs in ``RopePositionEmbedding``'s dtype, bfloat16 as both modules build it, so
# queries and keys of every token are cast there and back unless already bfloat16.
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
            traffic(phase, "elementwise", elements=2 * rows * width, dtype=cast)
            for phase in ("primal", "adjoint")
            for cast in (dt, torch.bfloat16)
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
        "elementwise",
        elements=coordinates + angles // 2 + 4 * angles,
        flops=2 * side + 6 * positions + angles // 2 + 2 * angles,
        dtype=torch.bfloat16,
    )
