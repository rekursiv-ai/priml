"""REG + SPRINT SiT used by SpeedrunDiT.

The small modules expose the architectural decisions independently: axial RoPE,
value residual attention, adaptive conditioning, sparse routing, and layerwise
MLP width. The default configuration is the published SiT-B/1 run.
"""

# PyTorch's module buffer and parameter stubs expose several values as Any.
# pyright: reportAny=false

from __future__ import annotations

from typing import NamedTuple, override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.math.diffusion.conditioning import modulate
from priml.math.position_embedding import image_token_positions, sincos_position_table
from priml.model.attention.rope import RoPE
from priml.model.attention.value_residual import ValueResidualAttention
from priml.model.conditioning import LabelEmbedder, TimestepEmbedder
from priml.model.norm import RMSNorm
from priml.model.token_routing import SparseDenseFusion, select_tokens


class SiTBlock(nn.Module):
    """RMSNorm + adaLN-Zero block with GELU feedforward."""

    def __init__(
        self,
        channels: int,
        heads: int,
        mlp_ratio: float,
        *,
        qk_norm: bool,
        value_residual: bool,
        reference_rope: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        self.attn = ValueResidualAttention.Config(
            channels=channels,
            heads=heads,
            qk_norm=qk_norm,
            value_residual=value_residual,
            reference_rope=reference_rope,
        ).make()
        self.norm2 = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 6 * channels),
        )

    @override
    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        rope_factors: tuple[Tensor, Tensor],
        v1: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply attention and feedforward updates under adaLN conditioning."""
        s1, a1, g1, s2, a2, g2 = self.adaLN_modulation(condition).chunk(6, dim=-1)
        attention, raw_v = self.attn(modulate(self.norm1(x), s1, a1), rope_factors, v1)
        x = x + g1[:, None] * attention
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))
        return x, raw_v


class FinalLayer(nn.Module):
    """Project the CLS and image tokens into their velocity targets."""

    def __init__(
        self,
        channels: int,
        patch_size: int,
        out_channels: int,
        cls_channels: int,
    ) -> None:
        super().__init__()
        self.norm_final = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(channels, patch_size**2 * out_channels)
        self.linear_cls = nn.Linear(channels, cls_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )

    @override
    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Return patch and CLS velocities."""
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])


class Projection(NamedTuple):
    """Student tokens and optional indices retained by SPRINT."""

    tokens: Tensor
    ids_keep: Tensor | None


class ModelOutput(NamedTuple):
    """Latent velocity, CLS velocity, and REG alignment projections."""

    velocity: Tensor
    cls_velocity: Tensor
    projections: tuple[Projection, ...]


class SpeedrunDiT(nn.Module):
    """Class conditional latent SiT with REG and SPRINT."""

    class Config(Fig["SpeedrunDiT"]):
        input_size: int = 16
        """Spatial side of the INVAE latent grid."""
        in_channels: int = 32
        """INVAE latent channels."""
        patch_size: int = 1
        """Latent cells in each patch side."""
        hidden_size: int = 768
        """Transformer token width."""
        depth: int = 12
        """Total number of transformer blocks."""
        num_heads: int = 12
        """Attention heads per block."""
        num_classes: int = 1000
        """ImageNet classes, excluding the classifier-free token."""
        cls_channels: int = 768
        """DINO CLS feature width."""
        projector_hidden: int = 2048
        """Width of the REG projection MLP."""
        projection_depths: tuple[int, ...] = (2, 4, 6)
        """Blocks whose tokens are aligned to DINO features."""
        mlp_ratio_min: float = 2.0
        """Feedforward expansion in the first block."""
        mlp_ratio_max: float = 6.0
        """Feedforward expansion in the last block."""
        drop_ratio: float = 0.75
        """Fraction of patch tokens removed from the sparse middle blocks."""
        path_drop_prob: float = 0.05
        """Chance of removing the sparse branch during training."""
        class_dropout_prob: float = 0.1
        """Chance of using the classifier-free class embedding."""
        qk_norm: bool = True
        """Normalize query and key heads with RMSNorm."""
        position_compute_dtype: torch.dtype = torch.float64
        """Intermediate precision for fixed sin/cos positions; REG used float64."""
        reference_rope: bool = True
        """Use the source rotary arithmetic for backward parity."""
        encoder_blocks: int = 2
        """Dense blocks before SPRINT token selection."""
        decoder_blocks: int = 2
        """Dense blocks after sparse and dense streams are fused."""

        def cost(
            self,
            *,
            batch_size: int = 1,
            dtype: torch.dtype | None = None,
            **kwargs: object,
        ) -> Cost:
            """Estimate a training forward, pricing SPRINT blocks at routed length."""
            del kwargs
            width = self.hidden_size
            grid = self.input_size // self.patch_size
            dense = grid * grid + 1
            sparse = max(1, int(dense * (1 - self.drop_ratio)))
            middle_end = self.depth - self.decoder_blocks

            def linear(channels_in: int, channels_out: int, rows: int) -> Cost:
                return matmul_cost(
                    channels_in=channels_in,
                    channels_out=channels_out,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )

            total = (
                linear(
                    self.in_channels * self.patch_size**2,
                    width,
                    batch_size * (dense - 1),
                )
                + linear(self.cls_channels, width, batch_size)
                + cost(
                    RMSNorm.Config(channels_in=width, elementwise_affine=True),
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    TimestepEmbedder.Config(channels_out=width),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    LabelEmbedder.Config(
                        channels_in=self.num_classes,
                        channels_out=width,
                        dropout=self.class_dropout_prob,
                    ),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    RoPE.Config(channels_head=(width // self.num_heads // 2,) * 2),
                    seq_len=dense,
                    batch_size=1,
                    dtype=dtype,
                )
            )
            projected_once = False
            for index in range(self.depth):
                length = sparse if self.encoder_blocks <= index < middle_end else dense
                rows = batch_size * length
                ratio = self.mlp_ratio_min + (
                    (self.mlp_ratio_max - self.mlp_ratio_min)
                    * index
                    / max(1, self.depth - 1)
                )
                hidden = int(width * ratio)
                total += (
                    cost(
                        ValueResidualAttention.Config(
                            channels=width,
                            heads=self.num_heads,
                            qk_norm=self.qk_norm,
                            value_residual=index > 0,
                        ),
                        seq_len=length,
                        batch_size=batch_size,
                        dtype=dtype,
                    )
                    + cost(
                        RMSNorm.Config(channels_in=width),
                        seq_len=length,
                        batch_size=2 * batch_size,
                        dtype=dtype,
                    )
                    + linear(width, hidden, rows)
                    + linear(hidden, width, rows)
                    + linear(width, 6 * width, batch_size)
                    + elementwise_cost(
                        primal=18 * rows * width + 5 * batch_size * width,
                        adjoint=18 * rows * width + 5 * batch_size * width,
                        channels=width,
                        rows=rows,
                        dtype=dtype,
                    )
                )
                if index + 1 in self.projection_depths:
                    projection = (
                        linear(width, self.projector_hidden, rows)
                        + linear(self.projector_hidden, self.projector_hidden, rows)
                        + linear(self.projector_hidden, self.cls_channels, rows)
                    )
                    total += projection.tile(1, copies=int(not projected_once))
                    projected_once = True
            total += cost(
                SparseDenseFusion.Config(channels=width),
                seq_len=dense,
                batch_size=batch_size,
                dtype=dtype,
            )
            return (
                total
                + cost(
                    RMSNorm.Config(channels_in=width),
                    seq_len=dense,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + linear(width, 2 * width, batch_size)
                + linear(
                    width,
                    self.patch_size**2 * self.in_channels,
                    batch_size * (dense - 1),
                )
                + linear(width, self.cls_channels, batch_size)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.depth < config.encoder_blocks + config.decoder_blocks:
            raise ValueError("depth shorter than SPRINT dense prefix and suffix")
        if tuple(sorted(set(config.projection_depths))) != config.projection_depths:
            raise ValueError("projection_depths must be strictly increasing")
        if any(d < 1 or d > config.depth for d in config.projection_depths):
            raise ValueError("projection depth outside the model")
        if config.input_size % config.patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.config = config
        self.grid_size = config.input_size // config.patch_size
        # Match the reference module registration order. Global gradient
        # clipping reduces gradients in that order, which affects low bits.
        self.fusion = SparseDenseFusion.Config(channels=config.hidden_size).make()
        self.x_embedder = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            config.patch_size,
        )
        self.t_embedder = TimestepEmbedder.Config(
            channels_out=config.hidden_size
        ).make()
        self.y_embedder = LabelEmbedder.Config(
            channels_in=config.num_classes,
            channels_out=config.hidden_size,
            dropout=config.class_dropout_prob,
        ).make()
        self.register_buffer(
            "pos_embed",
            sincos_position_table(
                config.hidden_size,
                self.grid_size,
                compute_dtype=config.position_compute_dtype,
            ).unsqueeze(0),
            persistent=True,
        )
        ratios = [
            config.mlp_ratio_min
            + (config.mlp_ratio_max - config.mlp_ratio_min)
            * i
            / max(1, config.depth - 1)
            for i in range(config.depth)
        ]
        self.blocks = nn.ModuleList(
            SiTBlock(
                config.hidden_size,
                config.num_heads,
                ratio,
                qk_norm=config.qk_norm,
                value_residual=i > 0,
                reference_rope=config.reference_rope,
            )
            for i, ratio in enumerate(ratios)
        )
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.projector_hidden),
            nn.SiLU(),
            nn.Linear(config.projector_hidden, config.projector_hidden),
            nn.SiLU(),
            nn.Linear(config.projector_hidden, config.cls_channels),
        )
        self.final_layer = FinalLayer(
            config.hidden_size,
            config.patch_size,
            config.in_channels,
            config.cls_channels,
        )
        self.cls_projector = nn.Linear(config.cls_channels, config.hidden_size)
        self.wg_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        axis_channels = config.hidden_size // config.num_heads // 2
        self.rope = RoPE.Config(channels_head=(axis_channels, axis_channels)).make()
        if config.reference_rope:
            factors = self.rope(
                image_token_positions(self.grid_size, torch.device("cpu"))
            )
            self.register_buffer("reference_rope_cos", factors[0], persistent=False)
            self.register_buffer("reference_rope_sin", factors[1], persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the reference SiT and adaLN-Zero initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.x_embedder.weight.flatten(1))
        assert self.x_embedder.bias is not None
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for module in (self.t_embedder.mlp[0], self.t_embedder.mlp[2]):
            nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            bias = block.adaLN_modulation[-1].bias
            assert bias is not None
            nn.init.zeros_(bias)
        for module in (
            self.final_layer.adaLN_modulation[-1],
            self.final_layer.linear,
            self.final_layer.linear_cls,
        ):
            nn.init.zeros_(module.weight)
            assert module.bias is not None
            nn.init.zeros_(module.bias)

    def _project(
        self,
        x: Tensor,
        layer: int,
        ids: Tensor | None,
        projections: list[Projection],
    ) -> None:
        if layer in self.config.projection_depths:
            projections.append(Projection(self.projector(x), ids))

    @override
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        cls_token: Tensor,
        *,
        force_drop_labels: Tensor | None = None,
        drop_sparse_path: bool = False,
        route_tokens: bool | None = None,
    ) -> ModelOutput:
        """Predict latent and CLS velocities plus intermediate DINO projections."""
        cfg = self.config
        batch, channels, height, width = x.shape
        if (channels, height, width) != (
            cfg.in_channels,
            cfg.input_size,
            cfg.input_size,
        ):
            raise ValueError("latent shape does not match model configuration")
        if cls_token.shape != (batch, cfg.cls_channels):
            raise ValueError("cls_token must have shape [batch, cls_channels]")
        spatial = self.x_embedder(x).flatten(2).transpose(1, 2)
        cls = self.wg_norm(self.cls_projector(cls_token))[:, None]
        x = torch.cat((cls, spatial), dim=1) + self.pos_embed
        rope_factors = (
            (self.reference_rope_cos, self.reference_rope_sin)
            if cfg.reference_rope
            else self.rope(image_token_positions(self.grid_size, x.device))
        )
        condition = self.t_embedder(t) + self.y_embedder(
            y,
            force_drop=force_drop_labels,
        )
        projections: list[Projection] = []
        first_v: Tensor | None = None
        for i in range(cfg.encoder_blocks):
            x, raw_v = self.blocks[i](x, condition, rope_factors, first_v)
            if first_v is None:
                first_v = raw_v
            self._project(x, i + 1, None, projections)
        dense = x
        if route_tokens is None:
            route_tokens = self.training
        sparse, kept = (
            select_tokens(dense, cfg.drop_ratio) if route_tokens else (dense, None)
        )
        if kept is None:
            sparse_rope_factors = rope_factors
        else:

            def select_factor(factor: Tensor) -> Tensor:
                return factor.expand(batch, -1, -1, -1).gather(
                    1,
                    kept[:, :, None, None].expand(-1, -1, 1, factor.shape[-1]),
                )

            sparse_rope_factors = (
                select_factor(rope_factors[0]),
                select_factor(rope_factors[1]),
            )
        sparse_v = (
            first_v.gather(
                2,
                kept[:, None, :, None].expand(
                    -1,
                    first_v.shape[1],
                    -1,
                    first_v.shape[3],
                ),
            )
            if kept is not None and first_v is not None
            else first_v
        )
        middle_end = cfg.depth - cfg.decoder_blocks
        for i in range(cfg.encoder_blocks, middle_end):
            sparse, _ = self.blocks[i](sparse, condition, sparse_rope_factors, sparse_v)
            self._project(sparse, i + 1, kept, projections)
        if self.training and cfg.path_drop_prob:
            coin = torch.rand((), device=x.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(coin, 0)
            drop_sparse_path = drop_sparse_path or bool(coin < cfg.path_drop_prob)
        x = self.fusion(dense, sparse, kept, drop_path=drop_sparse_path)
        for i in range(middle_end, cfg.depth):
            x, _ = self.blocks[i](x, condition, rope_factors, first_v)
            self._project(x, i + 1, None, projections)
        patches, cls_velocity = self.final_layer(x, condition)
        p = cfg.patch_size
        velocity = (
            patches.reshape(batch, self.grid_size, self.grid_size, p, p, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, height, width)
        )
        return ModelOutput(velocity, cls_velocity, tuple(projections))
