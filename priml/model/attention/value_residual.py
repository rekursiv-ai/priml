"""Self attention that mixes each layer's values with a reference value stream."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.attention.rope import RoPE, rotation_cost
from priml.model.norm import RMSNorm


def blend_values(v: Tensor, first: Tensor, weight: Tensor) -> Tensor:
    """Mix current and first-layer values in the reference's operation order."""
    return weight * first + (1.0 - weight) * v


class ValueResidualAttention(nn.Module):
    """QK normalized attention with 2D RoPE and a first-block value reference."""

    class Config(Fig["ValueResidualAttention"]):
        channels: int = -1
        """Input and output token width."""

        heads: int = -1
        """Attention heads."""

        qk_norm: bool = True
        """Apply RMSNorm to query and key heads."""

        value_residual: bool = True
        """Mix values with a supplied reference value stream."""

        reference_rope: bool = False
        """Use REG's unfused rotary multiplication order for backward parity."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost projections, QK normalization, rotary, SDPA, and value blend."""
            del kwargs
            head_dim = self.channels // self.heads
            rows = seq_len * batch_size
            total = (
                matmul_cost(
                    channels_in=self.channels,
                    channels_out=3 * self.channels,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.channels,
                    channels_out=self.channels,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + attention_kernel_cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    num_heads=self.heads,
                    channels_head=head_dim,
                )
                + rotation_cost(
                    RoPE.Config(channels_head=(head_dim // 2, head_dim // 2)),
                    rows=rows,
                    dtype=dtype,
                    channels_head=head_dim,
                    heads=2 * self.heads,
                )
            )
            if self.qk_norm:
                total += cost(
                    RMSNorm.Config(channels_in=head_dim, elementwise_affine=True),
                    seq_len=seq_len,
                    batch_size=batch_size * self.heads,
                    dtype=dtype,
                ).tile(2, copies=2)
            if self.value_residual:
                elements = rows * self.channels
                total += elementwise_cost(
                    primal=3 * elements,
                    adjoint=4 * elements,
                    channels=head_dim,
                    rows=rows * self.heads,
                    params=1,
                    inputs=2,
                    dtype=dtype,
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels % config.heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = config.heads
        self.head_dim = config.channels // config.heads
        self.reference_rope = config.reference_rope
        self.qkv = nn.Linear(config.channels, 3 * config.channels)
        self.q_norm = nn.RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.proj = nn.Linear(config.channels, config.channels)
        self.v1_lambda = (
            nn.Parameter(torch.tensor(0.5)) if config.value_residual else None
        )

    def forward(
        self,
        x: Tensor,
        rope_factors: tuple[Tensor, Tensor],
        v1: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Attend to image tokens and return the raw value stream for reuse."""
        batch, tokens, channels = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        raw_v = v
        if v1 is not None and self.v1_lambda is not None:
            v = blend_values(v, v1, self.v1_lambda)
        cos, sin = rope_factors
        q, k = self.q_norm(q), self.k_norm(k)
        if self.reference_rope:
            cos = cos.transpose(1, 2).repeat_interleave(2, dim=-1).to(q.dtype)
            sin = sin.transpose(1, 2).repeat_interleave(2, dim=-1).to(q.dtype)

            def rotate_half(heads: Tensor) -> Tensor:
                return torch.stack(
                    (-heads[..., 1::2], heads[..., 0::2]), dim=-1
                ).reshape_as(heads)

            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        else:
            q, k = RoPE.rotate(
                q.transpose(1, 2),
                k.transpose(1, 2),
                cos,
                sin,
                interleave=True,
            )
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        output = functional.scaled_dot_product_attention(q, k, v)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, channels)), raw_v
