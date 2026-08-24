"""GatedDeltaNet: linear attention via the gated delta rule.

A linear attention mechanism that uses a gated delta rule for
state updates, supporting both CUDA (via ``fla``) and a pure-torch
CPU fallback. Plugs into ``TransformerBlock.Config(attn=...)`` as a
``Makeable[nn.Module]``.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Any, Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional as f

import torch

from priml.math.basic import ceil_multiple
from priml.model.custom_types import ChannelsIn, TensorModule
from priml.model.linear import Linear
from priml.model.norm import CenteredRMSNorm


if TYPE_CHECKING:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
else:
    from wrapt import lazy_import

    chunk_gated_delta_rule = lazy_import(
        "fla.ops.gated_delta_rule", "chunk_gated_delta_rule"
    )


class GatedDeltaNet(nn.Module):
    """Linear attention via gated delta rule."""

    class Config(Fig["GatedDeltaNet"], kw_only=False):
        channels_in: int = -1
        """Model width read from the residual stream."""

        _: KW_ONLY

        heads_k: int = 16
        """Key/query heads; ``heads_v`` must be a multiple of this."""

        heads_v: int = 32
        """Value heads. Exceeding ``heads_k`` repeats q/k to match."""

        channels_k_head: int = 128
        """Per-head key/query width."""

        channels_v_head: int = 128
        """Per-head value width."""

        conv_kernel_size: int = 4
        """Depthwise causal convolution width over the q/k/v stream."""

        norm: Makeable[TensorModule] = field(default_factory=CenteredRMSNorm.Config)
        """Normalization applied per value head before the output projection."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.norm, ChannelsIn) and self.norm.channels_in == -1:
                self.norm.channels_in = self.channels_v_head
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Every count, not just heads_k: a zero elsewhere builds a zero-width
        # Linear or a negative Conv padding and fails inside torch, naming a
        # tensor shape rather than the field that produced it. heads_k comes
        # first because the modulus below divides by it.
        for name, value in (
            ("heads_k", config.heads_k),
            ("heads_v", config.heads_v),
            ("channels_in", config.channels_in),
            ("channels_k_head", config.channels_k_head),
            ("channels_v_head", config.channels_v_head),
            ("conv_kernel_size", config.conv_kernel_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}.")
        if config.heads_v % config.heads_k != 0:
            raise ValueError(
                f"heads_v={config.heads_v} must be an integer "
                f"multiple of heads_k={config.heads_k}.",
            )
        h = config.channels_in
        self.heads_k = config.heads_k
        self.heads_v = config.heads_v
        self.channels_k_head = config.channels_k_head
        self.channels_v_head = config.channels_v_head
        k_dim = config.heads_k * config.channels_k_head
        v_dim = config.heads_v * config.channels_v_head
        conv_dim = 2 * k_dim + v_dim

        self.in_proj_qkv = Linear.Config(
            channels_in=h,
            channels_out=conv_dim,
            bias=False,
        ).make()
        self.in_proj_z = Linear.Config(
            channels_in=h,
            channels_out=v_dim,
            bias=False,
        ).make()
        self.in_proj_b = Linear.Config(
            channels_in=h,
            channels_out=config.heads_v,
            bias=False,
        ).make()
        self.in_proj_a = Linear.Config(
            channels_in=h,
            channels_out=config.heads_v,
            bias=False,
        ).make()

        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            config.conv_kernel_size,
            bias=False,
            groups=conv_dim,
            padding=config.conv_kernel_size - 1,
        )
        # Raw params allocated empty; reset_parameters is the sole source of
        # their init values, so eager and meta materialization agree bit-for-bit.
        self.dt_bias = nn.Parameter(torch.empty(config.heads_v))
        self.A_log = nn.Parameter(torch.empty(config.heads_v))
        self.norm = config.norm.make()
        # ``depth`` scales the projection that writes back into the residual
        # stream, which is the one deep-network init schemes shrink; the input
        # projections are left unscaled, matching SwiGLU's down_proj.
        self.out_proj = Linear.Config(
            channels_in=v_dim,
            channels_out=h,
            bias=False,
            depth=config.depth,
        ).make()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # This module made every child below, so it owns re-initializing them
        # (and its own raw params). dt_bias and A_log carry deliberate
        # Mamba-style inits that meta materialization must reproduce.
        self.in_proj_qkv.reset_parameters()
        self.in_proj_z.reset_parameters()
        self.in_proj_b.reset_parameters()
        self.in_proj_a.reset_parameters()
        self.conv1d.reset_parameters()
        self.norm.reset_parameters()
        self.out_proj.reset_parameters()
        with torch.no_grad():
            nn.init.ones_(self.dt_bias)
            # Clamped off zero rather than sampled from a shifted range:
            # uniform_(0, 16) really does return exactly 0.0 (measured, once
            # in 10M draws), and log(0) is -inf, which the forward
            # exponentiates into a permanently closed gate. Clamping keeps the
            # distribution the reference specifies; shifting to [1, 17) would
            # move its mean from 1.775 to 2.007.
            draw = torch.empty_like(self.A_log).uniform_(0, 16)
            self.A_log.copy_(draw.clamp_(min=torch.finfo(draw.dtype).tiny).log_())

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        shape = x.shape
        S = shape[-2]
        x = x.reshape(-1, S, shape[-1])
        k_dim = self.heads_k * self.channels_k_head
        v_dim = self.heads_v * self.channels_v_head

        qkv = self.in_proj_qkv(x)
        qkv = f.silu(self.conv1d(qkv.transpose(1, 2))[:, :, :S]).transpose(1, 2)
        q, k, v = qkv.split([k_dim, k_dim, v_dim], dim=-1)

        z = self.in_proj_z(x).reshape(-1, S, self.heads_v, self.channels_v_head)
        beta = self.in_proj_b(x).sigmoid()
        a = self.in_proj_a(x)
        g = -self.A_log.float().exp() * f.softplus(a.float() + self.dt_bias)

        q = q.reshape(-1, S, self.heads_k, self.channels_k_head)
        k = k.reshape(-1, S, self.heads_k, self.channels_k_head)
        v = v.reshape(-1, S, self.heads_v, self.channels_v_head)

        if self.heads_v // self.heads_k > 1:
            r = self.heads_v // self.heads_k
            q = q.repeat_interleave(r, dim=-2)
            k = k.repeat_interleave(r, dim=-2)

        out: Tensor
        if x.is_cuda:
            out, _ = chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            out, _ = _torch_chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        out = self.norm(out.reshape(-1, self.channels_v_head)) * f.silu(
            z.reshape(-1, self.channels_v_head).float()
        ).type_as(out)
        return self.out_proj(out.reshape(-1, S, v_dim)).reshape(*shape[:-1], -1)


def _l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Pure-torch chunk_gated_delta_rule (from HF transformers, MIT-licensed)."""
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    B, H, S, dk = key.shape
    dv = value.shape[-1]
    pad = int(ceil_multiple(S, chunk_size)) - S
    query = f.pad(query, (0, 0, 0, pad))
    key = f.pad(key, (0, 0, 0, pad))
    value = f.pad(value, (0, 0, 0, pad))
    beta = f.pad(beta, (0, pad))
    g = f.pad(g, (0, pad))
    S_total = S + pad
    scale = 1.0 / math.sqrt(dk)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, H, -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device)
    )
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    attn = -(k_beta @ key.transpose(-1, -2) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = (
        torch.zeros(B, H, dk, dv, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    out = torch.zeros_like(value)
    for ci in range(S_total // chunk_size):
        q_i, k_i, v_i = query[:, :, ci], key[:, :, ci], value[:, :, ci]
        a = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]
        v_prime = k_cumdecay[:, :, ci] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, ci, :, None].exp()) @ state
        out[:, :, ci] = attn_inter + a @ v_new
        state = (
            state * g[:, :, ci, -1, None, None].exp()
            + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )
    if not output_final_state:
        state = None
    out = out.reshape(B, H, -1, out.shape[-1])[:, :, :S]
    return out.transpose(1, 2).contiguous().to(dtype), state
