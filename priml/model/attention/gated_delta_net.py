"""GatedDeltaNet: linear attention via the gated delta rule.

A linear attention mechanism that uses a gated delta rule for
state updates, supporting both CUDA (via ``fla``) and a pure-torch
CPU fallback. Plugs into ``TransformerBlock.Config(attn=...)`` as a
``Makeable[nn.Module]``.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional as f

import torch

from priml.math.basic import ceil_multiple
from priml.model.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    shared_rows,
    with_rows,
)
from priml.model.custom_types import (
    ChannelsIn,
    DepthIndex,
    TensorModule,
    infer_same_width,
)
from priml.model.init import InitFn, call_init, kaiming_uniform
from priml.model.legacy_keys import absorb_legacy_keys
from priml.model.linear import Linear
from priml.model.norm import CenteredRMSNorm


if TYPE_CHECKING:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
else:
    from wrapt import lazy_import

    chunk_gated_delta_rule = lazy_import(
        "fla.ops.gated_delta_rule",
        "chunk_gated_delta_rule",
    )


def _init_decay(weight: Tensor) -> None:
    # Clamp exact zero draws before log without shifting the sampled range.
    with torch.no_grad():
        draw = torch.empty_like(weight).uniform_(0, 16)
        weight.copy_(draw.clamp_(min=torch.finfo(draw.dtype).tiny).log_())


class GatedDeltaNet(nn.Module):
    """Linear attention via gated delta rule."""

    class Config(Fig["GatedDeltaNet"], kw_only=False):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

        _: KW_ONLY

        num_heads_k: int = 16
        """Key/query heads; ``num_heads_v`` must be a multiple of this."""

        num_heads_v: int = 32
        """Value heads. Exceeding ``num_heads_k`` repeats q/k to match."""

        channels_k_head: int = 128
        """Per-head key/query width."""

        channels_v_head: int = 128
        """Per-head value width."""

        conv_kernel_size: int = 4
        """Depthwise causal convolution width over the q/k/v stream."""

        norm: Makeable[TensorModule] = field(default_factory=CenteredRMSNorm.Config)
        """Normalization applied per value head before the output projection."""

        init_weight: InitFn = kaiming_uniform
        """Initializer for input and output projections."""

        init_conv_weight: InitFn = kaiming_uniform
        """Initializer for the depthwise convolution."""

        init_decay: InitFn = _init_decay
        """Initializer for the log-space decay rates."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            if isinstance(self.norm, ChannelsIn) and self.norm.channels_in == -1:
                self.norm.channels_in = self.channels_v_head
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price the projections, the depthwise conv, and the recurrent scan.

            Matrix scan counts retain the recurrent-model proxy of two MACs
            per state element, not the chunked implementation's actual products.
            Scalar scan counts estimate a decay, delta and weighted state update
            with its analytical derivative. Chunking, triangular solves and
            padding change executed work and are not represented by this model.
            Nonlinearities and q/k normalization are included separately.
            Scalar scan I/O uses two binary-map passes over the state, values
            and two key rows, with a three-output VJP over that working set.
            This is the same recurrent proxy, not executed chunk-kernel traffic.
            ``bytes_state`` is zero: no cache grows with token count.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            rows = shared_rows(seq_len, batch_size, **kwargs)
            dt = dtype
            h = self.channels_in
            k_dim = self.num_heads_k * self.channels_k_head
            v_dim = self.num_heads_v * self.channels_v_head
            conv_dim = 2 * k_dim + v_dim
            projections = sum(
                (
                    matmul_cost(
                        channels_in=c_in,
                        channels_out=c_out,
                        rows=rows,
                        dtype=dt,
                    )
                    for c_in, c_out in (
                        (h, conv_dim),
                        (h, v_dim),
                        (h, self.num_heads_v),
                        (h, self.num_heads_v),
                        (v_dim, h),
                    )
                ),
                Cost(),
            )
            # Depthwise: each channel is its own ``[taps] -> [1]`` map, followed
            # by a SiLU.
            conv = matmul_cost(
                channels_in=self.conv_kernel_size,
                channels_out=1,
                bias=False,
                rows=rows,
                dtype=dt,
            ).tile(conv_dim, copies=conv_dim) + elementwise_cost(
                primal=5 * conv_dim,
                adjoint=5 * conv_dim,
                channels=conv_dim,
                rows=rows,
                dtype=dt,
            )
            # The recurrence: per value head, ``k^T v`` writes the state and
            # ``S q`` reads it -- two activation products of ``k_head x v_head``
            # per token. That is the recurrent-model proxy; the chunked kernel
            # executes a different (larger) product count. Logical state I/O is
            # counted even when the implementation keeps the state on chip.
            state_write = matmul_cost(
                channels_in=self.channels_k_head,
                channels_out=self.channels_v_head,
                weight=False,
                rows=1,
                dtype=dt,
            )
            state_read = matmul_cost(
                channels_in=self.channels_k_head,
                channels_out=self.channels_v_head,
                weight=False,
                rows=1,
                dtype=dt,
            )
            # Decay, delta, and weighted update over the state; the q/k L2 norms
            # each reduce their own row once in both primal and adjoint.
            state = self.num_heads_v * self.channels_k_head * self.channels_v_head
            norms = (
                reduction_cost(
                    input_elements=self.num_heads_v * self.channels_k_head,
                    output_groups=self.num_heads_v,
                    dtype=dt,
                )
                + reduction_cost(
                    input_elements=self.num_heads_v * self.channels_k_head,
                    output_groups=self.num_heads_v,
                    dtype=dt,
                    phase="adjoint",
                )
            ).tile(2, copies=2)
            scan = (
                (state_write + state_read).tile(
                    self.num_heads_v,
                    copies=self.num_heads_v,
                )
                + elementwise_cost(
                    primal=2 * state
                    + 2 * v_dim
                    + self.num_heads_v * (6 * self.channels_k_head + 4),
                    adjoint=4 * state
                    + 4 * v_dim
                    + self.num_heads_v * (10 * self.channels_k_head + 7),
                    channels=state
                    + v_dim
                    + 2 * self.num_heads_v * self.channels_k_head,
                    inputs=4,
                    outputs=2,
                    adjoint_inputs=7,
                    adjoint_outputs=3,
                    rows=rows,
                    dtype=dt,
                )
                + norms
            )
            # ``dt_bias`` and ``A_log``: one gate parameter per value head each.
            # ``A_log`` is exponentiated once per batch, so that is shared.
            gates = elementwise_cost(
                primal=(9 + 2 / rows) * self.num_heads_v,
                adjoint=9 * self.num_heads_v,
                channels=self.num_heads_v,
                inputs=4,
                outputs=2,
                adjoint_inputs=6,
                adjoint_outputs=3,
                params=2 * self.num_heads_v,
                dtype=dt,
                rows=rows,
            )
            # The norm runs once per value head; its parameters exist once.
            norm = cost(
                self.norm,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **with_rows(rows * self.num_heads_v, **kwargs),
            ).tile(
                self.num_heads_v,
            )
            return (
                projections
                + conv
                + scan
                + gates
                + norm
                + self._output_gate_cost(rows=rows, dtype=dt)
            )

        def _output_gate_cost(self, *, rows: float, dtype: torch.dtype | None) -> Cost:
            """Price the separate post-norm SiLU and product."""
            width = self.num_heads_v * self.channels_v_head
            return elementwise_cost(
                primal=6 * width,
                adjoint=7 * width,
                channels=width,
                inputs=3,
                outputs=2,
                adjoint_inputs=6,
                adjoint_outputs=3,
                rows=rows,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Every count, not just num_heads_k: a zero elsewhere builds a zero-width
        # Linear or a negative Conv padding and fails inside torch, naming a
        # tensor shape rather than the field that produced it. num_heads_k comes
        # first because the modulus below divides by it.
        for name, value in (
            ("num_heads_k", config.num_heads_k),
            ("num_heads_v", config.num_heads_v),
            ("channels_in", config.channels_in),
            ("channels_k_head", config.channels_k_head),
            ("channels_v_head", config.channels_v_head),
            ("conv_kernel_size", config.conv_kernel_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}.")
        if config.num_heads_v % config.num_heads_k != 0:
            raise ValueError(
                f"num_heads_v={config.num_heads_v} must be an integer "
                f"multiple of num_heads_k={config.num_heads_k}.",
            )
        self._init_conv_weight = config.init_conv_weight
        self._init_decay = config.init_decay
        h = config.channels_in
        self.num_heads_k = config.num_heads_k
        self.num_heads_v = config.num_heads_v
        self.channels_k_head = config.channels_k_head
        self.channels_v_head = config.channels_v_head
        k_dim = config.num_heads_k * config.channels_k_head
        v_dim = config.num_heads_v * config.channels_v_head
        conv_dim = 2 * k_dim + v_dim

        self.proj_qkv = Linear.Config(
            channels_in=h,
            channels_out=conv_dim,
            bias=False,
            init_weight=config.init_weight,
        ).make()
        self.proj_z = Linear.Config(
            channels_in=h,
            channels_out=v_dim,
            bias=False,
            init_weight=config.init_weight,
        ).make()
        self.proj_b = Linear.Config(
            channels_in=h,
            channels_out=config.num_heads_v,
            bias=False,
            init_weight=config.init_weight,
        ).make()
        self.proj_a = Linear.Config(
            channels_in=h,
            channels_out=config.num_heads_v,
            bias=False,
            init_weight=config.init_weight,
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
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads_v))
        self.A_log = nn.Parameter(torch.empty(config.num_heads_v))
        self.norm = config.norm.make()
        # ``depth`` scales the projection that writes back into the residual
        # stream, which is the one deep-network init schemes shrink; the input
        # projections are left unscaled, matching SwiGLU's down_proj.
        self.proj_out = Linear.Config(
            channels_in=v_dim,
            channels_out=h,
            bias=False,
            depth_index=config.depth_index,
            init_weight=config.init_weight,
        ).make()
        absorb_legacy_keys(
            self,
            {
                "in_proj_qkv": "proj_qkv",
                "in_proj_z": "proj_z",
                "in_proj_b": "proj_b",
                "in_proj_a": "proj_a",
                "out_proj": "proj_out",
            },
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        # This module made every child below, so it owns re-initializing them
        # (and its own raw params). dt_bias and A_log carry deliberate
        # Mamba-style inits that meta materialization must reproduce.
        self.proj_qkv.reset_parameters()
        self.proj_z.reset_parameters()
        self.proj_b.reset_parameters()
        self.proj_a.reset_parameters()
        call_init(self._init_conv_weight, self.conv1d.weight)
        self.norm.reset_parameters()
        self.proj_out.reset_parameters()
        with torch.no_grad():
            nn.init.ones_(self.dt_bias)
            call_init(self._init_decay, self.A_log)

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        shape = x.shape
        S = shape[-2]
        x = x.reshape(-1, S, shape[-1])
        k_dim = self.num_heads_k * self.channels_k_head
        v_dim = self.num_heads_v * self.channels_v_head

        qkv = self.proj_qkv(x)
        qkv = f.silu(self.conv1d(qkv.transpose(1, 2))[:, :, :S]).transpose(1, 2)
        q, k, v = qkv.split([k_dim, k_dim, v_dim], dim=-1)

        z = self.proj_z(x).reshape(-1, S, self.num_heads_v, self.channels_v_head)
        beta = self.proj_b(x).sigmoid()
        a = self.proj_a(x)
        g = -self.A_log.float().exp() * f.softplus(a.float() + self.dt_bias)

        q = q.reshape(-1, S, self.num_heads_k, self.channels_k_head)
        k = k.reshape(-1, S, self.num_heads_k, self.channels_k_head)
        v = v.reshape(-1, S, self.num_heads_v, self.channels_v_head)

        if self.num_heads_v // self.num_heads_k > 1:
            r = self.num_heads_v // self.num_heads_k
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
            z.reshape(-1, self.channels_v_head).float(),
        ).type_as(out)
        return self.proj_out(out.reshape(-1, S, v_dim)).reshape(*shape[:-1], -1)


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
    scale = 1.0 / float(dk**0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, H, -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
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
                -1,
                -2,
            )
            @ v_new
        )
    if not output_final_state:
        state = None
    out = out.reshape(B, H, -1, out.shape[-1])[:, :, :S]
    return out.transpose(1, 2).contiguous().to(dtype), state
