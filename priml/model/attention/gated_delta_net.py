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

import torch
import torch.nn.functional

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
)
from priml.math import gated_delta_rule
from priml.math.basic import ceil_multiple
from priml.model.conv import conv_cost
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

        def uses_output_gate(self) -> bool:
            """Return whether this attention projects a post-delta gate."""
            return True

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the projections, the depthwise conv, and the recurrent scan.

            Matrix counts follow the 64-token PyTorch chunk implementation,
            including padding, constant initial state, and the final state update
            whose output is discarded. They do not qualify the fused CUDA kernel.
            Scalar scan counts estimate a decay, delta and weighted state update
            with its analytical derivative.
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
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            rows = seq_len * batch_size
            dt = dtype
            h = self.channels_in
            k_dim = self.num_heads_k * self.channels_k_head
            v_dim = self.num_heads_v * self.channels_v_head
            conv_dim = 2 * k_dim + v_dim
            projection_shapes = [
                (h, conv_dim),
                (h, self.num_heads_v),
                (h, self.num_heads_v),
                (v_dim, h),
            ]
            if self.uses_output_gate():
                projection_shapes.append((h, v_dim))
            projections = sum(
                (
                    matmul_cost(
                        channels_in=c_in,
                        channels_out=c_out,
                        rows=rows,
                        dtype=dt,
                    )
                    for c_in, c_out in projection_shapes
                ),
                Cost(),
            )
            conv = conv_cost(
                channels_in=conv_dim,
                channels_out=conv_dim,
                kernel_size=self.conv_kernel_size,
                ndim=1,
                groups=conv_dim,
                bias=False,
                input_grid=(seq_len,),
                batch_size=batch_size,
                stride=1,
                padding=self.conv_kernel_size - 1,
                dtype=dt,
            ) + elementwise_cost(
                primal=5 * conv_dim * rows,
                adjoint=5 * conv_dim * rows,
                channels=conv_dim,
                rows=rows,
                dtype=dt,
            )
            # Decay, delta, and weighted update over the state; the q/k L2 norms
            # each reduce their own row once in both primal and adjoint.
            state = self.num_heads_v * self.channels_k_head * self.channels_v_head
            norms = (
                (
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
                )
                .tile(2, copies=2)
                .tile(rows)
            )
            scan = (
                _chunk_matmul_cost(
                    seq_len=seq_len,
                    heads=self.num_heads_v,
                    key_width=self.channels_k_head,
                    value_width=self.channels_v_head,
                ).tile(batch_size)
                + elementwise_cost(
                    primal=rows
                    * (
                        2 * state
                        + 2 * v_dim
                        + self.num_heads_v * (6 * self.channels_k_head + 4)
                    ),
                    adjoint=rows
                    * (
                        4 * state
                        + 4 * v_dim
                        + self.num_heads_v * (10 * self.channels_k_head + 7)
                    ),
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
            # ``A_log`` is exponentiated once per invocation, so that work is shared.
            gates = _delta_gate_cost(
                rows=rows,
                heads=self.num_heads_v,
                dtype=dt,
            )
            # The norm runs once per value-head row; its parameters exist once.
            norm = cost(
                self.norm,
                seq_len=seq_len,
                batch_size=batch_size * self.num_heads_v,
                dtype=dtype,
                **kwargs,
            )
            return (
                projections
                + conv
                + scan
                + gates
                + norm
                + self._output_gate_cost(rows=rows, dtype=dt)
            )

        def _output_gate_cost(self, *, rows: int, dtype: torch.dtype | None) -> Cost:
            """Cost the separate post-norm SiLU and product."""
            width = self.num_heads_v * self.channels_v_head
            return elementwise_cost(
                primal=6 * width * rows,
                adjoint=7 * width * rows,
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
        self.proj_z: Linear | None = None
        if config.uses_output_gate():
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
        if self.proj_z is not None:
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
        qkv = torch.nn.functional.silu(
            self.conv1d(qkv.transpose(1, 2))[:, :, :S],
        ).transpose(1, 2)
        q, k, v = qkv.split([k_dim, k_dim, v_dim], dim=-1)

        if self.proj_z is None:
            raise ValueError("GatedDeltaNet requires an output gate projection.")
        z = self.proj_z(x).reshape(-1, S, self.num_heads_v, self.channels_v_head)
        beta = self.proj_b(x).sigmoid()
        a = self.proj_a(x)
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(
            a.float() + self.dt_bias,
        )

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
            out, _ = gated_delta_rule.chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        out = self.norm(
            out.reshape(-1, self.channels_v_head),
        ) * torch.nn.functional.silu(
            z.reshape(-1, self.channels_v_head).float(),
        ).type_as(out)
        return self.proj_out(out.reshape(-1, S, v_dim)).reshape(*shape[:-1], -1)


def _chunk_matmul_cost(
    *,
    seq_len: int,
    heads: int,
    key_width: int,
    value_width: int,
) -> Cost:
    chunk = 64
    chunks = int(ceil_multiple(seq_len, chunk)) // chunk
    # solve_triangular contributes primal products but dispatches no backward matmuls.
    shapes = [
        (chunk, key_width, chunk, 2),
        (chunk, chunk, value_width, 0),
        (chunk, chunk, key_width, 0),
    ] * chunks
    for index in range(chunks):
        state_gradients = 1 if index == 0 else 2
        shapes.extend(
            [
                (chunk, key_width, chunk, 2),
                (chunk, key_width, value_width, state_gradients),
                (chunk, key_width, value_width, state_gradients),
                (chunk, chunk, value_width, 2),
                (key_width, chunk, value_width, 0 if index == chunks - 1 else 2),
            ],
        )
    total = Cost()
    for rows, inner, columns, adjoints in shapes:
        primal = matmul_cost(
            channels_in=inner,
            channels_out=columns,
            weight=False,
            rows=rows,
            dtype=torch.float32,
        ).only("primal")
        total += primal + primal.relabel("adjoint").tile(adjoints)
    return total.tile(heads)


def _delta_gate_cost(*, rows: int, heads: int, dtype: torch.dtype | None) -> Cost:
    """Count the gate map and its two parameter-gradient reductions exactly."""
    dt = resolve_dtype(dtype)
    itemsize = dt.itemsize
    params = 2 * heads
    return Cost(
        cells={
            ("flops", "primal", "elementwise", dt): (9 * rows + 2) * heads,
            ("flops", "adjoint", "elementwise", dt): 9 * rows * heads,
            ("flops", "adjoint", "reduction", dt): params * (rows - 1),
            ("bytes", "primal", "elementwise", dt): itemsize
            * (6 * rows * heads + params),
            ("bytes", "adjoint", "elementwise", dt): itemsize
            * (9 * rows * heads + params * rows + params),
            ("bytes", "adjoint", "reduction", dt): itemsize * (params * rows + params),
        },
        params=params,
        params_active=params,
    )
