"""Feed-forward network with SwiGLU gating."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Any, Self, override

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)

import torch

from priml.math.activations import relu_squared
from priml.math.basic import ceil_multiple
from priml.math.custom_types import TensorFn
from priml.model.custom_types import ChannelsIn, ShardStyle, TensorModule
from priml.model.init import InitFn, kaiming_uniform, unit_fan_in_uniform
from priml.model.linear import Linear


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class SwiGLU(nn.Module):
    """Feed-forward network with optional SwiGLU gating.

    Uses a fused up_proj for gate+input (Muon-friendly: one matrix to
    orthogonalize). When gate=True, up_proj output is 2*channels_hidden
    and gets chunked into (gate, x).

    SwiGLU: out = down_proj(silu(gate) * x)  where gate, x = up_proj(input).chunk(2)
    Without gate: out = down_proj(silu(up_proj(input)))
    """

    class Config(Fig["SwiGLU"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        channels_hidden: int = -1
        """Hidden dimension (-1 to compute from channels_in * expansion)."""

        expansion: float = 8 / 3
        """Hidden-to-input channel ratio when channels_hidden is inferred."""

        round_to: int = 256
        """Round inferred channels_hidden up to nearest multiple of this."""

        gate: bool = True
        """Use SwiGLU gating (fused gate+input via 2x-wide up_proj)."""

        bias: bool = False
        """Include bias in linear projections."""

        split_gate_projection: bool = False
        """Run gate/up projections as separate matmuls.

        Keep this disabled for normal use: the fused projection is the
        loop-native path. The split path exists for HuggingFace parity tests,
        where matching HF's operation order avoids small floating-point drift.
        """

        norm: Makeable[TensorModule] | None = None
        """Optional norm inside the gate branch, for Muon compatibility.

        References:
          https://arxiv.org/abs/2601.19085
            Dillon, Joshua V. Speed is Confidence. 2026.
        """

        act: TensorFn = nn.functional.silu
        """Nonlinearity on the gate branch; ``norm`` requires it be ``silu``."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        init_weight: InitFn = kaiming_uniform
        """Weight init for ``up_proj``."""

        init_weight_out: InitFn | None = None
        """Weight init for ``down_proj``; ``None`` reuses ``init_weight``."""

        shard: ShardStyle | None = None
        """Tensor-parallel shard style over the mesh tp dim; ``None`` replicates."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.channels_hidden == -1:
                self.channels_hidden = int(
                    ceil_multiple(self.channels_in * self.expansion, self.round_to),
                )
            # Pushed here, not in __init__: the norm is finalized with the tree,
            # so a width written afterwards never reaches pprint or a diff.
            if isinstance(self.norm, ChannelsIn) and self.norm.channels_in == -1:
                self.norm.channels_in = self.channels_hidden
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        c_in = config.channels_in
        c_h = config.channels_hidden
        c_out = config.channels_out
        self.gate = config.gate
        self.depth = config.depth
        self.channels_hidden = c_h
        self.split_gate_projection = config.split_gate_projection
        self.shard = config.shard

        # Fused up_proj: output is 2*c_hidden when gated (gate + input in one matrix).
        self.up_proj = Linear.Config(
            channels_in=c_in,
            channels_out=(c_h * 2) if config.gate else c_h,
            bias=config.bias,
            depth=config.depth,
            init_weight=config.init_weight,
        ).make()
        self.down_proj = Linear.Config(
            channels_in=c_h,
            channels_out=c_out,
            bias=config.bias,
            depth=config.depth,
            init_weight=config.init_weight_out or config.init_weight,
        ).make()
        self.act = config.act
        if config.norm is None:
            self.norm = None
        elif not self.gate:
            raise ValueError("Norm can only be specified when gate is enabled.")
        elif self.act is not nn.functional.silu:
            raise ValueError("Norm can only be specified when act is silu.")
        else:
            if isinstance(config.norm, ChannelsIn) and config.norm.channels_in == -1:
                config.norm.channels_in = c_h
            self.norm = config.norm.make()

    def reset_parameters(self) -> None:
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
        if self.norm and hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        if self.gate:
            if self.split_gate_projection:
                gate, x = self._split_gate_projection(x)
            else:
                gate, x = self.up_proj(x).chunk(2, dim=-1)
            if self.norm is None:
                x = self.act(gate) * x
            else:
                # This exists for Muon compatibility as published in,
                #   https://arxiv.org/abs/2601.19085
                #   https://github.com/jvdillon/sic
                # Notice that when norm(x)=x then this path becomes the `else`,
                #   sigmoid(gate) * norm(gate * x) =
                #   = sigmoid(gate) * gate * x
                #   = silu(gate) * x
                x = torch.sigmoid(gate) * self.norm(gate * x)
        else:
            x = self.up_proj(x)
            x = self.act(x)
        return self.down_proj(x)

    def tensor_parallel_style(self) -> ParallelStyle:
        """Return the split-aligned ParallelStyle for tensor parallelism.

        The fused ``up_proj`` is column-sharded but its output is kept as a
        DTensor (``use_local_output=False``), so the ``chunk(2)`` in
        ``forward`` splits the *logical* tensor and gate/up halves stay
        aligned across ranks. ``down_proj`` is row-sharded.
        """
        if self.split_gate_projection:
            raise NotImplementedError(
                "SwiGLU tensor parallelism does not support "
                "split_gate_projection (it reads up_proj.weight directly).",
            )
        if self.norm is not None:
            raise NotImplementedError(
                "SwiGLU tensor parallelism does not support a gate norm "
                "(it normalizes over the sharded hidden dim).",
            )
        return _SwiGLUParallel()

    def _split_gate_projection(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Project gate/up separately while reusing the fused parameter layout."""
        c = self.channels_hidden
        w = self.up_proj.weight.to(x.dtype)
        bias = self.up_proj.bias
        b = bias.to(x.dtype) if bias is not None else None
        gate = torch.matmul(x, w[:c].T)
        up = torch.matmul(x, w[c:].T)
        if b is not None:
            gate = gate + b[:c]
            up = up + b[c:]
        return gate, up


class SwiGLUReluSquared(SwiGLU):
    """Ungated feed-forward with a squared-ReLU nonlinearity.

    One matrix in and one out, where gated SwiGLU is three: the nonlinearity
    carries what the gate otherwise would. Cheaper per parameter, and the shape
    the speedrun recipes settled on.

    Only the defaults differ from :class:`SwiGLU` -- no gate, ``relu**2`` for
    the activation, a hidden width that is a plain multiple of the input (no
    rounding), and a zero-initialized output projection, so a fresh block is the
    identity on its residual stream and the stack deepens as training proceeds.

    References:
        https://arxiv.org/abs/2109.08668
          So et al. Primer: Searching for Efficient Transformer for Language
          Modeling.

    """

    class Config(Makes["SwiGLUReluSquared"], SwiGLU.Config, kw_only=False):
        """:class:`SwiGLU.Config` re-defaulted; every field keeps its meaning."""

        gate: bool = False
        """Ungated: one matrix in, one out."""

        act: TensorFn = relu_squared
        """Carries the whole nonlinearity, in place of the gate."""

        expansion: float = 4.0
        """Hidden width as a multiple of ``channels_in``."""

        round_to: int = 1
        """Unrounded, so the hidden width is exactly ``expansion x channels_in``."""

        init_weight: InitFn = unit_fan_in_uniform
        """Input projection init."""

        init_weight_out: InitFn | None = nn.init.zeros_
        """Zeroed, so a fresh block is the identity on its residual stream."""


class _SwiGLUParallel(ParallelStyle):
    """Split-aligned tensor-parallel style for the fused SwiGLU block.

    Column-shards ``up_proj`` while keeping its output a DTensor sharded on
    the last dim, so the fused-gate ``chunk(2)`` in ``SwiGLU.forward`` stays
    aligned with the shard boundary (a local chunk would mis-pair gate/up).
    Row-shards ``down_proj`` to reduce-scatter back to the residual stream.
    """

    @override
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return parallelize_module(
            module,
            device_mesh,
            {
                "up_proj": ColwiseParallel(
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
                "down_proj": RowwiseParallel(input_layouts=Shard(-1)),
            },
        )
