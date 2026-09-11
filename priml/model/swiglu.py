"""Feed-forward network with SwiGLU gating."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Self, override

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)

from priml.math.basic import ceil_multiple
from priml.math.custom_types import TensorFn
from priml.model.custom_types import (
    ChannelsIn,
    DepthIndex,
    ShardStyle,
    TensorModule,
)
from priml.model.init import InitFn, unit_fan_in_uniform
from priml.model.linear import Linear


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def relu_squared(x: Tensor) -> Tensor:
    """Apply ``relu(x) ** 2``, with continuous derivative ``2 * relu(x)``.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``max(0, x) ** 2``, elementwise.

    References:
      https://arxiv.org/abs/2109.08668
        So et al. 2021, "Primer: Searching for Efficient Transformer for
        Language Modeling."

    """
    return nn.functional.relu(x).square()


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

        expansion: float = -1
        """Hidden-to-input ratio; -1 infers 8/3 when gated, otherwise 4."""

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
        """Optional norm inside the factored activation, for Muon compatibility.

        For act(x) = f(x) * x, normalize as f(g) * norm(g * x) when gated,
        or f(x) * norm(x) when ungated. Known factors are sigmoid for silu
        and relu for relu_squared.

        References:
          https://arxiv.org/abs/2601.19085
            Dillon, Joshua V. Speed is Confidence. 2026.
        """

        act: TensorFn = nn.functional.silu
        """Nonlinearity; ``norm`` requires a known silu or relu_squared factorization."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled initializers; empty means no scaling."""

        init_weight: InitFn = unit_fan_in_uniform
        """Weight init for ``up_proj``."""

        init_weight_out: InitFn = nn.init.zeros_
        """Weight init for ``down_proj``; zero keeps the residual stream unchanged."""

        shard: ShardStyle | None = None
        """Tensor-parallel shard style over the mesh tp dim; ``None`` replicates."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.expansion == -1:
                # ``4.0``, not ``4``: the field is declared ``float``, and an
                # int literal survives as an int through pprint -- which is a
                # golden-config diff, not a numeric one.
                self.expansion = 8 / 3 if self.gate else 4.0
            if self.channels_hidden == -1:
                self.channels_hidden = ceil_multiple(
                    self.channels_in * self.expansion, self.round_to
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
        self.depth_index = config.depth_index
        self.channels_hidden = c_h
        self.split_gate_projection = config.split_gate_projection
        self.shard = config.shard

        # Fused up_proj: output is 2*c_hidden when gated (gate + input in one matrix).
        self.up_proj = Linear.Config(
            channels_in=c_in,
            channels_out=(c_h * 2) if config.gate else c_h,
            bias=config.bias,
            depth_index=config.depth_index,
            init_weight=config.init_weight,
        ).make()
        self.down_proj = Linear.Config(
            channels_in=c_h,
            channels_out=c_out,
            bias=config.bias,
            depth_index=config.depth_index,
            init_weight=config.init_weight_out,
        ).make()
        self.act = config.act
        if config.norm is None:
            self.norm = None
        else:
            if self.act is nn.functional.silu:
                self.act = nn.functional.sigmoid
            elif self.act is relu_squared:
                self.act = nn.functional.relu
            else:
                raise ValueError("Norm requires act to be silu or relu_squared.")
            self.norm = config.norm.make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
        if self.norm is not None and hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if self.gate:
            if self.split_gate_projection:
                gate, x = self._split_gate_projection(x)
            else:
                gate, x = self.up_proj(x).chunk(2, dim=-1)
            if self.norm is None:
                x = self.act(gate) * x
            else:
                # SiLU grouping for Muon compatibility:
                #   https://arxiv.org/abs/2601.19085
                #   https://github.com/jvdillon/sic
                # Factor silu(g) = sigmoid(g)*g and relu_squared(g) = relu(g)*g
                # so identity normalization recovers the unnormalized branch.
                x = self.act(gate) * self.norm(gate * x)
        else:
            x = self.up_proj(x)
            x = self.act(x) if self.norm is None else self.act(x) * self.norm(x)
        return self.down_proj(x)

    def tensor_parallel_style(self) -> ParallelStyle:
        """Return the split-aligned ParallelStyle for tensor parallelism.

        The fused ``up_proj`` is column-sharded but its output is kept as a
        DTensor (``use_local_output=False``), so the ``chunk(2)`` in
        ``forward`` splits the *logical* tensor and gate/up halves stay
        aligned across ranks. ``down_proj`` is row-sharded.

        Returns:
          result: The ParallelStyle.

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
        gate = x @ w[:c].T
        up = x @ w[c:].T
        if b is not None:
            gate = gate + b[:c]
            up = up + b[c:]
        return gate, up


class SwiGLUReluSquared(SwiGLU):
    """Ungated feed-forward with a squared-ReLU nonlinearity.

    One matrix in and one out, where gated SwiGLU is three: the nonlinearity
    carries what the gate otherwise would. Cheaper per parameter, and the shape
    the speedrun recipes settled on.

    Only the gate and activation defaults differ from :class:`SwiGLU`:
    no gate and ``relu**2``. Both use rounded hidden widths and a zero-initialized
    output projection, so a fresh block leaves its residual stream unchanged.

    References:
        https://arxiv.org/abs/2109.08668
          So et al. Primer: Searching for Efficient Transformer for Language
          Modeling.

    """

    class Config(Makes["SwiGLUReluSquared"], SwiGLU.Config, kw_only=False):
        """:class:`SwiGLU.Config` re-defaulted; every field keeps its meaning."""

        _: KW_ONLY

        gate: bool = False
        """Ungated: one matrix in, one out."""

        act: TensorFn = relu_squared
        """Carries the whole nonlinearity, in place of the gate."""


class _SwiGLUParallel(ParallelStyle):
    """Split-aligned tensor-parallel style for the fused SwiGLU block.

    Column-shards ``up_proj`` while keeping its output a DTensor sharded on
    the last dim, so the fused-gate ``chunk(2)`` in ``SwiGLU.forward`` stays
    aligned with the shard boundary (a local chunk would mis-pair gate/up).
    Row-shards ``down_proj`` and all-reduces its output to replicated activations.
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
                "down_proj": RowwiseParallel(
                    input_layouts=Shard(-1),
                ),
            },
        )
