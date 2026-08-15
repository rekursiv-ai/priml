"""Linear layer with configurable initialization."""

from __future__ import annotations

from dataclasses import KW_ONLY
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Self, override

from configgle import Fig
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle

import torch

from priml.model.init import InitFn, call_init, kaiming_uniform


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class Linear(nn.Linear):
    """Linear layer with configurable init and depth-aware scaling."""

    class Config(Fig["Linear"], kw_only=False):
        channels_in: int = -1
        """Number of input features (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output features (-1 to infer from channels_in)."""

        _: KW_ONLY

        bias: bool = False
        """Include bias in the linear layer."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function."""

        init_bias: InitFn = nn.init.zeros_
        """Bias initialization function."""

        shard: Literal["none", "colwise", "rowwise", "vocab"] = "none"
        """Tensor-parallel shard style over the mesh tp dim; none = replicated."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.depth = config.depth
        self._init_weight = config.init_weight
        self._init_bias = config.init_bias
        self.shard = config.shard
        super().__init__(
            in_features=config.channels_in,
            out_features=config.channels_out,
            bias=config.bias,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return super().forward(input)

    @override
    def reset_parameters(self) -> None:
        call_init(self._init_weight, self.weight, depth=self.depth)
        if self.bias is not None:
            call_init(self._init_bias, self.bias, depth=self.depth)


class EnsembleLinear(nn.Module):
    """Linear with ensemble dim for per-head orthogonalization.

    Weight shape: [num_ensemble, channels_out, channels_in]
    Output shape: [..., num_ensemble, channels_out]
    """

    class Config(Fig["EnsembleLinear"], kw_only=False):
        channels_in: int = -1
        """Number of input features."""

        channels_out: int = -1
        """Number of output features per ensemble member."""

        _: KW_ONLY

        num_ensemble: int = -1
        """Number of ensemble members (e.g. attention heads)."""

        bias: bool = False
        """Include bias in each ensemble member."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        init_weight: InitFn = kaiming_uniform
        """Weight initialization function (applied per ensemble member)."""

        shard: Literal["none", "colwise", "rowwise", "vocab"] = "none"
        """Tensor-parallel shard style over the mesh tp dim; none = replicated."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.shard = config.shard
        self.weight = nn.Parameter(
            torch.empty(config.num_ensemble, config.channels_out, config.channels_in),
        )
        if config.bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(config.num_ensemble, config.channels_out),
            )
        else:
            self.bias = None
        self.depth = config.depth
        self._init_weight = config.init_weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.weight.shape[0]):
            call_init(self._init_weight, self.weight.data[i], depth=self.depth)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        w = self.weight.to(x.dtype)
        # Not an einsum over [e, d, c]: bit-identical, but 15.7us against
        # 24.2us at [1, 2048, 512] -> 4 heads of 128.
        out = torch.matmul(x, w.reshape(-1, w.shape[-1]).T)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype).reshape(-1)
        return out.reshape(*x.shape[:-1], *self.weight.shape[:-1])

    def tensor_parallel_style(self) -> ParallelStyle:
        """Return the ensemble-dim ParallelStyle for tensor parallelism.

        The einsum ``"...c,edc->...ed"`` needs both operands to be DTensors
        (a naive ``Shard(0)`` weight over a plain input fails with a mixed
        Tensor/DTensor ``bmm``). The style shards the ensemble (head) dim of
        the weight, replicates the input, and leaves the output sharded on
        the ensemble dim so downstream per-head splits stay consistent.
        """
        return _EnsembleParallel()


class _EnsembleParallel(ParallelStyle):
    """Ensemble-dim tensor-parallel style for :class:`EnsembleLinear`.

    Shards the ensemble (head) axis of the ``[e, d, c]`` weight (and the
    ``[e, d]`` bias), replicates the input, and runs the einsum entirely over
    DTensor operands. The output is a DTensor sharded on the ensemble dim
    (``-2``) so a downstream per-head ``split``/``movedim`` stays aligned with
    the shard boundary.
    """

    @override
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for name, param in list(module.named_parameters(recurse=False)):
            module.register_parameter(
                name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
            )
        module.register_forward_pre_hook(partial(_replicate_input, device_mesh))
        return module


def _replicate_input(
    device_mesh: DeviceMesh,
    module: nn.Module,
    inputs: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Forward-pre-hook: coerce the einsum input to a replicated DTensor."""
    del module
    x = inputs[0]
    if isinstance(x, DTensor):
        x = x.redistribute(device_mesh, [Replicate()])
    else:
        x = DTensor.from_local(x, device_mesh, [Replicate()], run_check=False)
    return (x, *inputs[1:])
