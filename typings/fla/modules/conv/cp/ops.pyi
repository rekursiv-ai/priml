from fla.ops.cp import FLACPContext

import torch

class CausalConv1dFunctionCP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        chunk_indices: torch.Tensor | None,
        cp_context: FLACPContext | None,
        chunk_size: int | None,
        backend: str = ...,
    ): ...
    @staticmethod
    def backward(
        ctx, dy: torch.Tensor
    ) -> tuple[Tensor, Any | None, Any | None, None, None, None, None, None]: ...

def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = ...,
    activation: str | None = ...,
    chunk_indices: torch.Tensor | None = ...,
    cp_context: FLACPContext | None = ...,
    chunk_size: int | None = ...,
    backend: str = ...,
) -> Any | None: ...
