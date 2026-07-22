from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4, 8, 16, 32]],
    key=["D", "PADDING_SIDE", "PACK"],
    **autotune_cache_kwargs,
)
@triton.jit
def packunpack_sequence_kernel(
    x,
    y,
    cu_seqlens,
    S,
    D,
    BD: tl.constexpr,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
) -> None: ...
def pack_sequence_fwdbwd(
    x: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str
) -> torch.Tensor: ...
def unpack_sequence_fwdbwd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    desired_shape: torch.Size,
) -> torch.Tensor: ...

class PackSequenceFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str
    ) -> torch.Tensor: ...
    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]: ...

class UnpackSequenceFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        padding_side: str,
        desired_shape: torch.Size | None = ...,
    ) -> torch.Tensor: ...
    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]: ...

def pack_sequence(
    x: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str = ...
) -> torch.Tensor: ...
def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = ...,
    desired_shape: torch.Size | None = ...,
) -> torch.Tensor: ...
