from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch

def chunk_gated_delta_product_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    num_householder: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
    chunk_indices_dp: torch.LongTensor | None = ...,
) -> tuple[Tensor | Any | None, Tensor | Any | None, Tensor, Tensor | Any, Any]: ...

class ChunkGatedDeltaProductFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        num_householder: int,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, Any]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do: torch.Tensor, dht: torch.Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | Any | None,
        Tensor,
        None,
        None,
        Tensor,
        None,
        None,
        None,
        None,
    ]: ...

@torch.compiler.disable
def chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> tuple[Any, Any]: ...
