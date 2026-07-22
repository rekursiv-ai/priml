from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import torch

def chunk_fwd_mesa_net_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_CG_iteration: int = ...,
    chunk_size: int = ...,
    h_kk_init: torch.Tensor | None = ...,
    h_kv_init: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
def chunk_fwd_mesa_net_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    q_star: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_CG_iteration: int = ...,
    chunk_size: int = ...,
    h_kk_init: torch.Tensor | None = ...,
    h_kv_init: torch.Tensor | None = ...,
    dh_kv_final: torch.Tensor | None = ...,
    dh_kk_final: torch.Tensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...

class ChunkMesaNetFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        lamb,
        cu_seqlens,
        cu_seqlens_cpu,
        max_CG_iteration,
        h_kk_init,
        h_kv_init,
        output_final_state,
        use_qk_l2norm_in_kernel,
    ) -> tuple[Any, Any, Any]: ...
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx, do, dh_kk_final=..., dh_kv_final=...
    ) -> tuple[
        Tensor | Any,
        Tensor | Any,
        Any,
        Any,
        Any,
        Any,
        None,
        None,
        None,
        Any,
        Any,
        None,
        None,
    ]: ...

@torch.compiler.disable
def chunk_mesa_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    h_kk_init: torch.Tensor | None = ...,
    h_kv_init: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    max_CG_iteration: int = ...,
    use_qk_l2norm_in_kernel: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> tuple[Any, Any, Any]: ...
