import torch

def prepare_moba_chunks(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]: ...

class ParallelMoBAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlens,
        moba_q,
        moba_kv,
        moba_cu_seqlens_q,
        moba_cu_seqlens_k,
        max_seqlen,
        chunk_size,
        moba_q_sh_indices,
    ) -> Tensor: ...
    @staticmethod
    def backward(
        ctx, d_output
    ) -> tuple[
        Tensor, Tensor, Tensor, None, Tensor, Tensor, None, None, None, None, None
    ]: ...

def parallel_moba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    max_seqlen: int,
    chunk_size: int,
    topk: int,
) -> torch.Tensor: ...
