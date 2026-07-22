from fla.ops.backends import BaseBackend

import torch

"""TileLang backend for common chunk operations.

Enabled by default on Hopper (sm90+) with Triton >= 3.4.0 to work around
hardware-specific regressions (see #640). Can also be forced via FLA_TILELANG=1.
"""

class TileLangBackend(BaseBackend):
    backend_type = ...
    package_name = ...
    env_var = ...
    @classmethod
    def is_available(cls) -> bool: ...
    def chunk_bwd_dqkwg_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        w: torch.Tensor | None = ...,
        g: torch.Tensor | None = ...,
        g_gamma: torch.Tensor | None = ...,
        dv: torch.Tensor | None = ...,
        scale: float | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_size: int = ...,
        chunk_indices: torch.LongTensor | None = ...,
        use_exp2: bool = ...,
        transpose_state_layout: bool = ...,
    ) -> tuple[bool, str | None]: ...
    def chunk_bwd_dqkwg(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        w: torch.Tensor | None = ...,
        g: torch.Tensor | None = ...,
        g_gamma: torch.Tensor | None = ...,
        dv: torch.Tensor | None = ...,
        scale: float | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_size: int = ...,
        chunk_indices: torch.LongTensor | None = ...,
        use_exp2: bool = ...,
        transpose_state_layout: bool = ...,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]: ...
    def parallel_attn_fwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        sink_bias: torch.Tensor | None,
        scale: float,
        window_size: int | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> tuple[bool, str | None]: ...
    def parallel_attn_fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        sink_bias: torch.Tensor | None,
        scale: float,
        window_size: int | None = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def parallel_attn_bwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        lse: torch.Tensor,
        do: torch.Tensor,
        sink_bias: torch.Tensor | None = ...,
        scale: float | None = ...,
        window_size: int | None = ...,
        chunk_size: int = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> tuple[bool, str | None]: ...
    def parallel_attn_bwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        g_cumsum: torch.Tensor | None,
        lse: torch.Tensor,
        do: torch.Tensor,
        sink_bias: torch.Tensor | None = ...,
        scale: float | None = ...,
        window_size: int | None = ...,
        chunk_size: int = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]: ...
