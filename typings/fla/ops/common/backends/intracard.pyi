from fla.ops.backends import BaseBackend

import torch

"""Intra-card CP backend for shared delta rule operations.

Accelerates prefill by splitting long sequences into sub-sequences
and processing them in parallel across SMs.

Only active under torch.inference_mode() with varlen (cu_seqlens != None).
"""
MAX_SUBSEQS = ...

class IntraCardCPBackend(BaseBackend):
    @classmethod
    def is_available(cls) -> bool: ...
    def chunk_gated_delta_rule_fwd_h_verifier(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = ...,
        gk: torch.Tensor | None = ...,
        initial_state: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        chunk_size: int = ...,
        save_new_value: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
        use_exp2: bool = ...,
        transpose_state_layout: bool = ...,
    ) -> tuple[bool, str | None]: ...
    def chunk_gated_delta_rule_fwd_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = ...,
        gk: torch.Tensor | None = ...,
        initial_state: torch.Tensor | None = ...,
        output_final_state: bool = ...,
        chunk_size: int = ...,
        save_new_value: bool = ...,
        cu_seqlens: torch.LongTensor | None = ...,
        cu_seqlens_cpu: torch.LongTensor | None = ...,
        chunk_indices: torch.LongTensor | None = ...,
        use_exp2: bool = ...,
        transpose_state_layout: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
