from torch import Tensor
from typing import Any
from fla.ops.cp import FLACPContext

import torch

def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    safe_gate: bool = ...,
    lower_bound: float | None = ...,
    use_gate_in_kernel: bool = ...,
    A_log: torch.Tensor | None = ...,
    dt_bias: torch.Tensor | None = ...,
    disable_recompute: bool = ...,
    return_intermediate_states: bool = ...,
    cp_context: FLACPContext | None = ...,
    transpose_state_layout: bool = ...,
) -> tuple[
    Tensor,
    Tensor | None,
    Tensor | Any,
    Tensor | Any,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor,
]: ...
