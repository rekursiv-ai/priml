from torch import Tensor
import torch

def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
) -> tuple[Tensor, Tensor | None]: ...
def naive_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = ...,
    scale: float = ...,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
) -> tuple[Tensor, Tensor | None]: ...
