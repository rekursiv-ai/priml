import torch

__all__ = [
    "inject_24",
    "marlin_24_workspace",
    "pack_to_marlin_24",
    "unpack_from_marlin_24",
]

def inject_24(
    w: torch.Tensor, size_k: int, size_n: int
) -> tuple[torch.Tensor, torch.Tensor]: ...
def marlin_24_workspace(
    out_features: int, min_thread_n: int = ..., max_parallel: int = ...
) -> torch.Tensor: ...
def pack_to_marlin_24(
    q_w_24: torch.Tensor, scales: torch.Tensor, num_bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def unpack_from_marlin_24(
    q_w_24_comp: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    original_shape: torch.Size,
    group_size: int,
    num_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
