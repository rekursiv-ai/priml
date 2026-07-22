import torch

__all__ = ["marlin_qqq_workspace", "pack_to_marlin_qqq", "unpack_from_marlin_qqq"]

def marlin_qqq_workspace(
    out_features: int, min_thread_n: int = ..., max_parallel: int = ...
) -> torch.Tensor: ...
def pack_to_marlin_qqq(
    q_w: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def unpack_from_marlin_qqq(
    q_w: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    original_shape: torch.Size,
    num_bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
