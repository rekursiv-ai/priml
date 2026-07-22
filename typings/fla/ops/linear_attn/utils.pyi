import torch

def normalize_with_z_state(
    o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    z_init: torch.Tensor | None,
    reverse: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
