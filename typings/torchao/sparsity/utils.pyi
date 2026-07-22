from torch.ao.quantization.observer import UniformQuantizationObserverBase

import torch

__all__ = [
    "PerChannelNormObserver",
    "create_block_sparse_tensor",
    "create_semi_structured_tensor",
    "mask_creator",
]

def create_block_sparse_tensor(M, N, blocksize, sparsity, dtype):  # -> Tensor:
    ...
def create_semi_structured_tensor(r, c, dtype):  # -> Tensor:
    ...

class PerChannelNormObserver(UniformQuantizationObserverBase):
    def __init__(self, **kwargs) -> None: ...
    def forward(self, x_orig): ...
    def calculate_qparams(self): ...

def mask_creator(tensor: torch.Tensor, N: int = ..., M: int = ...) -> torch.Tensor: ...
